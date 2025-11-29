"""
PAPN Adapter Module for integrating PAPN-style contrastive learning
into existing RD-LGC architecture

This module wraps the MultiProjectionLayer to add:
1. Part prototype extraction
2. MoCo-style momentum encoder
3. Global + Part feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.papn_loss import PartPrototypeExtractor


class PAPNProjectionAdapter(nn.Module):
    """
    Adapter that wraps MultiProjectionLayer with PAPN capabilities

    This adds:
    - Part prototype extractors for each projection level
    - Feature fusion (global + part)
    - Compatible interface with existing RD-LGC pipeline
    """

    def __init__(
        self,
        base_projection_layer,
        n_parts=5,
        feature_dims=None,
        enable_fusion=True
    ):
        """
        Args:
            base_projection_layer: existing MultiProjectionLayer instance
            n_parts: number of part prototypes
            feature_dims: list of feature dimensions for each level [256, 512, 1024]
            enable_fusion: whether to fuse global + part features
        """
        super().__init__()

        self.base_projection = base_projection_layer
        self.n_parts = n_parts
        self.enable_fusion = enable_fusion

        # Default feature dimensions for ResNet stages
        if feature_dims is None:
            feature_dims = [256, 512, 1024]  # layer1, layer2, layer3

        # Create part extractors for each level
        self.part_extractors = nn.ModuleList([
            PartPrototypeExtractor(n_parts=n_parts, feature_dim=dim)
            for dim in feature_dims
        ])

        # Fusion layers to combine global + part features
        if enable_fusion:
            self.fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim * 2, dim, kernel_size=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
                for dim in feature_dims
            ])
        else:
            self.fusion_layers = None

    def forward(self, features, return_parts=False):
        """
        Forward pass with PAPN enhancement

        Args:
            features: list of [N, C, H, W] feature maps from backbone
            return_parts: if True, return part features for loss computation

        Returns:
            if return_parts:
                projected_features: list of projected features
                part_features: list of [N, C] part features
                global_features: list of [N, C] global features
            else:
                projected_features: list of projected features
        """
        # First apply base projection
        projected = self.base_projection(features)

        if not self.enable_fusion:
            # Just return projected features without fusion
            if return_parts:
                # Extract parts from original features for loss
                part_feats = []
                global_feats = []
                for feat, extractor in zip(features, self.part_extractors):
                    g_feat = F.adaptive_avg_pool2d(feat, 1).squeeze()
                    if g_feat.dim() == 1:
                        g_feat = g_feat.unsqueeze(0)
                    p_feat, _ = extractor(feat)
                    global_feats.append(g_feat)
                    part_feats.append(p_feat)
                return projected, part_feats, global_feats
            return projected

        # Extract global and part features, then fuse
        fused_features = []
        part_features = []
        global_features = []

        for i, (feat, extractor, fusion) in enumerate(
            zip(projected, self.part_extractors, self.fusion_layers)
        ):
            N, C, H, W = feat.shape

            # Global feature: already have from projection
            # Just need to expand back to spatial
            global_feat = F.adaptive_avg_pool2d(feat, 1)  # [N, C, 1, 1]
            global_feat_spatial = global_feat.expand(-1, -1, H, W)  # [N, C, H, W]

            # Part feature: extract using prototypes
            part_feat, _ = extractor(feat)  # [N, C]
            part_feat_spatial = part_feat.view(N, C, 1, 1).expand(-1, -1, H, W)

            # Concatenate and fuse
            concat_feat = torch.cat([global_feat_spatial, part_feat_spatial], dim=1)
            # [N, 2*C, H, W]

            fused_feat = fusion(concat_feat)  # [N, C, H, W]
            fused_features.append(fused_feat)

            if return_parts:
                # Store for loss computation
                global_features.append(global_feat.squeeze())
                part_features.append(part_feat)

        if return_parts:
            return fused_features, part_features, global_features

        return fused_features

    def get_orthogonal_loss(self):
        """
        Get orthogonal regularization loss from all part extractors

        Returns:
            total orthogonal loss
        """
        total_loss = 0
        for extractor in self.part_extractors:
            total_loss += extractor.get_orthogonal_loss()
        return total_loss / len(self.part_extractors)

    @torch.no_grad()
    def update_prototypes_orthogonal(self):
        """
        Project all prototypes to orthonormal manifold
        """
        for extractor in self.part_extractors:
            extractor.project_to_orthonormal()


class PAPNEnhancedRDLGC(nn.Module):
    """
    Enhanced RDLGC with PAPN capabilities

    This is a wrapper around existing RDLGC that adds:
    1. Part prototype extraction
    2. Enhanced feature fusion
    3. Compatible training interface
    """

    def __init__(
        self,
        base_rdlgc_model,
        n_parts=5,
        feature_dims=None,
        enable_fusion=True
    ):
        """
        Args:
            base_rdlgc_model: existing RDLGC model instance
            n_parts: number of part prototypes
            feature_dims: list of feature dimensions
            enable_fusion: whether to enable global+part fusion
        """
        super().__init__()

        self.base_model = base_rdlgc_model

        # Wrap projection layers with PAPN adapter
        self.papn_adapter_q = PAPNProjectionAdapter(
            base_projection_layer=base_rdlgc_model.proj_layer,
            n_parts=n_parts,
            feature_dims=feature_dims,
            enable_fusion=enable_fusion
        )

        self.papn_adapter_k = PAPNProjectionAdapter(
            base_projection_layer=base_rdlgc_model.proj_layer_momentum,
            n_parts=n_parts,
            feature_dims=feature_dims,
            enable_fusion=enable_fusion
        )

        # Copy prototype extractors from q to k
        self._copy_prototypes()

        # Freeze k adapter prototypes (momentum update)
        for param in self.papn_adapter_k.parameters():
            param.requires_grad = False

    def _copy_prototypes(self):
        """Copy prototype weights from query to key adapter"""
        for extractor_q, extractor_k in zip(
            self.papn_adapter_q.part_extractors,
            self.papn_adapter_k.part_extractors
        ):
            extractor_k.part_proto.data = extractor_q.part_proto.data.clone()

    @torch.no_grad()
    def update_momentum_encoder(self, momentum=0.999):
        """
        Update momentum encoder including:
        1. Base projection layer (already handled by base model)
        2. Part prototype extractors
        """
        # Update base model momentum encoder
        self.base_model.update_momentum_encoder()

        # Update PAPN prototypes
        for extractor_q, extractor_k in zip(
            self.papn_adapter_q.part_extractors,
            self.papn_adapter_k.part_extractors
        ):
            extractor_k.part_proto.data = (
                extractor_k.part_proto.data * momentum +
                extractor_q.part_proto.data * (1 - momentum)
            )

    def train_forward(self, imgs, aug_imgs, return_parts=True):
        """
        Training forward pass with PAPN enhancement

        Args:
            imgs: input images
            aug_imgs: augmented images
            return_parts: whether to return part features for PAPN loss

        Returns:
            Same as RDLGC but with additional part features
        """
        # Get base features from teacher network
        feats_t = self.base_model.net_t(imgs)
        feats_k = self.base_model.net_t(aug_imgs)

        # Apply PAPN-enhanced projection
        if return_parts:
            feats_t_q_grid, part_feats_q, global_feats_q = self.papn_adapter_q(
                feats_t, return_parts=True
            )
            with torch.no_grad():
                feats_t_k_grid, part_feats_k, global_feats_k = self.papn_adapter_k(
                    feats_k, return_parts=True
                )
        else:
            feats_t_q_grid = self.papn_adapter_q(feats_t, return_parts=False)
            with torch.no_grad():
                feats_t_k_grid = self.papn_adapter_k(feats_k, return_parts=False)
            part_feats_q = part_feats_k = None
            global_feats_q = global_feats_k = None

        # Continue with rest of RDLGC pipeline
        feats_t_q = [f.detach() for f in feats_t]
        feats_t_k = [f.detach() for f in feats_k]

        # Add noise (same as original)
        add_noise = torch.randn(1)
        if add_noise > 0.5 and self.training:
            for i in range(len(feats_t_q_grid)):
                noise = torch.randn_like(feats_t_q_grid[i]).to(imgs.device)
                B, C, H, W = feats_t_q_grid[i].shape
                mask = torch.randint(0, 2, (B, 1, H, W)).to(imgs.device)
                feats_t_q_grid[i] = feats_t_q_grid[i] + noise * mask

        # MFF-OCE and student network
        mid = self.base_model.mff_oce(feats_t_q_grid)
        mid_k = self.base_model.mff_oce(feats_t_k_grid)
        feats_s = self.base_model.net_s(mid)

        # Global features for SCL loss
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze()
        glo_feats_k = F.adaptive_avg_pool2d(mid_k, 1).squeeze()

        # Return format compatible with trainer
        return {
            'feats_t_q': feats_t_q,
            'feats_s': feats_s,
            'feats_t_k': feats_t_k,
            'feats_t_q_grid': feats_t_q_grid,
            'feats_t_k_grid': feats_t_k_grid,
            'glo_feats': glo_feats,
            'glo_feats_k': glo_feats_k,
            'part_feats_q': part_feats_q,
            'part_feats_k': part_feats_k,
            'global_feats_q': global_feats_q,
            'global_feats_k': global_feats_k
        }

    def forward(self, imgs, aug_imgs=None):
        """
        Forward pass - dispatches to train or eval mode
        """
        if self.training and aug_imgs is not None:
            return self.train_forward(imgs, aug_imgs)

        # Eval mode: use base model
        feats_t = self.base_model.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        feats = self.papn_adapter_q(feats_t, return_parts=False)
        mid = self.base_model.mff_oce(feats)
        feats_s = self.base_model.net_s(mid)

        return {
            'feats_t_q': feats_t,
            'feats_s': feats_s,
            'feats_t_k': None,
            'feats_t_q_grid': None,
            'feats_t_k_grid': None,
            'glo_feats': None,
            'glo_feats_k': None
        }

    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        self.base_model.train(mode)
        self.papn_adapter_q.train(mode)
        # papn_adapter_k is always in eval mode (momentum encoder)
        self.papn_adapter_k.eval()
        return self

    def eval(self):
        """Set evaluation mode"""
        return self.train(False)

"""
RD model with K-values and Sigmoid Activation
Modified version that applies k-values and activation function to features before computing loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import Bottleneck

from model import get_model, MODEL
from model.rd import MFF_OCE, MultiProjectionLayer


class RDWithKVal(nn.Module):
    """
    RD model with K-values and activation function support

    Modifications:
    1. Loads k-values from stats_config
    2. Applies activation(features * k) before computing loss
    """
    def __init__(self, model_t, model_s, stats_config=None):
        super(RDWithKVal, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = get_model(model_s)

        self.frozen_layers = ['net_t']
        self.stats_config = stats_config

        # Parse stats_config
        if stats_config:
            self.activation_type = stats_config.get('activation_type', 'sigmoid').lower()
            k_list = stats_config.get('k_values_272', None)

            # Determine number of channels (2048 for wide_resnet50_2 after mff_oce)
            expected_channels = 2048

            if k_list is not None and len(k_list) == expected_channels:
                k_tensor = torch.tensor(k_list, dtype=torch.float32)
                print(f"[RDWithKVal] Loaded k-values: {len(k_list)} channels")
            else:
                k_tensor = torch.ones(expected_channels, dtype=torch.float32)
                print(f"[RDWithKVal] Using default k-values (all ones): {expected_channels} channels")
        else:
            self.activation_type = 'none'
            k_tensor = torch.ones(2048, dtype=torch.float32)
            print("[RDWithKVal] No stats_config provided, using default settings")

        # Register k-values as buffer (not trainable)
        self.register_buffer('channel_k_values', k_tensor)

    def _get_activation_fn(self):
        """Get activation function based on activation_type"""
        if self.activation_type == 'sigmoid':
            return torch.sigmoid
        elif self.activation_type == 'tanh':
            return torch.tanh
        elif self.activation_type == 'arctan':
            return torch.atan
        else:
            return lambda x: x  # Identity function

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        # Teacher features
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]

        # Merge features through MFF_OCE
        mid = self.mff_oce(feats_t)

        # Student features
        feats_s = self.net_s(mid)

        # Global features
        glb_feats = F.adaptive_avg_pool2d(mid, 1).squeeze()

        # Apply k-values and activation if configured
        if self.stats_config and self.activation_type != 'none':
            activation_fn = self._get_activation_fn()
            k_spatial = self.channel_k_values.view(1, -1, 1, 1)

            # Apply to teacher features (after mff_oce, which is mid)
            mid_activated = activation_fn(mid * k_spatial)

            # Apply to student features
            feats_s_activated = []
            for feat_s in feats_s:
                C = feat_s.shape[1]
                k_spatial_s = self.channel_k_values[:C].view(1, -1, 1, 1)
                feat_s_activated = activation_fn(feat_s * k_spatial_s)
                feats_s_activated.append(feat_s_activated)

            # Replace with activated features
            # Note: For loss computation, we'll use activated features
            feats_t_activated = [mid_activated]  # Use mid as reference
            return feats_t_activated, feats_s_activated, glb_feats

        return feats_t, feats_s, glb_feats


class RDLGCWithKVal(nn.Module):
    """
    RD-LGC model with K-values and activation function support
    """
    def __init__(self, model_t, model_s, dp=False, stats_config=None):
        super(RDLGCWithKVal, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.proj_layer = MultiProjectionLayer(base=64, dp=dp)
        self.net_s = get_model(model_s)

        self.frozen_layers = ['net_t']
        self.stats_config = stats_config

        # Parse stats_config
        if stats_config:
            self.activation_type = stats_config.get('activation_type', 'sigmoid').lower()
            k_list = stats_config.get('k_values_272', None)

            expected_channels = 2048

            if k_list is not None and len(k_list) == expected_channels:
                k_tensor = torch.tensor(k_list, dtype=torch.float32)
                print(f"[RDLGCWithKVal] Loaded k-values: {len(k_list)} channels")
            else:
                k_tensor = torch.ones(expected_channels, dtype=torch.float32)
                print(f"[RDLGCWithKVal] Using default k-values (all ones): {expected_channels} channels")
        else:
            self.activation_type = 'none'
            k_tensor = torch.ones(2048, dtype=torch.float32)
            print("[RDLGCWithKVal] No stats_config provided, using default settings")

        self.register_buffer('channel_k_values', k_tensor)

    def _get_activation_fn(self):
        """Get activation function based on activation_type"""
        if self.activation_type == 'sigmoid':
            return torch.sigmoid
        elif self.activation_type == 'tanh':
            return torch.tanh
        elif self.activation_type == 'arctan':
            return torch.atan
        else:
            return lambda x: x

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def apply_activation_to_features(self, feats):
        """Apply k-values and activation to feature list"""
        if self.stats_config and self.activation_type != 'none':
            activation_fn = self._get_activation_fn()
            activated_feats = []

            for feat in feats:
                C = feat.shape[1]
                k_spatial = self.channel_k_values[:C].view(1, -1, 1, 1).to(feat.device)
                feat_activated = activation_fn(feat * k_spatial)
                activated_feats.append(feat_activated)

            return activated_feats
        return feats

    def train_forward(self, imgs, aug_imgs):
        feats_t = self.net_t(imgs)
        feats_k = self.net_t(aug_imgs)
        feats_t_q_grid = self.proj_layer(feats_t)
        feats_t_k_grid = self.proj_layer(feats_k)

        feats_t_q = [f.detach() for f in feats_t]
        feats_t_k = [f.detach() for f in feats_k]

        add_noise = torch.randn(1)
        if add_noise > 0.5 and self.training:
            for i in range(len(feats_t_q)):
                noise = torch.randn_like(feats_t_q[i]).to(imgs.device)
                B, C, H, W = feats_t_q[i].shape
                mask = torch.randint(0, 2, (B, 1, H, W)).to(imgs.device)
                feats_t_q_grid[i] += noise * mask

        mid = self.mff_oce(feats_t_q_grid)
        mid_k = self.mff_oce(feats_t_k_grid)
        feats_s = self.net_s(mid)
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze()
        glo_feats_k = F.adaptive_avg_pool2d(mid_k, 1).squeeze()

        # Apply activation to projected features for comparison
        feats_t_q_grid_act = self.apply_activation_to_features(feats_t_q_grid)
        feats_t_k_grid_act = self.apply_activation_to_features(feats_t_k_grid)

        return feats_t_q_grid_act, feats_s, feats_t_k, feats_t_q_grid, feats_t_k_grid_act, glo_feats, glo_feats_k

    def forward(self, imgs, aug_imgs=None):
        if self.training:
            return self.train_forward(imgs, aug_imgs)

        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        feats = self.proj_layer(feats_t)
        mid = self.mff_oce(feats)
        feats_s = self.net_s(mid)

        glo_feats = None
        glo_feats_k = None
        feats_t_k = None
        feats_t_q_grid = self.apply_activation_to_features(feats)
        feats_t_k_grid = None

        return feats_t_q_grid, feats_s, feats_t_k, feats_t_q_grid, feats_t_k_grid, glo_feats, glo_feats_k


@MODEL.register_module
def rd_with_kval(pretrained=False, **kwargs):
    """RD model with K-values support"""
    model = RDWithKVal(**kwargs)
    return model


@MODEL.register_module
def rd_lgc_with_kval(pretrained=False, **kwargs):
    """RD-LGC model with K-values support"""
    model = RDLGCWithKVal(**kwargs)
    return model

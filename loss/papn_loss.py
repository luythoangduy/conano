"""
PAPN-Style Prototype Contrastive Loss with MoCo Queue - Version 2
Complete MoCo mechanism with separate query/key branches

Key improvements:
1. Separate prototype extractors for query and key (momentum)
2. Separate projectors for query and key (momentum)
3. Proper momentum update for both prototypes and projectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOSS


def generate_orthonormal_vectors(n, dim, device='cuda'):
    """Generate n orthonormal vectors using SVD"""
    A = torch.randn(dim, n, device=device)
    U, S, Vt = torch.svd(A)
    return U.T[:n].contiguous()


@torch.no_grad()
def concat_all_gather(tensor):
    """All_gather for distributed training"""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class PartPrototypeExtractor(nn.Module):
    """Extract part features using learnable prototypes"""

    def __init__(self, n_parts=5, feature_dim=2048):
        super().__init__()
        self.n_parts = n_parts
        self.feature_dim = feature_dim

        # Initialize prototypes as orthonormal vectors
        self.part_proto = nn.Parameter(
            generate_orthonormal_vectors(n_parts, feature_dim)
        )

    def forward(self, feat):
        """
        Args:
            feat: [N, C, H, W] feature map
        Returns:
            feat_part: [N, C] aggregated part feature
        """
        N, C, H, W = feat.shape
        M = self.n_parts

        # Flatten and normalize
        feat_flat = feat.flatten(2).permute(0, 2, 1)  # [N, H*W, C]
        feat_norm = F.normalize(feat_flat, dim=-1)
        proto_norm = F.normalize(self.part_proto, dim=-1)  # [M, C]

        # Compute similarity (attention)
        similarity = feat_norm @ proto_norm.T  # [N, H*W, M]
        similarity = similarity.permute(0, 2, 1).reshape(N, M, H, W)  # [N, M, H, W]

        # Weighted aggregation
        feat_parts = similarity.unsqueeze(2) * feat.unsqueeze(1)  # [N, M, C, H, W]
        feat_parts = feat_parts.flatten(3).sum(-1)  # [N, M, C]

        # Average across prototypes
        feat_part = feat_parts.mean(dim=1)  # [N, C]

        return feat_part


@LOSS.register_module
class PAPNMoCoLoss(nn.Module):
    """
    PAPN-style Prototype Contrastive Loss with proper MoCo mechanism

    Architecture:
        Query Branch (learnable):
            - part_extractor_q: learnable prototypes
            - projector_q: learnable projection head

        Key Branch (momentum):
            - part_extractor_k: momentum prototypes
            - projector_k: momentum projection head

    Args:
        feature_dim: input feature dimension (e.g., 1024)
        proj_dim: projection dimension (e.g., 256)
        n_parts: number of part prototypes (default: 5)
        queue_size: MoCo queue size (default: 4096)
        momentum: momentum coefficient (default: 0.999)
        temperature: temperature for InfoNCE (default: 0.1)
        lam: loss weight (default: 1.0)
    """

    def __init__(
        self,
        feature_dim=1024,
        proj_dim=256,
        n_parts=5,
        queue_size=4096,
        momentum=0.999,
        temperature=0.15,
        enforce_orthogonal=False,
        ortho_loss_weight=0.1,
        lam=1.0
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.n_parts = n_parts
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.lam = lam

        # ========== Query Branch (learnable) ==========
        self.part_extractor_q = PartPrototypeExtractor(n_parts, feature_dim)

        self.projector_q = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

        # ========== Key Branch (momentum) ==========
        self.part_extractor_k = PartPrototypeExtractor(n_parts, feature_dim)

        self.projector_k = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

        # Copy params from query to key
        self.part_extractor_k.part_proto.data.copy_(self.part_extractor_q.part_proto.data)
        self.projector_k.load_state_dict(self.projector_q.state_dict())

        # Freeze key branch
        for param in self.part_extractor_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        # ========== MoCo Queue ==========
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update for key branch:
            θ_k = m * θ_k + (1 - m) * θ_q
        """
        # Update prototypes
        for param_q, param_k in zip(
            self.part_extractor_q.parameters(),
            self.part_extractor_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        # Update projector
        for param_q, param_k in zip(
            self.projector_q.parameters(),
            self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys"""
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        """
        Forward pass

        Args:
            q_b, k_b: backbone features (not used, for API compatibility)
            q_grid: [N, C, H, W] or list - query features
            k_grid: [N, C, H, W] or list - key features
            labels: [N] class labels (not used)

        Returns:
            loss: scalar loss value
        """
        # Handle list input
        if isinstance(q_grid, list):
            q_feat = q_grid[-1]
        else:
            q_feat = q_grid

        if isinstance(k_grid, list):
            k_feat = k_grid[-1]
        else:
            k_feat = k_grid

        # ========== Query Branch ==========
        q_part = self.part_extractor_q(q_feat)  # [N, C]
        q_embed = self.projector_q(q_part)      # [N, proj_dim]
        q_embed = F.normalize(q_embed, dim=1)

        # ========== Key Branch (no grad) ==========
        with torch.no_grad():
            # Momentum update
            self._momentum_update()

            # Extract key features
            k_part = self.part_extractor_k(k_feat)  # [N, C]
            k_embed = self.projector_k(k_part)      # [N, proj_dim]
            k_embed = F.normalize(k_embed, dim=1)

        # ========== Contrastive Loss ==========
        # Positive: same image
        l_pos = torch.einsum('nc,nc->n', [q_embed, k_embed]).unsqueeze(-1)  # [N, 1]

        # Negative: queue
        l_neg = torch.einsum('nc,ck->nk', [q_embed, self.queue.clone().detach()])  # [N, K]

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]
        logits /= self.temperature

        # Labels (positive at index 0)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Loss
        loss = self.criterion(logits, targets)

        # Update queue
        self._dequeue_and_enqueue(k_embed)

        return loss * self.lam
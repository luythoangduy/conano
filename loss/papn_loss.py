"""
PAPN-Style Prototype Contrastive Loss with MoCo Queue - Version 3
Part-level queue for fine-grained contrastive learning

Key improvements:
1. Separate prototype extractors for query and key (momentum)
2. Separate projectors for query and key (momentum)
3. Proper momentum update for both prototypes and projectors
4. Part-level queue: each part has its own queue for fine-grained matching
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

    def forward(self, feat, return_parts=False):
        """
        Args:
            feat: [N, C, H, W] feature map
            return_parts: if True, return [N, M, C]; else return [N, C]
        Returns:
            feat_parts: [N, M, C] individual part features, or
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

        if return_parts:
            return feat_parts  # [N, M, C]
        else:
            # Average across prototypes
            feat_part = feat_parts.mean(dim=1)  # [N, C]
            return feat_part


@LOSS.register_module
class PAPNMoCoLoss(nn.Module):
    """
    PAPN-style Prototype Contrastive Loss with part-level MoCo queue

    Architecture:
        Query Branch (learnable):
            - part_extractor_q: learnable prototypes
            - projector_q: learnable projection head (shared across parts)

        Key Branch (momentum):
            - part_extractor_k: momentum prototypes
            - projector_k: momentum projection head (shared across parts)

        Queue: [proj_dim, n_parts, queue_size]
            - Each part maintains its own queue

    Args:
        feature_dim: input feature dimension (e.g., 1024)
        proj_dim: projection dimension (e.g., 256)
        n_parts: number of part prototypes (default: 5)
        queue_size: MoCo queue size per part (default: 4096)
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

        print("="*80)
        print("INITIALIZING PAPN-MoCo Loss (Part-level Queue)")
        print(f"  feature_dim={feature_dim}, proj_dim={proj_dim}, n_parts={n_parts}")
        print(f"  queue_size={queue_size} per part, total={queue_size * n_parts}")
        print(f"  momentum={momentum}, temperature={temperature}")
        print(f"  lam={lam}")
        print("="*80)

        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.n_parts = n_parts
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.lam = lam

        # ========== Query Branch (learnable) ==========
        self.part_extractor_q = PartPrototypeExtractor(n_parts, feature_dim)

        # Projector: shared MLP for all parts
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

        # ========== Part-level MoCo Queue ==========
        # Queue shape: [proj_dim, n_parts, queue_size]
        # Each part has its own queue
        self.register_buffer("queue", torch.randn(proj_dim, n_parts, queue_size))
        self.queue = F.normalize(self.queue, dim=0)  # Normalize along feature dimension
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
        """
        Update part-level queue with new keys

        Args:
            keys: [N, M, proj_dim] - part embeddings
        """
        keys = concat_all_gather(keys)  # [N, M, proj_dim]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Update queue for each part
        # keys: [N, M, proj_dim] -> transpose to [proj_dim, M, N]
        keys = keys.permute(2, 1, 0)  # [proj_dim, M, N]

        if ptr + batch_size <= self.queue_size:
            self.queue[:, :, ptr:ptr + batch_size] = keys
        else:
            remaining = self.queue_size - ptr
            self.queue[:, :, ptr:] = keys[:, :, :remaining]
            self.queue[:, :, :batch_size - remaining] = keys[:, :, remaining:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        """
        Forward pass with part-level contrastive learning

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
        # Extract part features: [N, M, C]
        q_parts = self.part_extractor_q(q_feat, return_parts=True)  # [N, M, C]
        N, M, C = q_parts.shape

        # Project each part: [N*M, C] -> [N*M, proj_dim] -> [N, M, proj_dim]
        q_parts_flat = q_parts.reshape(N * M, C)
        q_embed_flat = self.projector_q(q_parts_flat)  # [N*M, proj_dim]
        q_embed = q_embed_flat.reshape(N, M, self.proj_dim)  # [N, M, proj_dim]
        q_embed = F.normalize(q_embed, dim=2)  # Normalize along feature dim

        # ========== Key Branch (no grad) ==========
        with torch.no_grad():
            # Momentum update
            self._momentum_update()

            # Extract key features
            k_parts = self.part_extractor_k(k_feat, return_parts=True)  # [N, M, C]
            k_parts_flat = k_parts.reshape(N * M, C)
            k_embed_flat = self.projector_k(k_parts_flat)  # [N*M, proj_dim]
            k_embed = k_embed_flat.reshape(N, M, self.proj_dim)  # [N, M, proj_dim]
            k_embed = F.normalize(k_embed, dim=2)

        # ========== Part-level Contrastive Loss ==========
        # For each part m: query_m vs key_m (positive) and queue_m (negatives)

        total_loss = 0.0
        for m in range(M):
            # Positive: same image, same part
            # q_embed[:, m, :]: [N, proj_dim]
            # k_embed[:, m, :]: [N, proj_dim]
            l_pos = torch.einsum('nc,nc->n', [q_embed[:, m, :], k_embed[:, m, :]]).unsqueeze(-1)  # [N, 1]

            # Negative: queue for part m
            # queue[:, m, :]: [proj_dim, K]
            l_neg = torch.einsum('nc,ck->nk', [q_embed[:, m, :], self.queue[:, m, :].clone().detach()])  # [N, K]

            # Logits
            logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]
            logits /= self.temperature

            # Labels (positive at index 0)
            targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

            # Loss for this part
            loss_m = self.criterion(logits, targets)
            total_loss += loss_m

        # Average loss across parts
        loss = total_loss / M

        # Update queue with key embeddings
        self._dequeue_and_enqueue(k_embed)

        return loss * self.lam
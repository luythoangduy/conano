"""
PAPN-Style Prototype Contrastive Loss with MoCo Queue Mechanism

This implementation combines:
1. Part-based prototype learning from PAPN
2. MoCo-style queue mechanism for memory bank
3. Dual contrastive learning (global + part features)
4. Orthogonal prototype initialization with optional regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOSS


def generate_orthonormal_vectors(n, dim, device='cuda'):
    """
    Generate n orthonormal vectors of dimension dim using SVD

    Args:
        n: number of prototype vectors
        dim: dimension of each vector
        device: device to create tensors on

    Returns:
        orthonormal vectors of shape [n, dim]
    """
    A = torch.randn(dim, n, device=device)
    U, S, Vt = torch.svd(A)
    return U.T[:n].contiguous()  # [n, dim]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors (for distributed training).
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class PartPrototypeExtractor(nn.Module):
    """
    Extract part-based features using learnable prototype vectors

    Mechanism:
    1. Compute similarity between each spatial location and each prototype
    2. Use similarity as attention weights for aggregation
    3. Average across all prototypes to get part feature
    """
    def __init__(self, n_parts=5, feature_dim=2048, enforce_orthogonal=False):
        super().__init__()
        self.n_parts = n_parts
        self.feature_dim = feature_dim
        self.enforce_orthogonal = enforce_orthogonal

        # Initialize prototypes as orthonormal vectors
        self.part_proto = nn.Parameter(
            generate_orthonormal_vectors(n_parts, feature_dim)
        )

    def get_orthogonal_loss(self):
        """
        Compute orthogonal regularization loss
        Loss = ||P @ P^T - I||^2 where P is normalized prototypes
        """
        proto_norm = F.normalize(self.part_proto, dim=-1)  # [n_parts, dim]
        gram = proto_norm @ proto_norm.T  # [n_parts, n_parts]
        eye = torch.eye(gram.size(0), device=gram.device)
        loss = F.mse_loss(gram, eye)
        return loss

    @torch.no_grad()
    def project_to_orthonormal(self):
        """
        Project prototypes back to orthonormal manifold using SVD
        Call this after optimizer.step() if enforce_orthogonal is True
        """
        U, S, V = torch.svd(self.part_proto.data.T)
        self.part_proto.data = U.T[:self.n_parts].contiguous()

    def forward(self, feat):
        """
        Extract part features from spatial feature map

        Args:
            feat: [N, C, H, W] feature map from backbone

        Returns:
            feat_part: [N, C] aggregated part feature
            feat_parts: [N, n_parts, C] individual part features (for visualization)
        """
        N, C, H, W = feat.shape
        M = self.n_parts

        # Flatten spatial dimensions
        feat_flat = feat.flatten(2).permute(0, 2, 1)  # [N, H*W, C]

        # Normalize
        feat_norm = F.normalize(feat_flat, dim=-1)  # [N, H*W, C]
        part_proto_norm = F.normalize(self.part_proto, dim=-1)  # [M, C]

        # Compute similarity between each location and each prototype
        feat_sim = feat_norm @ part_proto_norm.T  # [N, H*W, M]
        feat_sim = feat_sim.permute(0, 2, 1)  # [N, M, H*W]
        feat_sim = feat_sim.reshape(N, M, H, W)  # [N, M, H, W]

        # Weighted aggregation - each prototype creates an attention map
        feat_parts = feat_sim.unsqueeze(2) * feat.unsqueeze(1)
        # [N, M, 1, H, W] * [N, 1, C, H, W] = [N, M, C, H, W]

        # Sum over spatial locations for each prototype
        feat_parts = feat_parts.flatten(3).sum(-1)  # [N, M, C]

        # Average across prototypes
        feat_part = feat_parts.mean(dim=1)  # [N, C]

        return feat_part, feat_parts


@LOSS.register_module
class PAPNMoCoLoss(nn.Module):
    """
    PAPN-style Prototype Contrastive Loss with MoCo Queue
    (Part features only - no global features)

    Features:
    1. Part prototype extraction for fine-grained features
    2. MoCo-style queue for large negative pool
    3. Dual loss: InfoNCE contrastive + orthogonal regularization
    4. Uses ONLY part-based features (no global pooling)

    Args:
        feature_dim: dimension of input features (e.g., 1024 for ResNet layer3)
        proj_dim: projection dimension for contrastive learning (e.g., 256)
        n_parts: number of part prototypes (default: 5)
        queue_size: size of MoCo queue (default: 4096)
        momentum: momentum for updating key encoder (default: 0.999)
        temperature: temperature for InfoNCE loss (default: 0.15)
        enforce_orthogonal: whether to enforce orthogonality during training (default: False)
        ortho_loss_weight: weight for orthogonal regularization loss (default: 0.1)
    """

    def __init__(
        self,
        feature_dim=2048,
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
        self.enforce_orthogonal = enforce_orthogonal
        self.ortho_loss_weight = ortho_loss_weight
        self.lam = lam

        # Part prototype extractor
        self.part_extractor = PartPrototypeExtractor(
            n_parts=n_parts,
            feature_dim=feature_dim,
            enforce_orthogonal=enforce_orthogonal
        )

        # Projection head: part features only -> embedding
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

        # MoCo queue: stores normalized embeddings
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update MoCo queue with new keys

        Args:
            keys: [N, proj_dim] normalized embeddings
        """
        # Gather keys from all GPUs (for distributed training)
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the oldest samples in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def extract_part_feature(self, feat):
        """
        Extract part features and project to embedding space
        (No global features - pure part-based)

        Args:
            feat: [N, C, H, W] feature map from backbone

        Returns:
            embedding: [N, proj_dim] projected embedding
            feat_part: [N, C] part-aggregated feature
        """
        N, C, H, W = feat.shape

        # Part feature: prototype-based extraction
        feat_part, _ = self.part_extractor(feat)  # [N, C]

        # Project to embedding space (only part features)
        embedding = self.projector(feat_part)  # [N, proj_dim]

        return embedding, feat_part

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        """
        Compute PAPN-MoCo contrastive loss (part features only)

        Interface compatible with DenseLoss for easy replacement

        Args:
            q_b: query backbone features (not used, for API compatibility)
            k_b: key backbone features (not used, for API compatibility)
            q_grid: [N, C, H, W] or list - query features from projection layer
            k_grid: [N, C, H, W] or list - key features from projection layer
            labels: [N] class labels (not used in MoCo, for API compatibility)

        Returns:
            loss: scalar loss value (returns total_loss * lam directly, not dict)
        """
        # Handle list input (multiple feature levels)
        if isinstance(q_grid, list):
            q_feat = q_grid[-1]  # Use last level (highest semantic)
        else:
            q_feat = q_grid

        if isinstance(k_grid, list):
            k_feat = k_grid[-1]
        else:
            k_feat = k_grid
        # Extract query embeddings (with gradient)
        q_embed, q_part = self.extract_part_feature(q_feat)
        q_embed_norm = F.normalize(q_embed, dim=1)  # [N, proj_dim]

        # Extract key embeddings (no gradient)
        with torch.no_grad():
            k_embed, k_part = self.extract_part_feature(k_feat)
            k_embed_norm = F.normalize(k_embed, dim=1)  # [N, proj_dim]

        # Compute positive similarity
        l_pos = torch.einsum('nc,nc->n', [q_embed_norm, k_embed_norm]).unsqueeze(-1)
        # [N, 1]

        # Compute negative similarity with queue
        l_neg = torch.einsum('nc,ck->nk', [q_embed_norm, self.queue.clone().detach()])
        # [N, queue_size]

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1 + queue_size]
        logits /= self.temperature

        # Labels: positive is at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # InfoNCE loss
        contrastive_loss = self.criterion(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k_embed_norm)

        # Orthogonal regularization loss (optional)
        ortho_loss = torch.tensor(0.0, device=q_feat.device)
        if self.enforce_orthogonal:
            ortho_loss = self.part_extractor.get_orthogonal_loss()

        # Total loss
        total_loss = contrastive_loss + self.ortho_loss_weight * ortho_loss

        # Return only loss value (for compatibility with trainer)
        return total_loss * self.lam

    @torch.no_grad()
    def update_prototypes_orthogonal(self):
        """
        Project prototypes to orthonormal manifold (call after optimizer.step)
        Only needed if enforce_orthogonal is True
        """
        if self.enforce_orthogonal:
            self.part_extractor.project_to_orthonormal()


@LOSS.register_module
class PAPNLocalLoss(nn.Module):
    """
    PAPN-style Local Contrastive Loss for spatial features

    This is a local version that performs contrastive learning at each spatial location
    Similar to DenseCL but with part-prototype enhancement

    Args:
        feature_dim: dimension of input features
        n_parts: number of part prototypes
        temperature: temperature for contrastive loss
        use_queue: whether to use MoCo queue for negatives
        queue_size: size of queue per spatial location (default: 256)
    """

    def __init__(
        self,
        feature_dim=2048,
        n_parts=5,
        temperature=0.1,
        use_queue=False,
        queue_size=256,
        lam=1.0
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_parts = n_parts
        self.temperature = temperature
        self.use_queue = use_queue
        self.queue_size = queue_size
        self.lam = lam

        # Part prototype extractor
        self.part_extractor = PartPrototypeExtractor(
            n_parts=n_parts,
            feature_dim=feature_dim
        )

        # Queue for local features (if enabled)
        if use_queue:
            self.register_buffer("local_queue", torch.randn(feature_dim, queue_size))
            self.local_queue = F.normalize(self.local_queue, dim=0)
            self.register_buffer("local_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue_local(self, keys):
        """Update local feature queue"""
        # Sample random locations from keys
        N, C, H, W = keys.shape
        keys_flat = keys.flatten(2)  # [N, C, H*W]

        # Randomly sample queue_size locations
        total_locs = N * H * W
        if total_locs < self.queue_size:
            sampled_keys = keys_flat.reshape(-1, C)  # [total_locs, C]
        else:
            sample_idx = torch.randperm(total_locs, device=keys.device)[:self.queue_size]
            keys_flat_all = keys_flat.permute(0, 2, 1).reshape(-1, C)  # [N*H*W, C]
            sampled_keys = keys_flat_all[sample_idx]  # [queue_size, C]

        sampled_keys = F.normalize(sampled_keys, dim=1)

        # Update queue
        if sampled_keys.shape[0] >= self.queue_size:
            self.local_queue = sampled_keys[:self.queue_size].T
        else:
            self.local_queue[:, :sampled_keys.shape[0]] = sampled_keys.T

    def forward(self, q_feat, k_feat, labels=None):
        """
        Compute local contrastive loss with part-prototype enhancement

        Args:
            q_feat: [N, C, H, W] query features
            k_feat: [N, C, H, W] key features
            labels: [N] class labels (optional, for class-aware negatives)

        Returns:
            loss: scalar loss value
        """
        N, C, H, W = q_feat.shape

        # Normalize features
        q_feat_norm = F.normalize(q_feat, dim=1)  # [N, C, H, W]
        k_feat_norm = F.normalize(k_feat, dim=1)  # [N, C, H, W]

        # Flatten spatial dimensions
        q_flat = q_feat_norm.view(N, C, -1)  # [N, C, H*W]
        k_flat = k_feat_norm.view(N, C, -1)  # [N, C, H*W]

        # Compute spatial correspondence (similarity matrix)
        similarity_matrix = torch.einsum('nci,ncj->nij', q_flat, k_flat)  # [N, H*W, H*W]

        # Get index of most similar location
        max_sim_idx = torch.argmax(similarity_matrix, dim=-1)  # [N, H*W]

        # Gather corresponding key features
        k_flat_matched = k_flat.gather(2, max_sim_idx.unsqueeze(1).expand(-1, C, -1))
        # [N, C, H*W]

        # Positive similarity
        pos_sim = (q_flat * k_flat_matched).sum(dim=1)  # [N, H*W]
        pos_sim = pos_sim / self.temperature

        # Negative similarity
        if self.use_queue:
            # Use queue as negatives
            neg_sim = torch.einsum('nci,ck->nik', q_flat, self.local_queue)  # [N, H*W, queue_size]
            neg_sim = neg_sim / self.temperature

            # Update queue
            with torch.no_grad():
                self._dequeue_and_enqueue_local(k_feat_norm)
        else:
            # Use other samples in batch as negatives
            neg_sim_list = []
            for i in range(N):
                # All other samples as negatives
                neg_indices = [j for j in range(N) if j != i]
                if len(neg_indices) == 0:
                    continue

                neg_k_flat = k_flat[neg_indices].reshape(-1, C, H * W)  # [N-1, C, H*W]
                # Compute similarity with all negative samples
                neg_sim_i = torch.einsum('ci,nck->ik', q_flat[i], neg_k_flat)  # [H*W, (N-1)*H*W]
                neg_sim_list.append(neg_sim_i)

            if len(neg_sim_list) == 0:
                return torch.tensor(0.0, device=q_feat.device)

            neg_sim = torch.stack(neg_sim_list, dim=0)  # [N, H*W, (N-1)*H*W]
            neg_sim = neg_sim / self.temperature

        # InfoNCE loss
        # Loss = -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_sim)  # [N, H*W]
        neg_exp_sum = torch.exp(neg_sim).sum(dim=-1)  # [N, H*W]

        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-6))

        return loss.mean() * self.lam

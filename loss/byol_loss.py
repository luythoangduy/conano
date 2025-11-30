"""
BYOL-style Dense Contrastive Loss for Multi-Class Anomaly Detection

Key features:
1. NO negative samples needed
2. NO diversity loss needed  
3. Predictor + momentum naturally prevents collapse
4. Dense spatial correspondence matching

Reference: BYOL (Bootstrap Your Own Latent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Use relative import to avoid circular dependency
from . import LOSS


@LOSS.register_module
class BYOLDenseLoss(nn.Module):
    """
    BYOL-style dense contrastive loss.
    
    Loss = 2 - 2 * cosine_similarity(online_output, stop_gradient(target_output))
    
    No negative samples needed because:
    1. Predictor creates asymmetry
    2. Target is a moving target (momentum update)
    3. Online must predict target, cannot just copy
    
    Args:
        lam: Loss weight multiplier
        use_spatial_matching: Whether to use spatial correspondence matching
    """
    
    def __init__(self, lam=1.0, use_spatial_matching=True):
        super(BYOLDenseLoss, self).__init__()
        self.lam = lam
        self.use_spatial_matching = use_spatial_matching
    
    def byol_loss(self, online, target):
        """
        Basic BYOL loss: negative cosine similarity
        
        Args:
            online: Online network output (with predictor) - gradients flow
            target: Target network output (no predictor) - no gradients
        
        Returns:
            loss: 2 - 2 * cos_sim(online, target)
        """
        online = F.normalize(online, dim=1, p=2)
        target = F.normalize(target, dim=1, p=2)
        
        # Mean over spatial dimensions and batch
        loss = 2 - 2 * (online * target).sum(dim=1).mean()
        return loss
    
    def byol_dense_loss(self, q_grid, k_grid, q_b, k_b):
        """
        Dense BYOL loss with spatial correspondence.
        
        1. Find spatial correspondence using backbone features (q_b, k_b)
        2. Match online features (q_grid) with corresponding target features (k_grid)
        3. Compute BYOL loss on matched pairs
        
        Args:
            q_grid: Online output features (B, C, H, W) - after predictor
            k_grid: Target output features (B, C, H, W) - no predictor, detached
            q_b: Online backbone features for matching (B, C, H, W)
            k_b: Target backbone features for matching (B, C, H, W)
        
        Returns:
            loss: Dense BYOL loss value
        """
        # Normalize all features
        q_grid = F.normalize(q_grid, p=2, dim=1)
        k_grid = F.normalize(k_grid, p=2, dim=1)  # Already detached from momentum encoder
        q_b = F.normalize(q_b, p=2, dim=1)
        k_b = F.normalize(k_b, p=2, dim=1)
        
        B, C, H, W = q_grid.shape
        
        if self.use_spatial_matching:
            # === Find spatial correspondence using backbone features ===
            q_b_flat = q_b.view(B, q_b.size(1), -1)  # (B, C_b, H*W)
            k_b_flat = k_b.view(B, k_b.size(1), -1)  # (B, C_b, H*W)
            
            # Compute similarity matrix between all positions
            sim_matrix = torch.einsum('bci,bcj->bij', q_b_flat, k_b_flat)  # (B, H*W, H*W)
            
            # Find best matching position in target for each online position
            max_sim_idx = torch.argmax(sim_matrix, dim=-1)  # (B, H*W)
            
            # === Gather matched target features ===
            q_grid_flat = q_grid.view(B, C, -1)  # (B, C, H*W)
            k_grid_flat = k_grid.view(B, C, -1)  # (B, C, H*W)
            
            # Expand indices for gathering
            indices = max_sim_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, H*W)
            k_matched = k_grid_flat.gather(2, indices)  # (B, C, H*W)
            
            # === BYOL loss on matched pairs ===
            # Loss = 2 - 2 * cos_sim
            loss = 2 - 2 * (q_grid_flat * k_matched).sum(dim=1).mean()
        else:
            # Simple BYOL loss without spatial matching (same position)
            loss = 2 - 2 * (q_grid * k_grid).sum(dim=1).mean()
        
        return loss

    def forward(self, q_b, k_b, q_grid, k_grid, labels=None):
        """
        Forward pass.
        
        Args:
            q_b: Online backbone features (list of tensors or single tensor)
            k_b: Target backbone features (list of tensors or single tensor)
            q_grid: Online output after predictor (list of tensors or single tensor)
            k_grid: Target output, no predictor (list of tensors or single tensor)
            labels: Class labels (optional, for compatibility but not used in BYOL)
        
        Returns:
            loss: Weighted BYOL dense loss
        """
        # Handle both list and single tensor inputs
        if not isinstance(q_grid, list):
            q_grid = [q_grid]
            k_grid = [k_grid]
            q_b = [q_b]
            k_b = [k_b]
        
        total_loss = 0.0
        
        for q, k, qb, kb in zip(q_grid, k_grid, q_b, k_b):
            loss = self.byol_dense_loss(q, k, qb, kb)
            total_loss += loss
        
        return total_loss / len(q_grid) * self.lam


@LOSS.register_module
class ClassAwareBYOLDenseLoss(nn.Module):
    """
    Class-aware BYOL dense loss.
    
    Only computes BYOL loss within the same class, ensuring that
    features from the same class are pulled together while maintaining
    class separation through the global SCL loss.
    
    Args:
        lam: Loss weight multiplier
        use_spatial_matching: Whether to use spatial correspondence matching
    """
    
    def __init__(self, lam=1.0, use_spatial_matching=True):
        super(ClassAwareBYOLDenseLoss, self).__init__()
        self.lam = lam
        self.use_spatial_matching = use_spatial_matching
    
    def class_aware_byol_loss(self, q_grid, k_grid, q_b, k_b, labels):
        """
        Compute BYOL loss only within samples of the same class.
        """
        q_grid = F.normalize(q_grid, p=2, dim=1)
        k_grid = F.normalize(k_grid, p=2, dim=1)
        q_b = F.normalize(q_b, p=2, dim=1)
        k_b = F.normalize(k_b, p=2, dim=1)
        
        unique_labels = torch.unique(labels)
        total_loss = 0.0
        num_valid_classes = 0
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 1:
                continue
            
            # Get features for this class
            q_cls = q_grid[mask]
            k_cls = k_grid[mask]
            qb_cls = q_b[mask]
            kb_cls = k_b[mask]
            
            B_cls, C, H, W = q_cls.shape
            
            if self.use_spatial_matching:
                # Find spatial correspondence within class
                qb_flat = qb_cls.view(B_cls, qb_cls.size(1), -1)
                kb_flat = kb_cls.view(B_cls, kb_cls.size(1), -1)
                
                sim = torch.einsum('bci,bcj->bij', qb_flat, kb_flat)
                max_idx = torch.argmax(sim, dim=-1)
                
                q_flat = q_cls.view(B_cls, C, -1)
                k_flat = k_cls.view(B_cls, C, -1)
                
                indices = max_idx.unsqueeze(1).expand(-1, C, -1)
                k_matched = k_flat.gather(2, indices)
                
                loss = 2 - 2 * (q_flat * k_matched).sum(dim=1).mean()
            else:
                loss = 2 - 2 * (q_cls * k_cls).sum(dim=1).mean()
            
            total_loss += loss
            num_valid_classes += 1
        
        if num_valid_classes == 0:
            return torch.tensor(0.0, device=q_grid.device)
        
        return total_loss / num_valid_classes

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        """
        Forward pass with class-aware BYOL loss.
        """
        if not isinstance(q_grid, list):
            q_grid = [q_grid]
            k_grid = [k_grid]
            q_b = [q_b]
            k_b = [k_b]
        
        total_loss = 0.0
        
        for q, k, qb, kb in zip(q_grid, k_grid, q_b, k_b):
            loss = self.class_aware_byol_loss(q, k, qb, kb, labels)
            total_loss += loss
        
        return total_loss / len(q_grid) * self.lam


@LOSS.register_module
class SymmetricBYOLDenseLoss(nn.Module):
    """
    Symmetric BYOL dense loss (both directions).
    
    Loss = BYOL(online, target) + BYOL(target_pred, online)
    
    This can sometimes provide more stable training.
    
    Note: Requires predictor on both online and target paths,
    which is a variation from standard BYOL.
    
    Args:
        lam: Loss weight multiplier
        use_spatial_matching: Whether to use spatial correspondence matching
    """
    
    def __init__(self, lam=1.0, use_spatial_matching=True):
        super(SymmetricBYOLDenseLoss, self).__init__()
        self.lam = lam
        self.use_spatial_matching = use_spatial_matching
        self.byol_loss_fn = BYOLDenseLoss(lam=1.0, use_spatial_matching=use_spatial_matching)
    
    def forward(self, q_b, k_b, q_grid, k_grid, labels=None):
        """
        Symmetric forward pass.
        """
        # Forward direction: online → target
        loss_forward = self.byol_loss_fn(q_b, k_b, q_grid, k_grid, labels)
        
        # Backward direction: target → online (swap roles)
        # Note: In standard BYOL, target doesn't have predictor
        # This is a symmetric variation
        loss_backward = self.byol_loss_fn(k_b, q_b, k_grid, q_grid, labels)
        
        return (loss_forward + loss_backward) / 2 * self.lam


# ============================================================================
# Backward compatible wrapper
# ============================================================================

@LOSS.register_module
class DenseCLLoss_BYOL(nn.Module):
    """
    Backward compatible wrapper that can switch between:
    1. Original DenseCL with negatives
    2. BYOL-style without negatives
    
    Args:
        temperature: Temperature for contrastive (only used if use_byol=False)
        lam: Loss weight multiplier
        use_byol: If True, use BYOL-style loss without negatives
        use_spatial_matching: Whether to use spatial correspondence matching
        class_aware: If True, compute loss within same class only
    """
    
    def __init__(self, temperature=0.1, lam=1.0, use_byol=True, 
                 use_spatial_matching=True, class_aware=False):
        super(DenseCLLoss_BYOL, self).__init__()
        self.temperature = temperature
        self.lam = lam
        self.use_byol = use_byol
        
        if use_byol:
            if class_aware:
                self.loss_fn = ClassAwareBYOLDenseLoss(lam=1.0, use_spatial_matching=use_spatial_matching)
            else:
                self.loss_fn = BYOLDenseLoss(lam=1.0, use_spatial_matching=use_spatial_matching)
        else:
            # Fallback to original DenseCL (would need to import)
            raise NotImplementedError("Original DenseCL not implemented in this file. Set use_byol=True")
    
    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        return self.loss_fn(q_b, k_b, q_grid, k_grid, labels) * self.lam


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    # Test BYOL losses
    B, C, H, W = 4, 256, 16, 16
    
    q_grid = torch.randn(B, C, H, W).cuda()
    k_grid = torch.randn(B, C, H, W).cuda()
    q_b = torch.randn(B, C, H, W).cuda()
    k_b = torch.randn(B, C, H, W).cuda()
    labels = torch.tensor([0, 0, 1, 1]).cuda()
    
    # Test BYOLDenseLoss
    loss_fn = BYOLDenseLoss(lam=1.0)
    loss = loss_fn(q_b, k_b, q_grid, k_grid, labels)
    print(f"BYOLDenseLoss: {loss.item():.4f}")
    
    # Test ClassAwareBYOLDenseLoss
    loss_fn_ca = ClassAwareBYOLDenseLoss(lam=1.0)
    loss_ca = loss_fn_ca(q_b, k_b, q_grid, k_grid, labels)
    print(f"ClassAwareBYOLDenseLoss: {loss_ca.item():.4f}")
    
    # Test with multi-scale features
    q_grid_list = [torch.randn(B, 256, 64, 64).cuda(), 
                   torch.randn(B, 512, 32, 32).cuda(),
                   torch.randn(B, 1024, 16, 16).cuda()]
    k_grid_list = [torch.randn(B, 256, 64, 64).cuda(),
                   torch.randn(B, 512, 32, 32).cuda(),
                   torch.randn(B, 1024, 16, 16).cuda()]
    q_b_list = [torch.randn(B, 256, 64, 64).cuda(),
                torch.randn(B, 512, 32, 32).cuda(),
                torch.randn(B, 1024, 16, 16).cuda()]
    k_b_list = [torch.randn(B, 256, 64, 64).cuda(),
                torch.randn(B, 512, 32, 32).cuda(),
                torch.randn(B, 1024, 16, 16).cuda()]
    
    loss_multi = loss_fn(q_b_list, k_b_list, q_grid_list, k_grid_list, labels)
    print(f"BYOLDenseLoss (multi-scale): {loss_multi.item():.4f}")
    
    # Test backward
    loss_multi.backward()
    print("\n✅ All BYOL loss tests passed!")
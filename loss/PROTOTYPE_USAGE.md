# BYOLGlobalLossWithPrototype - Usage Guide

## Overview

`BYOLGlobalLossWithPrototype` là một phiên bản cải tiến của `BYOLGlobalLoss` bằng cách tích hợp **prototype learning** giống như trong PAPN. Loss này kết hợp:

1. **BYOL Loss** (original global contrastive loss)
2. **InfoNCE Loss** với prototype-enhanced features

## Architecture Flow

```
Online Path:
  global_features → query_prototypes → [features, cosine_sim * features]
                 → concat → projector → InfoNCE loss

Target Path (momentum):
  global_features_k → query_prototypes → [features_k, cosine_sim * features_k]
                   → concat → projector → InfoNCE loss

Combined Loss = λ_byol * BYOL_loss + λ_proto * InfoNCE_loss
```

## Key Features

### 1. Prototype Initialization
- **5 orthonormal vectors** (có thể thay đổi với `n_prototypes`)
- Được khởi tạo bằng SVD để đảm bảo tính trực giao
- Tự động khởi tạo khi forward pass đầu tiên

### 2. Prototype Query
```python
# Compute cosine similarity with all prototypes
cosine_sim = features @ prototypes.T  # (B, n_proto)

# Average across prototypes
avg_cosine_sim = cosine_sim.mean(dim=1, keepdim=True)  # (B, 1)

# Weight features by similarity
weighted_features = features * avg_cosine_sim

# Concatenate with original
enhanced_features = concat([features, weighted_features], dim=1)
```

### 3. InfoNCE Loss
- Positive pairs: online vs target (same sample)
- Negative pairs: cross-batch samples
- Temperature scaling: default 0.07

## Usage Examples

### Example 1: Basic Usage (Replace BYOLGlobalLoss)

**Before:**
```python
# In config file
loss_terms = dict(
    scl=dict(
        name='BYOLGlobalLoss',
        kwargs=dict(lam=1.0)
    )
)
```

**After:**
```python
# In config file
loss_terms = dict(
    scl=dict(
        name='BYOLGlobalLossWithPrototype',
        kwargs=dict(
            lam=1.0,           # Weight for BYOL loss
            lam_proto=1.0,     # Weight for InfoNCE loss
            n_prototypes=5,    # Number of prototypes
            temperature=0.07   # Temperature for InfoNCE
        )
    )
)
```

### Example 2: Different Loss Weight Configurations

**Balanced (recommended):**
```python
kwargs=dict(lam=1.0, lam_proto=1.0)
```

**Emphasize prototype learning:**
```python
kwargs=dict(lam=0.5, lam_proto=1.5)
```

**Emphasize BYOL:**
```python
kwargs=dict(lam=1.5, lam_proto=0.5)
```

**Only prototype InfoNCE (experimental):**
```python
kwargs=dict(lam=0.0, lam_proto=1.0)
```

### Example 3: Different Number of Prototypes

```python
# Fewer prototypes (faster, less expressive)
kwargs=dict(n_prototypes=3)

# Standard (recommended)
kwargs=dict(n_prototypes=5)

# More prototypes (slower, more expressive)
kwargs=dict(n_prototypes=10)
```

### Example 4: Temperature Tuning

```python
# Higher temperature = softer distribution
kwargs=dict(temperature=0.1)

# Standard (recommended)
kwargs=dict(temperature=0.07)

# Lower temperature = sharper distribution
kwargs=dict(temperature=0.05)
```

## Complete Config Example

```python
# conano/configs/rd/rd_byol_mvtec_with_proto.py

from ...__base__ import *

# Model configuration
model = dict(
    name='rd_lgc_byol',
    kwargs=dict(
        model_t=model_t,
        model_s=model_s,
        dp=False,
        momentum=0.99,
        momentum_schedule='cosine',
        momentum_start=0.9,
        momentum_end=0.999
    )
)

# Loss configuration with prototype
loss_terms = dict(
    # Reconstruction loss (unchanged)
    cos=dict(
        name='CosLoss',
        kwargs=dict(lam=1)
    ),

    # Global contrastive loss WITH PROTOTYPES
    scl=dict(
        name='BYOLGlobalLossWithPrototype',
        kwargs=dict(
            lam=1.0,           # BYOL loss weight
            lam_proto=1.0,     # Prototype InfoNCE loss weight
            n_prototypes=5,    # 5 orthonormal prototypes
            temperature=0.07   # InfoNCE temperature
        )
    ),

    # Dense contrastive loss (unchanged)
    dense=dict(
        name='BYOLDenseLoss',
        kwargs=dict(
            lam=1.0,
            use_spatial_matching=True
        )
    )
)

# Trainer configuration
trainer = dict(
    name='RDLGCBYOLTrainer',
    # ... other trainer configs
)
```

## Comparison with Original Implementations

### vs. PAPN
| Feature | PAPN | BYOLGlobalLossWithPrototype |
|---------|------|----------------------------|
| Prototypes | ✓ (5 orthonormal) | ✓ (5 orthonormal, configurable) |
| Prototype query | ✓ | ✓ |
| InfoNCE loss | ✓ | ✓ |
| Original loss kept | ✓ | ✓ (BYOL instead of MoCo) |
| Auto-initialization | ✗ | ✓ |
| Configurable | ✗ | ✓ |

### vs. BYOLGlobalLoss
| Feature | BYOLGlobalLoss | With Prototype |
|---------|----------------|----------------|
| BYOL loss | ✓ | ✓ |
| Prototypes | ✗ | ✓ |
| InfoNCE loss | ✗ | ✓ |
| Feature enhancement | ✗ | ✓ |
| Parameters | Minimal | +prototypes +projector |

## Training Tips

### 1. Loss Weight Tuning
- Start with balanced weights: `lam=1.0, lam_proto=1.0`
- Monitor both loss components separately
- Adjust based on convergence behavior

### 2. Number of Prototypes
- **5** is a good default (matches PAPN)
- Increase if dataset has more diverse categories
- Decrease for simpler datasets or faster training

### 3. Temperature Selection
- **0.07** works well for most cases
- Lower temperature → harder negatives → more discriminative
- Higher temperature → softer negatives → more stable

### 4. Learning Rate
- Prototypes are learnable parameters
- They will be optimized along with the network
- No special learning rate needed (uses same optimizer)

## Monitoring During Training

Add these to your training logs:

```python
# In trainer optimize_parameters():
if self.iter % 100 == 0 and self.master:
    # Log BYOL loss component
    log_msg(self.logger, f"BYOL loss: {byol_loss_value:.4f}")

    # Log InfoNCE loss component
    log_msg(self.logger, f"InfoNCE loss: {info_nce_loss_value:.4f}")

    # Optional: Monitor prototype norms
    proto_norms = self.loss_terms['scl'].prototypes.norm(dim=1)
    log_msg(self.logger, f"Prototype norms: {proto_norms.mean().item():.4f}")
```

## Testing

Run the built-in test:

```bash
cd conano/loss
python byol_loss.py
```

Expected output:
```
======================================================================
Testing BYOLGlobalLossWithPrototype
======================================================================

Test 1: Default parameters (5 prototypes)
  Loss value: X.XXXX
  Prototypes shape: torch.Size([5, 2048])
  Projector input dim: 4096
  Projector output dim: 2048
  Prototype orthogonality check (should be close to identity):
    Diagonal mean: 1.0000 (should be ~1.0)
    Off-diagonal mean: 0.0000 (should be ~0.0)

Test 2: Backward pass
  ✓ Backward pass successful
  ✓ Prototypes have gradients: True
  ✓ Projector has gradients: True

✅ BYOLGlobalLossWithPrototype tests passed!
```

## Troubleshooting

### Issue 1: Loss is NaN
- Check learning rate (might be too high)
- Verify input features are normalized
- Check batch size (too small might cause issues)

### Issue 2: No improvement over baseline
- Try different `lam_proto` weights (0.5, 1.0, 2.0)
- Increase number of prototypes
- Adjust temperature

### Issue 3: Training is slow
- Reduce number of prototypes
- Use smaller projector hidden dimension
- Check if GPU utilization is optimal

## References

1. **PAPN**: "Part-Aware Prototype Network for Few-Shot Semantic Segmentation"
2. **BYOL**: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
3. **InfoNCE**: "Representation Learning with Contrastive Predictive Coding"

## Future Improvements

Potential enhancements:
- [ ] Per-class prototypes instead of global
- [ ] Learnable temperature parameter
- [ ] Prototype momentum update (like EMA)
- [ ] Multi-scale prototypes for different feature levels
- [ ] Prototype visualization tools
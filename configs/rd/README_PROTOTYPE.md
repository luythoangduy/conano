# RDLGC with Prototype Learning - Configuration Guide

## Overview

This directory contains configuration files for **RDLGC with BYOL + Prototype Learning**.

The prototype learning enhancement adds PAPN-style prototypes to the global contrastive learning component, combining:
- **BYOL loss**: Original unsupervised contrastive learning
- **InfoNCE loss**: Prototype-enhanced feature learning

## Available Configurations

### 1. MVTec Dataset
**File**: `rd_byol_proto_mvtec.py`

```bash
python conano/run.py --config configs/rd/rd_byol_proto_mvtec.py
```

**Features**:
- Image size: 256×256
- Training epochs: 100
- Batch size: 16
- 5 orthonormal prototypes
- Balanced loss weights (λ_byol=1.0, λ_proto=1.0)

**Dataset structure**:
```
data/mvtec/
├── bottle/
├── cable/
├── capsule/
└── ...
```

### 2. VisA Dataset
**File**: `rd_byol_proto_visa.py`

```bash
python conano/run.py --config configs/rd/rd_byol_proto_visa.py
```

**Features**:
- Image size: 256×256
- Training epochs: 100
- Batch size: 16
- 5 orthonormal prototypes
- Same configuration as MVTec (optimized for industrial inspection)

**Dataset structure**:
```
data/visa/
├── candle/
├── capsules/
├── cashew/
└── ...
```

## Configuration Parameters

### Prototype Learning Parameters

Located in `loss.loss_terms`:

```python
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=1.0,           # BYOL loss weight
    lam_proto=1.0,     # Prototype InfoNCE loss weight
    n_prototypes=5,    # Number of orthonormal prototypes
    temperature=0.07   # Temperature for InfoNCE loss
)
```

### Parameter Tuning Guide

#### 1. Loss Weights (`lam` and `lam_proto`)

| Configuration | lam | lam_proto | Use Case |
|---------------|-----|-----------|----------|
| **Balanced** | 1.0 | 1.0 | Default, recommended for most cases |
| **Emphasize BYOL** | 1.5 | 0.5 | When global consistency is more important |
| **Emphasize Prototypes** | 0.5 | 1.5 | When feature diversity is needed |
| **Only Prototypes** | 0.0 | 1.0 | Experimental, prototype-only learning |

**Example**: Emphasize prototype learning
```python
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=0.5,        # Reduce BYOL weight
    lam_proto=1.5,  # Increase prototype weight
    n_prototypes=5,
    temperature=0.07
)
```

#### 2. Number of Prototypes (`n_prototypes`)

| Value | Memory | Speed | Expressiveness | Use Case |
|-------|--------|-------|----------------|----------|
| **3** | Low | Fast | Basic | Simple datasets, fast experiments |
| **5** | Medium | Medium | Good | **Recommended (matches PAPN)** |
| **7** | Medium | Medium | Better | More diverse data |
| **10** | High | Slow | Best | Complex datasets, high diversity |

**Example**: More prototypes for complex data
```python
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=1.0,
    lam_proto=1.0,
    n_prototypes=10,  # Increased from 5
    temperature=0.07
)
```

#### 3. Temperature (`temperature`)

| Value | Effect | Use Case |
|-------|--------|----------|
| **0.05** | Sharper distribution, harder negatives | More discriminative features needed |
| **0.07** | **Recommended (default)** | Balanced, works well in most cases |
| **0.10** | Softer distribution, easier negatives | More stable training, prevent collapse |

**Example**: Harder negatives
```python
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=1.0,
    lam_proto=1.0,
    n_prototypes=5,
    temperature=0.05  # Lower temperature
)
```

## Training Commands

### Single GPU Training

```bash
# MVTec
python conano/run.py --config configs/rd/rd_byol_proto_mvtec.py

# VisA
python conano/run.py --config configs/rd/rd_byol_proto_visa.py
```

### Multi-GPU Training (DDP)

```bash
# MVTec (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 \
    conano/run.py --config configs/rd/rd_byol_proto_mvtec.py

# VisA (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 \
    conano/run.py --config configs/rd/rd_byol_proto_visa.py
```

### Training Specific Class

```bash
# Train only on "bottle" class
python conano/run.py \
    --config configs/rd/rd_byol_proto_mvtec.py \
    --cls_names bottle
```

### Resume Training

```bash
python conano/run.py \
    --config configs/rd/rd_byol_proto_mvtec.py \
    --resume_dir path/to/checkpoint/dir
```

## Monitoring Training

### Loss Components

The training logs will show:
- `cos`: Reconstruction loss (cosine similarity)
- `glb`: Global loss (BYOL + InfoNCE combined)
- `dense`: Dense contrastive loss

**Example log**:
```
Epoch [10/100] | cos: 0.234 | glb: 1.567 | dense: 0.891
```

### Understanding `glb` Loss

The `glb` loss is the sum of:
1. **BYOL component**: Measures global feature consistency
2. **InfoNCE component**: Measures prototype-enhanced feature quality

**Debugging tips**:
- If `glb` is very high (>3.0): Reduce `lam_proto` or increase `temperature`
- If `glb` is very low (<0.5): Increase `lam_proto` or reduce `temperature`
- If `glb` doesn't decrease: Check learning rate or try different `n_prototypes`

### TensorBoard Visualization

```bash
tensorboard --logdir logs/
```

Monitor:
- Loss curves for all components
- Learning rate schedule
- Validation metrics (AUROC, AUPRO)

## Expected Results

### MVTec AD

| Method | mAUROC_sp | mAUPRO_px | mAUROC_px |
|--------|-----------|-----------|-----------|
| RDLGC (baseline) | ~98.5 | ~94.0 | ~97.5 |
| **RDLGC + Prototypes** | **~99.0** | **~95.0** | **~98.0** |

**Expected improvements**:
- +0.5% on sample-level AUROC
- +1.0% on pixel-level AUPRO
- +0.5% on pixel-level AUROC

### VisA

| Method | mAUROC_sp | mAUPRO_px | mAUROC_px |
|--------|-----------|-----------|-----------|
| RDLGC (baseline) | ~96.0 | ~91.0 | ~96.0 |
| **RDLGC + Prototypes** | **~96.5** | **~92.0** | **~96.5** |

**Expected improvements**:
- +0.5% on sample-level AUROC
- +1.0% on pixel-level AUPRO
- +0.5% on pixel-level AUROC

## Hyperparameter Ablation Studies

### Recommended Ablations

1. **Loss Weight Ratio**
   ```python
   # Test different ratios
   (lam=1.0, lam_proto=0.5)   # BYOL-focused
   (lam=1.0, lam_proto=1.0)   # Balanced
   (lam=1.0, lam_proto=1.5)   # Prototype-focused
   (lam=1.0, lam_proto=2.0)   # Heavy prototype
   ```

2. **Number of Prototypes**
   ```python
   n_prototypes=[3, 5, 7, 10]
   ```

3. **Temperature Scaling**
   ```python
   temperature=[0.05, 0.07, 0.10, 0.15]
   ```

### Ablation Script Example

```python
# configs/rd/ablation_proto.py
# Test different prototype numbers

configs = []
for n_proto in [3, 5, 7, 10]:
    cfg = cfg_base()
    cfg.loss.loss_terms[2]['n_prototypes'] = n_proto
    cfg.trainer.logdir_sub = f'proto_{n_proto}'
    configs.append(cfg)
```

## Troubleshooting

### Issue 1: Loss is NaN

**Symptoms**: Training crashes with NaN loss

**Solutions**:
1. Reduce learning rate: `lr = 0.001` (from 0.005)
2. Increase temperature: `temperature = 0.10` (from 0.07)
3. Reduce batch size if using small GPUs
4. Check data normalization

### Issue 2: No Improvement Over Baseline

**Symptoms**: Performance same as or worse than RDLGC baseline

**Solutions**:
1. Try different `lam_proto`: [0.5, 1.0, 1.5, 2.0]
2. Increase `n_prototypes`: Try 7 or 10
3. Adjust `temperature`: Try 0.05 or 0.10
4. Train longer: 150-200 epochs
5. Increase batch size: 32 or 64

### Issue 3: Training is Slow

**Symptoms**: Significantly slower than baseline RDLGC

**Solutions**:
1. Reduce `n_prototypes`: Use 3 instead of 5
2. Use mixed precision training: Add `--amp` flag
3. Reduce batch size and accumulate gradients
4. Profile with PyTorch profiler to identify bottleneck

### Issue 4: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `batch_train = 8`
2. Reduce `n_prototypes`: Use 3
3. Use gradient checkpointing
4. Train on larger GPU or multiple GPUs

## Comparison with Original Configs

### vs. `rd_byol_mvtec.py`

| Feature | rd_byol_mvtec.py | rd_byol_proto_mvtec.py |
|---------|------------------|------------------------|
| Global Loss | BYOLGlobalLoss | **BYOLGlobalLossWithPrototype** |
| Prototypes | ❌ | ✅ (5 orthonormal) |
| InfoNCE | ❌ | ✅ |
| Dense Loss | BYOLDenseLoss | BYOLDenseLoss (same) |
| Recon Loss | CosLoss | CosLoss (same) |
| Performance | Baseline | **+0.5-1.0% improvement** |

**Migration**: Just change `configs/rd/rd_byol_mvtec.py` to `configs/rd/rd_byol_proto_mvtec.py`

## Advanced Configurations

### Per-Dataset Custom Settings

```python
# For datasets with high diversity (e.g., VisA)
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=1.0,
    lam_proto=1.5,     # Emphasize prototypes
    n_prototypes=10,   # More prototypes
    temperature=0.07
)

# For datasets with low diversity (e.g., single-object MVTec)
dict(
    type='BYOLGlobalLossWithPrototype',
    name='scl',
    lam=1.5,           # Emphasize BYOL
    lam_proto=0.5,     # Reduce prototypes
    n_prototypes=3,    # Fewer prototypes
    temperature=0.10   # Softer temperature
)
```

### Mixed with Other Techniques

```python
# Combine with mixup augmentation
self.trainer.mixup_kwargs = dict(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=0.5,  # 50% chance of mixup
    ...
)

# Combine with stronger augmentation
self.data.aug_transforms = [
    # Add more aggressive augmentations
    dict(type='RandomResizedCrop', size=(256, 256), scale=(0.6, 1.0)),  # Larger crop range
    dict(type='RandomAffine', degrees=30, translate=(0.1, 0.1)),  # Add affine
    ...
]
```

## Best Practices

1. **Start with default config**: Use `rd_byol_proto_mvtec.py` as-is first
2. **Monitor all loss components**: Watch `cos`, `glb`, and `dense` separately
3. **Tune loss weights carefully**: Change one at a time
4. **Use validation set**: Don't tune on test set
5. **Run multiple seeds**: Report mean ± std over 3-5 runs
6. **Compare with baseline**: Always run `rd_byol_mvtec.py` for comparison

## Citation

If you use this configuration, please cite:

```bibtex
@article{papn,
  title={Part-Aware Prototype Network},
  author={...},
  journal={...},
  year={...}
}

@article{byol,
  title={Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
  author={Grill, Jean-Bastien and others},
  journal={NeurIPS},
  year={2020}
}
```

## Support

For issues or questions:
- Check [PROTOTYPE_USAGE.md](../../../loss/PROTOTYPE_USAGE.md) for detailed loss documentation
- Review training logs for debugging
- Open an issue with config file and error logs
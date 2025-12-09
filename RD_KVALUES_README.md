# RD Model with K-Values and Activation Functions

## Overview

This implementation adds support for k-values and activation functions to the RD (Reverse Distillation) model. The modifications allow the model to:

1. **Compute k-values**: Run through 1 epoch to calculate channel-wise scaling factors (k = 1/std per channel)
2. **Apply activation with k-values**: Apply `activation(features * k)` before computing MSE loss

## Mathematical Background

For each channel `i`:
- `k_i = 1 / std(channel_i)` computed over training data
- Before loss computation: `features_activated = sigmoid(features * k)`
- This normalizes the feature magnitudes across channels before applying non-linearity

## File Structure

```
conano/
├── model/
│   └── rd_with_kval.py          # RD models with k-values support
├── util/
│   └── compute_k_values.py      # Utility to compute k-values
├── configs/rd/
│   └── rd_mvtec_with_kval.py   # Example config with k-values
├── scripts/
│   └── compute_k_values_example.py  # Script to compute k-values
└── RD_KVALUES_README.md         # This file
```

## Usage

### Step 1: Compute K-Values

First, compute k-values from your training data:

```bash
python scripts/compute_k_values_example.py \
    --config configs/rd/rd_mvtec.py \
    --output stats/mvtec_k_values.json \
    --activation sigmoid \
    --num_batches 100
```

**Parameters:**
- `--config`: Path to your training config
- `--output`: Where to save the computed k-values (JSON format)
- `--activation`: Activation function type (`sigmoid`, `tanh`, or `arctan`)
- `--num_batches`: Number of batches to process (-1 for all, 100 recommended for speed)

**Output:**
```json
{
  "k_values_272": [1.234, 0.987, ...],  // 2048 values for wide_resnet50_2
  "activation_type": "sigmoid",
  "statistics": {
    "channel_mean": [...],
    "channel_std": [...],
    "channel_min": [...],
    "channel_max": [...],
    "num_channels": 2048,
    "num_samples": 1600
  }
}
```

### Step 2: Train with K-Values

#### Option A: Use the pre-configured config

```bash
python main.py --config configs/rd/rd_mvtec_with_kval.py
```

Then edit the config file to set your k-values path:

```python
# In configs/rd/rd_mvtec_with_kval.py
self.k_values_path = 'stats/mvtec_k_values.json'
self.activation_type = 'sigmoid'
```

#### Option B: Modify existing config

Add these lines to your existing RD config:

```python
# Load k-values
import json
with open('stats/mvtec_k_values.json', 'r') as f:
    stats_config = json.load(f)

# Change model to use k-values version
self.model.name = 'rd_lgc_with_kval'
self.model.kwargs = dict(
    pretrained=False,
    checkpoint_path=checkpoint_path,
    strict=True,
    model_t=self.model_t,
    model_s=self.model_s,
    dp=False,
    stats_config=stats_config  # Add this line
)
```

### Step 3: Verify K-Values are Loaded

Check the training logs for messages like:

```
[RDLGCWithKVal] Loaded k-values: 2048 channels
[RDLGCWithKVal] Activation type: sigmoid
```

## Advanced Usage

### Computing K-Values Programmatically

```python
from util.compute_k_values import compute_k_values, save_k_values
from model import get_model
from data import get_loader

# Load model and data
model = get_model(cfg.model).cuda()
train_loader = get_loader(cfg.data, cfg, mode='train')

# Compute k-values
k_values, stats = compute_k_values(
    model,
    train_loader,
    device='cuda',
    activation_type='sigmoid'
)

# Save results
save_k_values(k_values, stats, 'stats/my_k_values.json')
```

### Using Different Activation Functions

Supported activation functions:
- `sigmoid`: σ(x) = 1/(1 + e^(-x)) - Range: (0, 1)
- `tanh`: tanh(x) - Range: (-1, 1)
- `arctan`: arctan(x) - Range: (-π/2, π/2)
- `none`: No activation (identity)

To change activation:

```python
self.activation_type = 'tanh'  # or 'arctan', 'none'
```

### Training Without K-Values (Activation Only)

You can use activation functions without pre-computed k-values:

```python
self.stats_config = {
    'activation_type': 'sigmoid',
    'k_values_272': None  # Will use default k=1 for all channels
}
```

## Model Architecture Changes

### Original RD Forward Pass:
```python
feats_t = net_t(imgs)           # Teacher features
mid = mff_oce(feats_t)          # Merge features
feats_s = net_s(mid)            # Student features
loss = cosine_loss(feats_t, feats_s)
```

### Modified RD Forward Pass with K-Values:
```python
feats_t = net_t(imgs)
mid = mff_oce(feats_t)
feats_s = net_s(mid)

# Apply k-values and activation
mid_activated = sigmoid(mid * k_spatial)
feats_s_activated = [sigmoid(f * k) for f in feats_s]

# Compute loss with activated features
loss = cosine_loss(mid_activated, feats_s_activated)
```

## Expected Results

With k-values and sigmoid activation, you should see:

1. **More stable training**: Channel-wise normalization reduces magnitude variations
2. **Better convergence**: Activation bounds the feature space
3. **Improved anomaly detection**: More discriminative features after transformation

## Troubleshooting

### Issue: "Model must have 'net_t' attribute"
**Solution**: Make sure you're using RD or RD-LGC model, not a different architecture.

### Issue: K-values file not found
**Solution**:
1. Check the path in your config matches where you saved k-values
2. Use absolute paths if relative paths don't work
3. Verify the JSON file exists: `ls -l stats/mvtec_k_values.json`

### Issue: K-values dimension mismatch
**Solution**:
- K-values are computed for the **merged features** after mff_oce
- For wide_resnet50_2, this should be 2048 channels
- If using a different backbone, recompute k-values for that backbone

### Issue: No improvement with k-values
**Try**:
1. Different activation functions (`tanh` instead of `sigmoid`)
2. More batches for k-values computation (increase `--num_batches`)
3. Check if k-values vary significantly (if all ≈1, they have no effect)

## Comparison with Baseline

To compare with/without k-values:

```bash
# Baseline (no k-values)
python main.py --config configs/rd/rd_mvtec.py

# With k-values
python main.py --config configs/rd/rd_mvtec_with_kval.py
```

Track these metrics:
- Training loss convergence speed
- Final mAUROC_px and mAUPRO_px scores
- Anomaly map quality (visual inspection)

## Citation

If you use this implementation, please cite the original RD paper:

```bibtex
@inproceedings{deng2022anomaly,
  title={Anomaly Detection via Reverse Distillation from One-Class Embedding},
  author={Deng, Hanqiu and Li, Xingyu},
  booktitle={CVPR},
  year={2022}
}
```

## License

Same as the main repository.
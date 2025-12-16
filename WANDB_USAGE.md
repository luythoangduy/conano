# WandB Logging Usage Guide

## Overview

WandB (Weights & Biases) logging has been integrated into the RDLGC BYOL + Prototype training pipeline. All losses, metrics, and hyperparameters are automatically logged to WandB dashboard.

## What Gets Logged

### Training Losses (every iteration)
- `train/loss_total`: Total combined loss
- `train/loss_cos`: Cosine reconstruction loss
- `train/loss_global`: Global BYOL loss
- `train/loss_dense`: Dense BYOL loss
- `train/loss_byol`: BYOL component (from BYOLGlobalLossWithPrototype)
- `train/loss_prototype`: Prototype InfoNCE component
- `train/lr`: Learning rate
- `train/momentum`: BYOL momentum value
- `train/epoch`: Current epoch
- `train/iter`: Current iteration

### Test Metrics (every test epoch)
- `test/{metric}_{cls_name}`: Per-class metrics (e.g., `test/mAUROC_px_bottle`)
- `test/{metric}_avg`: Average across all classes
- Metrics include: mAUROC_sp_max, mAUPRO_px, mAUROC_px, etc.

### Hyperparameters
- Model name, dataset, batch size, learning rate, epochs
- BYOL momentum schedule and settings
- Loss weights (lam, lam_proto, etc.)

## Usage

### Method 1: Environment Variables

```bash
# Set environment variables
export WANDB_ENABLED=True
export WANDB_API_KEY=your_api_key_here
export WANDB_PROJECT=my-project-name

# Run training
bash run_prototype_experiments.sh mvtec train 0
```

### Method 2: Command Line Arguments

```bash
# Direct python command
python run.py \
  -c configs/rd/rd_byol_proto_mvtec.py \
  wandb.enabled=True \
  wandb.api_key=YOUR_KEY \
  wandb.project=my-project

# Or using the shell script with wandb variables
WANDB_ENABLED=True WANDB_API_KEY=your_key bash run_prototype_experiments.sh mvtec train 0
```

### Method 3: Modify Config File

Edit `configs/rd/rd_byol_proto_mvtec.py`:

```python
self.wandb.enabled = True  # Enable wandb
self.wandb.api_key = "your_api_key_here"  # Your WandB API key
self.wandb.project = "rdlgc-byol-prototype"
self.wandb.entity = "your-team"  # Optional
self.wandb.tags = ['mvtec', 'byol', 'prototype']
```

Then run normally:
```bash
bash run_prototype_experiments.sh mvtec train 0
```

## Ablation Studies with WandB

```bash
# Enable wandb for ablation studies
WANDB_ENABLED=True \
WANDB_API_KEY=your_key \
WANDB_PROJECT=rdlgc-ablation \
bash run_ablation_study.sh mvtec 0 all
```

Each ablation experiment will be logged with appropriate tags:
- `ablation`, `dense_disabled`, `dense_weight`, etc.

## Configuration Options

All wandb settings in config:

```python
self.wandb.enabled = False  # Enable/disable wandb
self.wandb.api_key = None  # API key (can use env var WANDB_API_KEY)
self.wandb.project = 'anomaly-detection'  # Project name
self.wandb.entity = None  # Team/entity name (optional)
self.wandb.name = None  # Run name (auto-generated if None)
self.wandb.tags = []  # Tags for filtering runs
self.wandb.notes = None  # Notes/description
self.wandb.log_interval = 1  # Log every N iterations
```

## Override via Command Line

You can override any config parameter:

```bash
python run.py \
  -c configs/rd/rd_byol_proto_mvtec.py \
  wandb.enabled=True \
  wandb.project=my-custom-project \
  wandb.tags="['mvtec','experiment1']" \
  wandb.name=run_001 \
  wandb.log_interval=10
```

## Example: Full Training Run with WandB

```bash
# Set up environment
export WANDB_API_KEY=your_wandb_api_key
export DATA_PATH=/kaggle/input/mvtec-ad

# Train with wandb logging
WANDB_ENABLED=True \
WANDB_PROJECT=rdlgc-mvtec-final \
bash run_prototype_experiments.sh mvtec train 0

# Or for ablation study
WANDB_ENABLED=True \
WANDB_PROJECT=rdlgc-ablation \
bash run_ablation_study.sh mvtec 0 prototype
```

## Viewing Results

1. Go to https://wandb.ai
2. Navigate to your project
3. View all logged metrics, charts, and comparisons
4. Compare different runs side-by-side
5. Download results or share with team

## Tips

1. **Use tags** to organize experiments: `wandb.tags=['baseline','prototype','v2']`
2. **Name your runs** meaningfully: `wandb.name=proto_5_temp_007`
3. **Group related runs** using the same project name
4. **Add notes** to document experiment setup: `wandb.notes='Testing prototype count'`
5. **Adjust log_interval** to reduce logging overhead: `wandb.log_interval=10`

## Troubleshooting

### "wandb not installed"
```bash
pip install wandb
```

### "API key not found"
```bash
# Login to wandb
wandb login

# Or set API key explicitly
export WANDB_API_KEY=your_key
```

### "Too many API calls"
Increase `wandb.log_interval`:
```bash
python run.py ... wandb.log_interval=10  # Log every 10 iterations
```

## Offline Mode

If you want to log locally and sync later:

```bash
export WANDB_MODE=offline
python run.py -c configs/rd/rd_byol_proto_mvtec.py wandb.enabled=True

# Later, sync to cloud
wandb sync wandb/run-xxx
```

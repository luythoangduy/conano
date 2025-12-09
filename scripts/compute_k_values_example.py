"""
Example script to compute k-values for RD models

Usage:
    python scripts/compute_k_values_example.py --config configs/rd/rd_mvtec.py --output stats/mvtec_k_values.json
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util.compute_k_values import compute_k_values, save_k_values
from model import get_model
from data import get_loader
from util.util import set_seed


def main():
    parser = argparse.ArgumentParser(description='Compute k-values for RD models')
    parser.add_argument('--config', type=str, default='configs/rd/rd_mvtec.py',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='stats/mvtec_k_values.json',
                        help='Output path for k-values JSON file')
    parser.add_argument('--activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'tanh', 'arctan'],
                        help='Activation function type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches to process (default: 100, -1 for all)')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config_path = args.config.replace('/', '.').replace('\\', '.').replace('.py', '')
    if config_path.startswith('.'):
        config_path = config_path[1:]

    # Import config
    exec(f"from {config_path} import cfg")
    cfg_obj = eval('cfg()')

    # Set seed
    set_seed(cfg_obj.seed)

    # Create model
    print("Creating model...")
    model = get_model(cfg_obj.model).to(args.device)
    model.eval()

    # Create dataloader
    print("Creating dataloader...")
    train_loader = get_loader(cfg_obj.data, cfg_obj, mode='train')

    # Compute k-values
    print(f"\nComputing k-values using activation: {args.activation}")
    print(f"Processing batches: {args.num_batches if args.num_batches > 0 else 'all'}")

    # Modify compute_k_values to accept num_batches limit
    from tqdm import tqdm

    channel_values_list = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader, desc="Computing k-values")):
            imgs = batch['img'].to(args.device)

            # Extract features through backbone
            if hasattr(model, 'net_t'):
                feats_backbone = model.net_t(imgs)
                feats_backbone = [f.detach() for f in feats_backbone]
            else:
                raise ValueError("Model must have 'net_t' attribute")

            # For RD models, get merged features
            if hasattr(model, 'proj_layer') and hasattr(model, 'mff_oce'):
                # RD_LGC: pass through proj_layer then MFF_OCE
                feats_proj = model.proj_layer(feats_backbone)
                feats_merge = model.mff_oce(feats_proj)
            elif hasattr(model, 'mff_oce'):
                # RD: pass through MFF_OCE directly
                feats_merge = model.mff_oce(feats_backbone)
            else:
                raise ValueError("Model must have 'mff_oce' attribute")

            # Collect features
            channel_values_list.append(feats_merge.cpu())

            # Limit number of batches
            if args.num_batches > 0 and idx >= args.num_batches - 1:
                break

    # Concatenate and compute statistics
    print("\nComputing statistics...")
    all_features = torch.cat(channel_values_list, dim=0)
    print(f"Collected features shape: {all_features.shape}")

    # Compute statistics per channel
    C = all_features.shape[1]
    all_features_flat = all_features.permute(1, 0, 2, 3).reshape(C, -1)

    channel_mean = all_features_flat.mean(dim=1)
    channel_std = all_features_flat.std(dim=1)
    channel_min = all_features_flat.min(dim=1)[0]
    channel_max = all_features_flat.max(dim=1)[0]

    # k_i = 1 / std_i
    eps = 1e-6
    k_values = 1.0 / (channel_std + eps)

    # Normalize k_values
    k_values = k_values / k_values.mean()

    stats = {
        'activation_type': args.activation,
        'k_values_272': k_values.tolist(),
        'channel_mean': channel_mean.tolist(),
        'channel_std': channel_std.tolist(),
        'channel_min': channel_min.tolist(),
        'channel_max': channel_max.tolist(),
        'num_channels': C,
        'num_samples': all_features.shape[0]
    }

    # Save k-values
    save_k_values(k_values.tolist(), stats, args.output)

    print("\n" + "="*60)
    print("K-values computation completed!")
    print(f"Output saved to: {args.output}")
    print("="*60)
    print("\nTo use these k-values in training:")
    print(f"1. Set 'k_values_path' in your config to: '{args.output}'")
    print(f"2. Or use the config: configs/rd/rd_mvtec_with_kval.py")


if __name__ == '__main__':
    main()
"""
Script to compute k-values for each channel over 1 epoch
k_i = 1 / std(feature_i) for channel i
"""
import torch
import torch.nn as nn
import json
import os
from tqdm import tqdm


def compute_k_values(model, dataloader, device, activation_type='sigmoid'):
    """
    Compute k-values for each channel over 1 epoch

    Args:
        model: The RD model (or any model with net_t and mff_oce)
        dataloader: Training dataloader
        device: torch device
        activation_type: Type of activation function ('sigmoid', 'tanh', 'arctan')

    Returns:
        k_values: List of k-values for each channel
        stats: Dictionary with additional statistics
    """
    print(f"Computing k-values with activation type: {activation_type}")
    model.eval()
    channel_values_list = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing k-values")):
            imgs = batch['img'].to(device)

            # Extract features through backbone
            if hasattr(model, 'net_t'):
                feats_backbone = model.net_t(imgs)
                feats_backbone = [f.detach() for f in feats_backbone]
            else:
                raise ValueError("Model must have 'net_t' attribute")

            # For RD models, we need to get the merged features
            if hasattr(model, 'mff_oce'):
                # For RD: pass through MFF_OCE
                feats_merge = model.mff_oce(feats_backbone)
            elif hasattr(model, 'proj_layer') and hasattr(model, 'mff_oce'):
                # For RD_LGC: pass through proj_layer then MFF_OCE
                feats_proj = model.proj_layer(feats_backbone)
                feats_merge = model.mff_oce(feats_proj)
            else:
                raise ValueError("Model must have 'mff_oce' attribute")

            # B x C x H x W -> collect all spatial values per channel
            channel_values_list.append(feats_merge.cpu())

            # Limit to avoid memory issues
            if idx >= 100:  # Adjust based on your memory
                break

    # Concatenate all batches: [B, C, H, W]
    all_features = torch.cat(channel_values_list, dim=0)
    print(f"Collected features shape: {all_features.shape}")

    # Compute statistics per channel across all spatial locations and batches
    # Reshape to [C, -1]
    C = all_features.shape[1]
    all_features_flat = all_features.permute(1, 0, 2, 3).reshape(C, -1)

    # Compute statistics
    channel_mean = all_features_flat.mean(dim=1)
    channel_std = all_features_flat.std(dim=1)
    channel_min = all_features_flat.min(dim=1)[0]
    channel_max = all_features_flat.max(dim=1)[0]

    # k_i = 1 / std_i (add epsilon to avoid division by zero)
    eps = 1e-6
    k_values = 1.0 / (channel_std + eps)

    # Optionally normalize k_values to a reasonable range
    # This helps with numerical stability
    k_values = k_values / k_values.mean()

    stats = {
        'activation_type': activation_type,
        'k_values_272': k_values.tolist(),
        'channel_mean': channel_mean.tolist(),
        'channel_std': channel_std.tolist(),
        'channel_min': channel_min.tolist(),
        'channel_max': channel_max.tolist(),
        'num_channels': C,
        'num_samples': all_features.shape[0]
    }

    return k_values.tolist(), stats


def save_k_values(k_values, stats, save_path):
    """Save k-values and stats to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    output = {
        'k_values_272': k_values,
        'activation_type': stats['activation_type'],
        'statistics': stats
    }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"K-values saved to: {save_path}")
    print(f"Number of channels: {stats['num_channels']}")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"K-values range: [{min(k_values):.4f}, {max(k_values):.4f}]")
    print(f"K-values mean: {sum(k_values)/len(k_values):.4f}")


def load_k_values(load_path):
    """Load k-values from JSON file"""
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    """
    Example usage:

    from util.compute_k_values import compute_k_values, save_k_values
    from model import get_model
    from data import get_loader

    # Load model
    model = get_model(cfg.model).cuda()
    model.eval()

    # Load data
    train_loader = get_loader(cfg.data, mode='train')

    # Compute k-values
    k_values, stats = compute_k_values(model, train_loader, 'cuda', activation_type='sigmoid')

    # Save k-values
    save_k_values(k_values, stats, 'stats/mvtec_k_values.json')
    """
    pass
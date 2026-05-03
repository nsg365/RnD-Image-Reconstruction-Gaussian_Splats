"""
Extract detailed Gaussian parameters from trained checkpoint.
Saves: coordinates, scales, colors, opacities, sigma, covariance matrices.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path


def extract_gaussian_params(checkpoint_path, output_path='gaussian_params'):
    """
    Extract all Gaussian parameters from checkpoint.
    
    Args:
        checkpoint_path: path to .pt file
        output_path: prefix for output files (creates .npz, .json, .csv)
    
    Returns:
        Dictionary with all parameters
    """
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    means = checkpoint['means'].numpy()  # (N, 2)
    log_scales = checkpoint['log_scales'].numpy()  # (N, 2)
    rotations = checkpoint['rotations'].numpy()  # (N,)
    colors = checkpoint['colors'].numpy()  # (N, 3)
    opacities = checkpoint['opacities'].numpy()  # (N,)
    
    num_gaussians = means.shape[0]
    print(f"Number of Gaussians: {num_gaussians}\n")
    
    # Apply sigmoid to colors and opacities
    colors_sig = 1 / (1 + np.exp(-colors))  # sigmoid
    opacities_sig = 1 / (1 + np.exp(-opacities))  # sigmoid
    
    # Compute sigma from log_scales
    scales = np.exp(log_scales)  # (N, 2)
    
    # Compute covariance matrices for each Gaussian
    covs = []
    for i in range(num_gaussians):
        # Create rotation matrix
        cos_rot = np.cos(rotations[i])
        sin_rot = np.sin(rotations[i])
        R = np.array([
            [cos_rot, -sin_rot],
            [sin_rot, cos_rot]
        ])
        
        # Scale matrix
        S = np.diag(scales[i])
        
        # Covariance: R @ S^2 @ R^T
        cov = R @ (S @ S) @ R.T
        covs.append(cov)
    
    covs = np.array(covs)  # (N, 2, 2)
    
    # Create output structure
    params = {
        'num_gaussians': num_gaussians,
        'means': means,  # (N, 2) - x, y coordinates
        'scales': scales,  # (N, 2) - sigma_x, sigma_y
        'rotations': rotations,  # (N,)
        'colors_raw': colors,  # (N, 3) - raw parameters
        'colors': colors_sig,  # (N, 3) - after sigmoid
        'opacities_raw': opacities,  # (N,)
        'opacities': opacities_sig,  # (N,)
        'covariances': covs,  # (N, 2, 2)
    }
    
    # Save as NPZ (binary, efficient)
    print(f"Saving to {output_path}.npz")
    np.savez_compressed(
        f'{output_path}.npz',
        num_gaussians=num_gaussians,
        means=means,
        scales=scales,
        rotations=rotations,
        colors=colors_sig,
        opacities=opacities_sig,
        covariances=covs
    )
    
    # Save as CSV (human-readable)
    print(f"Saving to {output_path}.csv")
    with open(f'{output_path}.csv', 'w') as f:
        f.write('gaussian_id,x,y,sigma_x,sigma_y,rotation,r,g,b,opacity\n')
        for i in range(num_gaussians):
            f.write(f'{i},'
                   f'{means[i, 0]:.4f},'
                   f'{means[i, 1]:.4f},'
                   f'{scales[i, 0]:.4f},'
                   f'{scales[i, 1]:.4f},'
                   f'{rotations[i]:.4f},'
                   f'{colors_sig[i, 0]:.4f},'
                   f'{colors_sig[i, 1]:.4f},'
                   f'{colors_sig[i, 2]:.4f},'
                   f'{opacities_sig[i]:.4f}\n')
    
    # Save covariances separately as JSON
    print(f"Saving covariance matrices to {output_path}_covs.json")
    covs_dict = {}
    for i in range(num_gaussians):
        covs_dict[f'gaussian_{i}'] = covs[i].tolist()
    with open(f'{output_path}_covs.json', 'w') as f:
        json.dump(covs_dict, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("GAUSSIAN PARAMETERS SUMMARY")
    print("="*80)
    print(f"\nTotal Gaussians: {num_gaussians}\n")
    
    print("Coordinates (x, y):")
    print(f"  X range: [{means[:, 0].min():.2f}, {means[:, 0].max():.2f}]")
    print(f"  Y range: [{means[:, 1].min():.2f}, {means[:, 1].max():.2f}]")
    
    print("\nScales (sigma_x, sigma_y):")
    print(f"  Sigma_X range: [{scales[:, 0].min():.4f}, {scales[:, 0].max():.4f}]")
    print(f"  Sigma_Y range: [{scales[:, 1].min():.4f}, {scales[:, 1].max():.4f}]")
    print(f"  Mean: [{scales[:, 0].mean():.4f}, {scales[:, 1].mean():.4f}]")
    
    print("\nColors (R, G, B):")
    print(f"  R range: [{colors_sig[:, 0].min():.4f}, {colors_sig[:, 0].max():.4f}]")
    print(f"  G range: [{colors_sig[:, 1].min():.4f}, {colors_sig[:, 1].max():.4f}]")
    print(f"  B range: [{colors_sig[:, 2].min():.4f}, {colors_sig[:, 2].max():.4f}]")
    
    print("\nOpacities:")
    print(f"  Range: [{opacities_sig.min():.4f}, {opacities_sig.max():.4f}]")
    print(f"  Mean: {opacities_sig.mean():.4f}")
    
    print("\nCovariance matrices saved for each Gaussian")
    
    print("\n" + "="*80)
    print(f"✓ Output files:")
    print(f"  - {output_path}.npz (binary, for Python)")
    print(f"  - {output_path}.csv (human-readable)")
    print(f"  - {output_path}_covs.json (covariance matrices)")
    print("="*80 + "\n")
    
    return params


def load_gaussian_params(npz_path):
    """Load previously saved Gaussian parameters from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'num_gaussians': int(data['num_gaussians']),
        'means': data['means'],
        'scales': data['scales'],
        'rotations': data['rotations'],
        'colors': data['colors'],
        'opacities': data['opacities'],
        'covariances': data['covariances'],
    }


if __name__ == "__main__":
    # Usage: python extract_gaussian_params.py <checkpoint.pt> [output_prefix]
    
    if len(sys.argv) < 2:
        print("Usage: python extract_gaussian_params.py <checkpoint.pt> [output_prefix]")
        print("\nExample:")
        print("  python extract_gaussian_params.py ./test_outputs/ref_train_gpu/final_model.pt ref_params")
        print("  python extract_gaussian_params.py ./test_outputs/ref_train_gpu/best_model.pt ref_best_params")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'gaussian_params'
    
    extract_gaussian_params(checkpoint_path, output_prefix)
    
    # Show how to load it back
    print("\nTo load the parameters back in Python:")
    print(f"  data = np.load('{output_prefix}.npz')")
    print("  or use: params = load_gaussian_params(...)")

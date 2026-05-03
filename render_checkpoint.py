"""
Load a checkpoint and render the reconstruction.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.gaussian import GaussianSplat2D
from models.render import render_gaussians_efficient
from utils.image_utils import save_image, load_image
import argparse

def render_checkpoint(checkpoint_path, image_path, output_path):
    """
    Load a checkpoint and render the reconstruction.
    
    Args:
        checkpoint_path: path to .pt checkpoint
        image_path: path to original image (for size info)
        output_path: where to save rendered image
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Get image dimensions
    target_image = load_image(image_path, device=device)
    img_height, img_width = target_image.shape[0], target_image.shape[1]
    
    # Infer num_gaussians from checkpoint
    num_gaussians = checkpoint['means'].shape[0]
    print(f"Number of Gaussians: {num_gaussians}")
    print(f"Image size: {img_height} x {img_width}")
    
    # Create model and load checkpoint
    gaussians = GaussianSplat2D(
        num_gaussians=num_gaussians,
        img_height=img_height,
        img_width=img_width,
        device=device
    )
    
    # Load parameters
    with torch.no_grad():
        gaussians.means.data = checkpoint['means'].to(device)
        gaussians.log_scales.data = checkpoint['log_scales'].to(device)
        gaussians.rotations.data = checkpoint['rotations'].to(device)
        gaussians.depths.data = checkpoint['depths'].to(device)
        gaussians.colors.data = checkpoint['colors'].to(device)
        gaussians.opacities.data = checkpoint['opacities'].to(device)
    
    print("Rendering...")
    with torch.no_grad():
        rendered = render_gaussians_efficient(
            gaussians,
            img_height,
            img_width,
            background_color=torch.zeros(3, device=device)
        )
    
    # Save
    save_image(rendered, output_path)
    print(f"Saved rendered image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to original image")
    parser.add_argument("--output", type=str, required=True, help="Output path for rendered image")
    
    args = parser.parse_args()
    render_checkpoint(args.checkpoint, args.image, args.output)

"""
Main training script: Fit 2D Gaussians to image for reconstruction.
Step 1 of the pipeline.
"""

import torch
import torch.optim as optim
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Import custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gaussian import GaussianSplat2D
from models.render import render_gaussians_efficient, compute_ssim
from utils.image_utils import load_image, save_image, get_image_info


def train_gaussians(
    image_path: str,
    num_gaussians: int = 500,
    num_iterations: int = 10000,
    learning_rate: float = 0.01,
    output_dir: str = "./outputs",
    device: str = "cuda",
    max_size: int = None,  # Max dimension for testing (None = full resolution)
    resume: bool = True,  # Resume from checkpoint if it exists
    resume_from: str = "latest"  # "latest" or "best"
):
    """
    Train Gaussian splats to reconstruct target image.
    
    Args:
        image_path: path to target image
        num_gaussians: number of Gaussian splats
        num_iterations: training iterations
        learning_rate: optimizer learning rate
        output_dir: directory to save results
        device: cuda or cpu
        max_size: maximum image dimension for testing (None = full resolution)
        resume: resume from checkpoint if it exists
        resume_from: "latest" to resume training, "best" to start from best model
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # torch.autograd.set_detect_anomaly(True)
    
    print(f"Loading image from {image_path}")
    target_image = load_image(image_path, device=device)
    img_height, img_width = target_image.shape[0], target_image.shape[1]
    
    print(f"Original image shape: {img_height} x {img_width}")
    
    # Optionally resize for faster testing on CPU
    if max_size is not None and max(img_height, img_width) > max_size:
        import torch.nn.functional as F
        scale_factor = max_size / max(img_height, img_width)
        new_height = int(img_height * scale_factor)
        new_width = int(img_width * scale_factor)
        print(f"Resizing to {new_height} x {new_width} for faster testing")
        target_image = F.interpolate(
            target_image.permute(2, 0, 1).unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        img_height, img_width = new_height, new_width
    
    print(f"Training image shape: {img_height} x {img_width}")
    print(f"Initializing {num_gaussians} Gaussians")
    
    # Initialize Gaussians
    gaussians = GaussianSplat2D(
        num_gaussians=num_gaussians,
        img_height=img_height,
        img_width=img_width,
        device=device
    )
    
    # Check for checkpoint and resume
    start_iteration = 0
    checkpoint_filename = "latest_checkpoint.pt" if resume_from == "latest" else "best_model.pt"
    checkpoint_path = output_dir / checkpoint_filename
    
    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        with torch.no_grad():
            gaussians.means.data = checkpoint['means'].to(device)
            gaussians.log_scales.data = checkpoint['log_scales'].to(device)
            gaussians.rotations.data = checkpoint['rotations'].to(device)
            gaussians.depths.data = checkpoint['depths'].to(device)
            gaussians.colors.data = checkpoint['colors'].to(device)
            gaussians.opacities.data = checkpoint['opacities'].to(device)
        
        # Only load iteration number and optimizer state for latest checkpoint
        if resume_from == "latest":
            start_iteration = checkpoint.get('iteration', 0)
            print(f"Resumed from iteration {start_iteration}")
        else:
            print("Loaded best model (starting from iteration 0)")
            start_iteration = 0
    
    # Optimizer
    optimizer = optim.Adam(gaussians.parameters(), lr=learning_rate)
    
    # Load optimizer state only for latest checkpoint
    if resume and resume_from == "latest" and checkpoint_path.exists():
        if 'optimizer_state' in torch.load(checkpoint_path, map_location=device):
            optimizer_checkpoint = torch.load(checkpoint_path, map_location=device)
            optimizer.load_state_dict(optimizer_checkpoint['optimizer_state'])
            print("Loaded optimizer state")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations - start_iteration)
    
    # Loss weights
    lambda_l2 = 1.0
    lambda_ssim = 0.2
    
    print(f"Starting training for {num_iterations - start_iteration} iterations (resuming from iteration {start_iteration})...")
    
    best_loss = float('inf')
    pbar = tqdm(range(start_iteration, num_iterations))
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        # Render with BLACK background (standard for additive Gaussian splatting)
        # Dark regions are created by low alpha/opacity, not by subtracting from white
        rendered = render_gaussians_efficient(
            gaussians,
            img_height,
            img_width,
            background_color=torch.zeros(3, device=device)  # Black background, not white
        )
        
        # Loss computation
        l2_loss = torch.mean((rendered - target_image) ** 2)
        l2_loss = torch.clamp(l2_loss, 0, 10)  # Prevent explosion
        
        # SSIM loss (optional, can be computationally expensive)
        if iteration % 100 == 0:
            ssim = compute_ssim(rendered, target_image)
            ssim_loss = 1 - ssim
            ssim_loss = torch.clamp(ssim_loss, 0, 2)
        else:
            ssim_loss = torch.tensor(0.0, device=device)
        
        total_loss = lambda_l2 * l2_loss + lambda_ssim * ssim_loss
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Clamp parameters to reasonable ranges after optimizer step
        with torch.no_grad():
            gaussians.means.data[:, 0].clamp_(0, img_width - 1e-6)
            gaussians.means.data[:, 1].clamp_(0, img_height - 1e-6)
            gaussians.log_scales.data.clamp_(-10, 10)
            gaussians.depths.data.clamp_(-10, 10)
        
        # Clear cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Logging
        if iteration % 100 == 0:
            pbar.set_description(
                f"L2: {l2_loss.item():.6f}, SSIM: {ssim_loss.item():.6f}, "
                f"Total: {total_loss.item():.6f}"
            )
            
            # Save checkpoint
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    'means': gaussians.means.data.cpu(),
                    'log_scales': gaussians.log_scales.data.cpu(),
                    'rotations': gaussians.rotations.data.cpu(),
                    'depths': gaussians.depths.data.cpu(),
                    'colors': gaussians.colors.data.cpu(),
                    'opacities': gaussians.opacities.data.cpu(),
                }, checkpoint_path)
            
            # Always save latest checkpoint for resuming
            latest_checkpoint_path = output_dir / "latest_checkpoint.pt"
            torch.save({
                'means': gaussians.means.data.cpu(),
                'log_scales': gaussians.log_scales.data.cpu(),
                'rotations': gaussians.rotations.data.cpu(),
                'depths': gaussians.depths.data.cpu(),
                'colors': gaussians.colors.data.cpu(),
                'opacities': gaussians.opacities.data.cpu(),
                'iteration': iteration + 1,
                'optimizer_state': optimizer.state_dict(),
            }, latest_checkpoint_path)
    
    print(f"Training complete! Best loss: {best_loss:.6f}")
    
    # Final rendering
    with torch.no_grad():
        final_rendered = render_gaussians_efficient(
            gaussians,
            img_height,
            img_width,
            background_color=torch.zeros(3, device=device)  # Black background
        )
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save results
    save_image(final_rendered, str(output_dir / "reconstructed.png"))
    save_image(target_image, str(output_dir / "target.png"))
    
    # Compute final metrics
    final_l2 = torch.mean((final_rendered - target_image) ** 2).item()
    final_ssim = compute_ssim(final_rendered, target_image).item()
    
    print(f"\nFinal L2 Loss: {final_l2:.6f}")
    print(f"Final SSIM: {final_ssim:.6f}")
    
    # Save Gaussian parameters
    checkpoint_path = output_dir / "final_model.pt"
    torch.save({
        'means': gaussians.means.data.cpu(),
        'log_scales': gaussians.log_scales.data.cpu(),
        'rotations': gaussians.rotations.data.cpu(),
        'depths': gaussians.depths.data.cpu(),
        'colors': gaussians.colors.data.cpu(),
        'opacities': gaussians.opacities.data.cpu(),
        'img_height': img_height,
        'img_width': img_width,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")
    print(f"Results saved to {output_dir}")
    
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Gaussians for image reconstruction")
    parser.add_argument("--image", type=str, required=True, help="Path to target image")
    parser.add_argument("--num-gaussians", type=int, default=500, help="Number of Gaussians")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--max-size", type=int, default=None, help="Max image size for testing (None=full resolution)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--resume-from", type=str, choices=["latest", "best"], default="latest", 
                        help="Resume from 'latest' checkpoint or 'best' model")
    
    args = parser.parse_args()
    
    train_gaussians(
        image_path=args.image,
        num_gaussians=args.num_gaussians,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        output_dir=args.output,
        device=args.device,
        max_size=args.max_size,
        resume=not args.no_resume,
        resume_from=args.resume_from
    )

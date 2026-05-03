"""
Gaussian rasterization for 2D rendering.
Renders 2D Gaussians to image with alpha compositing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def render_gaussians(
    gaussians,
    img_height: int,
    img_width: int,
    background_color: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Render Gaussian splats to image using alpha compositing.
    
    Args:
        gaussians: GaussianSplat2D instance
        img_height: output image height
        img_width: output image width
        background_color: (3,) background RGB color, defaults to white
    
    Returns:
        torch.Tensor: (img_height, img_width, 3) rendered image
    """
    device = gaussians.means.device
    
    if background_color is None:
        background_color = torch.ones(3, device=device)
    
    # Create coordinate grid
    y_coords = torch.arange(img_height, device=device, dtype=torch.float32)
    x_coords = torch.arange(img_width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
    
    # Stack coordinates: (H, W, 2)
    coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
    
    # Get Gaussian parameters
    means = gaussians.means  # (N, 2)
    cov_inv = gaussians.get_covariance_inv()  # (N, 2, 2)
    cov_det = gaussians.get_covariance_det()  # (N,)
    colors = gaussians.get_colors()  # (N, 3)
    opacities = gaussians.get_opacity()  # (N,)
    
    # Compute Gaussian values at each pixel
    # For each Gaussian: G(x,y) = exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu))
    
    deltas = coords.unsqueeze(2) - means  # (H, W, N, 2)
    
    # Mahalanobis distance: (x - mu)^T * Sigma^-1 * (x - mu)
    # deltas: (H, W, N, 2, 1)
    # cov_inv: (N, 2, 2)
    deltas_expanded = deltas.unsqueeze(-1)  # (H, W, N, 2, 1)
    
    # (H, W, N, 2) @ (N, 2, 2) -> broadcast and multiply
    # Reshape for batch matrix multiplication
    H, W, N = deltas.shape[0], deltas.shape[1], deltas.shape[2]
    deltas_flat = deltas.reshape(H * W, N, 2, 1)  # (H*W, N, 2, 1)
    
    # (H*W, N, 1, 2) @ (N, 2, 2) @ (H*W, N, 2, 1) = (H*W, N, 1, 1)
    deltas_T = deltas.reshape(H * W, N, 1, 2)  # (H*W, N, 1, 2)
    quad_form = torch.matmul(deltas_T, cov_inv)  # (H*W, N, 1, 2)
    quad_form = torch.matmul(quad_form, deltas_flat)  # (H*W, N, 1, 1)
    quad_form = quad_form.squeeze(-1).squeeze(-1)  # (H*W, N)
    quad_form = quad_form.reshape(H, W, N)
    
    # Gaussian values
    gaussian_vals = torch.exp(-0.5 * quad_form)  # (H, W, N)
    
    # Multiply by opacity
    alphas = gaussian_vals * opacities.unsqueeze(0).unsqueeze(0)  # (H, W, N)
    
    # Alpha composite from back to front (proper "over" compositing)
    # C_out = C_prev + T * alpha * color
    # where T is transmittance (product of (1 - alpha) from previous layers)
    # Initialize with background
    output = background_color.unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()  # (H, W, 3)
    T = torch.ones(H, W, 1, device=device)  # Transmittance
    
    for i in range(N):
        alpha_i = alphas[:, :, i:i+1]  # (H, W, 1)
        color_i = colors[i:i+1].unsqueeze(0)  # (1, 1, 3)
        
        # Proper "over" composite: blend color into output based on alpha and transmittance
        # The new color contribution is attenuated by transmittance from previous layers
        output = output + T * alpha_i * color_i
        T = T * (1 - alpha_i)  # Reduce transmittance for next layer
    
    # Ensure output is in valid range [0, 1]
    output = torch.clamp(output, 0, 1)
    
    return output


def render_gaussians_efficient(
    gaussians,
    img_height: int,
    img_width: int,
    background_color: Optional[torch.Tensor] = None,
    threshold: float = 1e-3,
    max_bbox_size: int = 256
) -> torch.Tensor:
    """
    Render Gaussian splats efficiently by only computing within each Gaussian's bounding box.
    
    Args:
        gaussians: GaussianSplat2D instance
        img_height: output image height
        img_width: output image width
        background_color: (3,) background RGB color, defaults to black
        threshold: minimum Gaussian contribution threshold
        max_bbox_size: maximum size of bounding box in any dimension
    
    Returns:
        torch.Tensor: (img_height, img_width, 3) rendered image
    """
    device = gaussians.means.device
    
    if background_color is None:
        background_color = torch.zeros(3, device=device, requires_grad=False)
    
    # Initialize output with background - make sure it's a fresh tensor for gradient tracking
    output = torch.full((img_height, img_width, 3), 0.0, device=device, dtype=torch.float32)
    for c in range(3):
        output[:, :, c] = background_color[c]
    
    T = torch.ones(img_height, img_width, 1, device=device, dtype=torch.float32)
    
    # Get parameters
    means = gaussians.means  # (N, 2)
    cov_inv = gaussians.get_covariance_inv()  # (N, 2, 2)
    colors = gaussians.get_colors()  # (N, 3)
    opacities = gaussians.get_opacity()  # (N,)
    
    # Get bounding boxes
    bboxes = gaussians.get_bounding_boxes(threshold, max_bbox_size)  # (N, 4) [min_x, min_y, max_x, max_y]
    
    # Sort Gaussians by depth (higher depth first, assuming higher depth = back)
    depths = gaussians.depths
    sort_indices = torch.argsort(depths, descending=True)
    
    # Apply sorting
    means = means[sort_indices]  # (N, 2)
    cov_inv = cov_inv[sort_indices]  # (N, 2, 2)
    colors = colors[sort_indices]  # (N, 3)
    opacities = opacities[sort_indices]  # (N,)
    bboxes = bboxes[sort_indices]  # (N, 4)
    
    N = means.shape[0]
    
    for i in range(N):
        min_x, min_y, max_x, max_y = bboxes[i]
        
        # Skip if bounding box is invalid (NaN, inf, or empty)
        if (torch.isnan(min_x) or torch.isnan(min_y) or torch.isnan(max_x) or torch.isnan(max_y) or
            torch.isinf(min_x) or torch.isinf(min_y) or torch.isinf(max_x) or torch.isinf(max_y) or
            min_x >= max_x or min_y >= max_y):
            continue
        
        # Skip if bounding box is empty or out of bounds
        if max_x <= 0 or max_y <= 0 or min_x >= img_width or min_y >= img_height:
            continue
        
        # Clamp to image bounds
        min_x = max(0, int(min_x))
        max_x = min(img_width, int(max_x))
        min_y = max(0, int(min_y))
        max_y = min(img_height, int(max_y))
        
        # Skip if still empty
        if min_x >= max_x or min_y >= max_y:
            continue
        
        # Create local coordinate grid for this bounding box
        local_height = max_y - min_y
        local_width = max_x - min_x
        
        y_coords = torch.arange(min_y, max_y, device=device, dtype=torch.float32)
        x_coords = torch.arange(min_x, max_x, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        local_coords = torch.stack([xx, yy], dim=-1)  # (local_H, local_W, 2)
        
        # Compute deltas from Gaussian mean
        deltas = local_coords - means[i]  # (local_H, local_W, 2)
        
        # Compute Mahalanobis distance
        deltas_expanded = deltas.unsqueeze(-1)  # (local_H, local_W, 2, 1)
        quad_form = torch.matmul(deltas_expanded.transpose(-2, -1), cov_inv[i])  # (local_H, local_W, 1, 2)
        quad_form = torch.matmul(quad_form, deltas_expanded)  # (local_H, local_W, 1, 1)
        quad_form = quad_form.squeeze(-1).squeeze(-1)  # (local_H, local_W)
        
        # Clamp to prevent extreme values
        quad_form = torch.clamp(quad_form, -50, 50)
        
        # Gaussian values
        gaussian_vals = torch.exp(-0.5 * quad_form)
        
        # Multiply by opacity
        alphas = gaussian_vals * opacities[i]
        
        # Composite into the full image
        # The new color contribution is attenuated by transmittance from previous layers
        color_i = colors[i].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        alphas_expanded = alphas.unsqueeze(-1)  # (local_H, local_W, 1)
        
        # Get current values (clone to avoid in-place issues)
        current_output = output[min_y:max_y, min_x:max_x].clone()
        current_T = T[min_y:max_y, min_x:max_x].clone()
        
        # Compute new values
        new_contribution = current_T * alphas_expanded * color_i
        new_output = current_output + new_contribution
        new_T = current_T * (1 - alphas_expanded)
        
        # Update (this is still in-place, but let's see if it works)
        output[min_y:max_y, min_x:max_x] = new_output
        T[min_y:max_y, min_x:max_x] = new_T
    
    # Ensure output is in valid range [0, 1]
    output = torch.clamp(output, 0, 1)
    
    return output
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1, img2: (H, W, 3) images
        window_size: Gaussian window size
    
    Returns:
        SSIM value between -1 and 1
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Convert to grayscale if needed
    if img1.dim() == 3:
        img1 = img1.mean(dim=-1, keepdim=True)
        img2 = img2.mean(dim=-1, keepdim=True)
    
    # Compute means
    mu1 = F.avg_pool2d(img1.permute(2, 0, 1).unsqueeze(0), window_size, padding=window_size//2).squeeze()
    mu2 = F.avg_pool2d(img2.permute(2, 0, 1).unsqueeze(0), window_size, padding=window_size//2).squeeze()
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d((img1.squeeze() ** 2).unsqueeze(0).unsqueeze(0), window_size, padding=window_size//2).squeeze() - mu1_sq
    sigma2_sq = F.avg_pool2d((img2.squeeze() ** 2).unsqueeze(0).unsqueeze(0), window_size, padding=window_size//2).squeeze() - mu2_sq
    sigma12 = F.avg_pool2d((img1.squeeze() * img2.squeeze()).unsqueeze(0).unsqueeze(0), window_size, padding=window_size//2).squeeze() - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()
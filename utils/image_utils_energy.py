"""
Utility functions for image loading, preprocessing, and Gaussian fitting.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict
import torchvision.transforms as transforms


def load_image(image_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load image and convert to tensor.
    
    Args:
        image_path: path to image file
        device: device to load tensor to (default: "cpu", use "cuda" if available)
    
    Returns:
        torch.Tensor: (H, W, 3) normalized image in [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).to(device)
    return image


def save_image(tensor: torch.Tensor, output_path: str) -> None:
    """
    Save tensor as image file.
    
    Args:
        tensor: (H, W, 3) tensor with values in [0, 1]
        output_path: path to save image
    """
    # Clamp and convert to uint8
    image = torch.clamp(tensor, 0, 1)
    image = (image * 255).byte()
    image = image.cpu().numpy()
    
    Image.fromarray(image, 'RGB').save(output_path)


def get_image_info(image_path: str) -> Tuple[int, int]:
    """Get image height and width without loading full tensor."""
    image = Image.open(image_path)
    return image.height, image.width


def compute_image_energy(
    image: torch.Tensor,
    smooth_kernel_size: int = 9,
    gradient_weight: float = 0.7,
    variance_weight: float = 0.3,
    energy_power: float = 1.0,
    energy_floor: float = 0.02
) -> torch.Tensor:
    """
    Compute a per-pixel sampling energy map for Gaussian initialization.

    High-gradient and locally varied regions receive more energy, while flat
    regions like clear sky receive only the small uniform floor.

    Args:
        image: (H, W, 3) image tensor in [0, 1]
        smooth_kernel_size: local averaging window for stable energy
        gradient_weight: contribution from edges/texture gradients
        variance_weight: contribution from local RGB variance
        energy_power: >1 sharpens focus on detailed regions, <1 flattens it
        energy_floor: uniform probability mixed into every pixel

    Returns:
        torch.Tensor: (H, W) normalized energy map that sums to 1
    """
    if image.dim() != 3 or image.shape[-1] != 3:
        raise ValueError("Expected image tensor with shape (H, W, 3)")

    energy_floor = min(max(float(energy_floor), 0.0), 1.0)
    device = image.device
    image_nchw = image.permute(2, 0, 1).unsqueeze(0)
    gray = image.mean(dim=-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_energy = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    grad_energy = grad_energy.squeeze(0).squeeze(0)

    kernel_size = max(1, int(smooth_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2

    local_mean = F.avg_pool2d(image_nchw, kernel_size, stride=1, padding=padding)
    local_sq_mean = F.avg_pool2d(image_nchw.square(), kernel_size, stride=1, padding=padding)
    variance_energy = torch.clamp(local_sq_mean - local_mean.square(), min=0.0)
    variance_energy = variance_energy.mean(dim=1).squeeze(0)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        return x / (x.max() + 1e-8)

    energy = gradient_weight * normalize(grad_energy) + variance_weight * normalize(variance_energy)
    energy = normalize(energy)
    energy = torch.pow(energy + 1e-8, max(1e-6, energy_power))
    energy = energy / (energy.sum() + 1e-8)

    floor = torch.full_like(energy, 1.0 / energy.numel())
    energy = (1.0 - energy_floor) * energy + energy_floor * floor
    energy = torch.clamp(energy, min=0.0)
    return energy / (energy.sum() + 1e-8)


def sample_points_from_energy(
    energy: torch.Tensor,
    num_points: int
) -> torch.Tensor:
    """
    Sample image-space points from a normalized energy map.

    Args:
        energy: (H, W) non-negative energy/probability map
        num_points: number of points to sample

    Returns:
        torch.Tensor: (num_points, 2) points as x/y image coordinates
    """
    if energy.dim() != 2:
        raise ValueError("Expected energy tensor with shape (H, W)")

    height, width = energy.shape
    flat_probs = energy.reshape(-1)
    flat_probs = flat_probs / (flat_probs.sum() + 1e-8)
    replacement = num_points > flat_probs.numel()
    indices = torch.multinomial(flat_probs, num_points, replacement=replacement)

    ys = torch.div(indices, width, rounding_mode='floor').float()
    xs = (indices % width).float()

    jitter = torch.rand(num_points, 2, device=energy.device)
    xs = torch.clamp(xs + jitter[:, 0], 0, width - 1e-6)
    ys = torch.clamp(ys + jitter[:, 1], 0, height - 1e-6)

    return torch.stack([xs, ys], dim=1)


def create_mask_from_region(
    img_height: int,
    img_width: int,
    region_coords: Tuple[int, int, int, int],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create binary mask from region coordinates.
    
    Args:
        img_height, img_width: image dimensions
        region_coords: (y1, x1, y2, x2) bounding box
        device: device
    
    Returns:
        torch.Tensor: (H, W) binary mask
    """
    mask = torch.zeros(img_height, img_width, device=device)
    y1, x1, y2, x2 = region_coords
    mask[y1:y2, x1:x2] = 1.0
    return mask


def get_gaussian_subset_by_mask(
    gaussians,
    mask: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Get indices of Gaussians that are mostly within masked region.
    
    Args:
        gaussians: GaussianSplat2D instance
        mask: (H, W) binary mask
        threshold: fraction of Gaussian that must be in mask
    
    Returns:
        torch.Tensor: indices of selected Gaussians
    """
    means = gaussians.means  # (N, 2)
    
    # Simple heuristic: select Gaussians whose centers are in mask
    indices = []
    for i in range(gaussians.num_gaussians):
        x, y = means[i]
        x, y = int(x.item()), int(y.item())
        
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            if mask[y, x] > threshold:
                indices.append(i)
    
    if len(indices) == 0:
        return torch.tensor([], dtype=torch.long, device=gaussians.means.device)
    
    return torch.tensor(indices, dtype=torch.long, device=gaussians.means.device)


def extract_gaussian_subset(gaussians, indices: torch.Tensor) -> Optional[Dict]:
    """
    Extract a subset of Gaussians by indices.
    
    Returns a dict with subset parameters.
    """
    if len(indices) == 0:
        return None
    
    subset = {
        'means': gaussians.means[indices].clone(),
        'log_scales': gaussians.log_scales[indices].clone(),
        'rotations': gaussians.rotations[indices].clone(),
        'colors': gaussians.colors[indices].clone(),
        'opacities': gaussians.opacities[indices].clone(),
    }
    return subset


def apply_transform_to_gaussians(
    gaussian_subset: Dict,
    transform: torch.Tensor,
    device: str = "cuda"
) -> Dict:
    """
    Apply 2D affine transform to Gaussian subset.
    
    Args:
        gaussian_subset: dict with Gaussian parameters
        transform: (2, 3) affine transform matrix
        device: device
    
    Returns:
        dict: transformed Gaussian parameters
    """
    # Transform means: [x, y] -> [x', y']
    means = gaussian_subset['means'].to(device)  # (N, 2)
    ones = torch.ones(means.shape[0], 1, device=device)
    means_homog = torch.cat([means, ones], dim=1)  # (N, 3)
    
    # Apply transform
    transform = transform.to(device)
    means_transformed = (transform @ means_homog.T).T  # (N, 2)
    
    # Update subset
    transformed_subset = {
        'means': means_transformed,
        'log_scales': gaussian_subset['log_scales'].to(device),
        'rotations': gaussian_subset['rotations'].to(device),
        'colors': gaussian_subset['colors'].to(device),
        'opacities': gaussian_subset['opacities'].to(device),
    }
    
    return transformed_subset


def compute_image_correspondence(
    ref_image: torch.Tensor,
    target_image: torch.Tensor,
    ref_region: Tuple[int, int, int, int],
    target_region: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Compute affine transform between reference and target regions.
    Simple implementation using center and scale.
    
    Args:
        ref_image, target_image: images
        ref_region: (y1, x1, y2, x2) in reference
        target_region: (y1, x1, y2, x2) in target
    
    Returns:
        torch.Tensor: (2, 3) affine transform
    """
    # Get region centers and sizes
    ref_y1, ref_x1, ref_y2, ref_x2 = ref_region
    tar_y1, tar_x1, tar_y2, tar_x2 = target_region
    
    ref_center = torch.tensor([(ref_x1 + ref_x2) / 2, (ref_y1 + ref_y2) / 2], dtype=torch.float32)
    tar_center = torch.tensor([(tar_x1 + tar_x2) / 2, (tar_y1 + tar_y2) / 2], dtype=torch.float32)
    
    ref_size = torch.tensor([ref_x2 - ref_x1, ref_y2 - ref_y1], dtype=torch.float32)
    tar_size = torch.tensor([tar_x2 - tar_x1, tar_y2 - tar_y1], dtype=torch.float32)
    
    # Compute scale
    scale = tar_size / (ref_size + 1e-6)
    
    # Identity transform with scale and translation
    transform = torch.zeros(2, 3, dtype=torch.float32)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[0, 2] = tar_center[0] - ref_center[0] * scale[0]
    transform[1, 2] = tar_center[1] - ref_center[1] * scale[1]
    
    return transform

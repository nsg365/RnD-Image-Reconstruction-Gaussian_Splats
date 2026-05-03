"""
Utility functions for image loading, preprocessing, and Gaussian fitting.
"""

import torch
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

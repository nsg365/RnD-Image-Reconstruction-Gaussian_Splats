"""
2D Gaussian Splat representation for image editing.
Based on MiraGe paper: flat Gaussians constrained to 2D image plane.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GaussianSplat2D(nn.Module):
    """
    2D Gaussian Splat constrained to image plane.
    
    Each Gaussian is parameterized as:
    - means: (x, y) position in image coordinates
    - covariance: 2x2 symmetric matrix controlling spread (via eigenvectors + scales)
    - color: RGB values
    - opacity: alpha value in [0, 1]
    """
    
    def __init__(self, num_gaussians: int, img_height: int, img_width: int, device: str = "cuda"):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.img_height = img_height
        self.img_width = img_width
        self.device = device
        
        # Initialize Gaussian parameters
        # means: (num_gaussians, 2) - x, y positions
        means = torch.rand(num_gaussians, 2, device=device)
        means[:, 0] *= img_width
        means[:, 1] *= img_height
        self.register_parameter("means", nn.Parameter(means))
        
        # log_scales: (num_gaussians, 2) - log of eigenvalues for covariance
        # Initialize to reasonable size (sqrt of image size / 10)
        log_scales = torch.ones(num_gaussians, 2, device=device) * np.log(max(img_height, img_width) / 20)
        self.register_parameter("log_scales", nn.Parameter(log_scales))
        
        # rotation: (num_gaussians,) - rotation angle in radians for covariance eigenvectors
        rotations = torch.zeros(num_gaussians, device=device)
        self.register_parameter("rotations", nn.Parameter(rotations))
        
        # depths: (num_gaussians,) - depth for compositing order (lower depth = front)
        depths = torch.randn(num_gaussians, device=device)
        self.register_parameter("depths", nn.Parameter(depths))
        
        # colors: (num_gaussians, 3) - RGB in [0, 1]
        colors = torch.rand(num_gaussians, 3, device=device)
        self.register_parameter("colors", nn.Parameter(colors))
        
        # opacities: (num_gaussians,) - alpha in [0, 1]
        opacities = torch.ones(num_gaussians, device=device) * 0.5
        self.register_parameter("opacities", nn.Parameter(opacities))
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute covariance matrices from log_scales and rotations.
        
        Returns:
            torch.Tensor: (num_gaussians, 2, 2) covariance matrices
        """
        # Get scales from log_scales
        scales = torch.exp(self.log_scales)  # (num_gaussians, 2)
        
        # Create diagonal scale matrices
        scale_matrices = torch.diag_embed(scales)  # (num_gaussians, 2, 2)
        
        # Create rotation matrices
        cos_rot = torch.cos(self.rotations)  # (num_gaussians,)
        sin_rot = torch.sin(self.rotations)  # (num_gaussians,)
        
        rot_matrices = torch.stack([
            torch.stack([cos_rot, -sin_rot], dim=1),
            torch.stack([sin_rot, cos_rot], dim=1)
        ], dim=1)  # (num_gaussians, 2, 2)
        
        # Covariance: R @ S @ S^T @ R^T
        # But since we have independent scales, it's R @ S^2 @ R^T
        scale_sq = scale_matrices @ scale_matrices  # (num_gaussians, 2, 2)
        cov = rot_matrices @ scale_sq @ rot_matrices.transpose(-2, -1)  # (num_gaussians, 2, 2)
        
        return cov
    
    def get_covariance_inv(self) -> torch.Tensor:
        """Get inverse covariance matrices for rendering."""
        cov = self.get_covariance_matrices()
        # Add small regularization to prevent singular matrices
        cov = cov + 1e-6 * torch.eye(2, device=cov.device).unsqueeze(0)
        cov_inv = torch.linalg.inv(cov)  # (num_gaussians, 2, 2)
        return cov_inv
    
    def get_covariance_det(self) -> torch.Tensor:
        """Get determinant of covariance matrices."""
        cov = self.get_covariance_matrices()
        det = torch.det(cov)  # (num_gaussians,)
        return det
    
    def get_opacity(self) -> torch.Tensor:
        """Get opacity values in [0, 1]."""
        return torch.sigmoid(self.opacities)
    
    def get_colors(self) -> torch.Tensor:
        """Get color values in [0, 1]."""
        return torch.sigmoid(self.colors)
    
    def translate(self, delta: torch.Tensor) -> None:
        """
        Translate all Gaussians by delta.
        
        Args:
            delta: (2,) or (num_gaussians, 2) translation vector(s)
        """
        with torch.no_grad():
            if delta.dim() == 1:
                self.means.data += delta.unsqueeze(0)
            else:
                self.means.data += delta
    
    def scale_gaussians(self, scale_factor: float) -> None:
        """
        Scale all Gaussian sizes by scale_factor.
        
        Args:
            scale_factor: multiplicative scale
        """
        with torch.no_grad():
            self.log_scales.data += np.log(scale_factor)
    
    def get_bounding_boxes(self, threshold: float = 1e-3, max_bbox_size: int = 256) -> torch.Tensor:
        """
        Compute axis-aligned bounding boxes for each Gaussian.
        
        The bounding box covers the region where the Gaussian value > threshold.
        For a 2D Gaussian, this is an ellipse, and we compute the AABB of that ellipse.
        
        Args:
            threshold: minimum Gaussian value to include (default 1e-3)
            max_bbox_size: maximum size of bounding box in any dimension (default 256)
            
        Returns:
            torch.Tensor: (num_gaussians, 4) bounding boxes as [min_x, min_y, max_x, max_y]
        """
        # Get covariance matrices
        cov = self.get_covariance_matrices()  # (N, 2, 2)
        
        # For threshold, we need the Mahalanobis distance where exp(-0.5 * dist^2) = threshold
        # So -0.5 * dist^2 = log(threshold)
        # dist^2 = -2 * log(threshold)
        dist_sq = -2 * torch.log(torch.tensor(threshold, device=self.device))
        
        # The bounding box of the ellipse is mean ± sqrt(dist_sq) * sqrt(eigenvalues) in eigenvector directions
        # But for axis-aligned box, we need to consider the maximum extent in x and y
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(cov)  # eigenvals: (N, 2), eigenvecs: (N, 2, 2)
        
        # The semi-axis lengths are sqrt(dist_sq * eigenvals)
        semi_axes = torch.sqrt(torch.clamp(dist_sq * eigenvals, min=1e-8))  # Prevent NaN from negative values
        
        # For axis-aligned bounding box of rotated ellipse
        # The half-extent in x direction: |a * cos(theta)| + |b * sin(theta)|
        # where a, b are semi-axes, theta is rotation
        # eigenvecs[:, 0] is the direction of the first eigenvector
        cos_theta = eigenvecs[:, 0, 0]  # cos of angle for first eigenvector
        sin_theta = eigenvecs[:, 0, 1]  # sin of angle for first eigenvector
        
        # Half extents
        half_width = semi_axes[:, 0] * torch.abs(cos_theta) + semi_axes[:, 1] * torch.abs(sin_theta)
        half_height = semi_axes[:, 0] * torch.abs(sin_theta) + semi_axes[:, 1] * torch.abs(cos_theta)
        
        # Clamp extents to prevent huge bounding boxes
        half_width = torch.clamp(half_width, 0, max_bbox_size / 2)
        half_height = torch.clamp(half_height, 0, max_bbox_size / 2)
        
        # Bounding box
        min_x = self.means[:, 0] - half_width
        max_x = self.means[:, 0] + half_width
        min_y = self.means[:, 1] - half_height
        max_y = self.means[:, 1] + half_height
        
        # Clip to image bounds
        min_x = torch.clamp(min_x, 0, self.img_width)
        max_x = torch.clamp(max_x, 0, self.img_width)
        min_y = torch.clamp(min_y, 0, self.img_height)
        max_y = torch.clamp(max_y, 0, self.img_height)
        
        return torch.stack([min_x, min_y, max_x, max_y], dim=1)  # (N, 4)
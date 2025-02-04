#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import colorsys
import numpy as np
from .gpu_manager import GPUManager

class ImageProcessor:
    def __init__(self, height, width, kernel_size, gaussian_size):
        self.height = height
        self.width = width
        self.kernels = self._initialize_kernels(kernel_size, gaussian_size)
        self.color_map = {}

    def _initialize_kernels(self, kernel_size, gaussian_size):
        """Initialize all required kernels"""
        kernels = {
            'isolation': torch.ones(kernel_size, kernel_size),
            'sobel_x': torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        }
        kernels['sobel_y'] = kernels['sobel_x'].t()
        kernels['gaussian'] = self._create_gaussian_kernel(gaussian_size)
        return kernels

    def _create_gaussian_kernel(self, size):
        """Create Gaussian kernel for depth smoothing"""
        sigma = size / 6.0
        center = size // 2
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return torch.tensor(kernel, dtype=torch.float32)

    def get_kernel(self, name, device):
        """Get kernel and move to specified device"""
        if name in self.kernels:
            return GPUManager.to_device(self.kernels[name], device)
        return None

    def generate_color(self, leaf_id):
        """Generate distinct color for leaf visualization"""
        if leaf_id not in self.color_map:
            golden_ratio = 0.618033988749895
            hue = (leaf_id * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self.color_map[leaf_id] = tuple(int(255 * x) for x in rgb)
        return self.color_map[leaf_id]

    def calculate_centroid(self, leaf_mask):
        """Calculate centroid of leaf mask"""
        y_indices, x_indices = torch.where(leaf_mask)
        centroid_x = float(x_indices.float().mean())
        centroid_y = float(y_indices.float().mean())
        return (centroid_x, centroid_y)

    def smooth_depth(self, depth_patch, device):
        """Apply Gaussian smoothing to depth data"""
        gaussian_kernel = self.get_kernel('gaussian', device)
        depth_patch = GPUManager.to_device(depth_patch, device)
        padded_depth = F.pad(depth_patch.unsqueeze(0).unsqueeze(0), 
                           (gaussian_kernel.shape[0]//2,)*4, mode='reflect')
        return F.conv2d(padded_depth, 
                       gaussian_kernel.view(1, 1, *gaussian_kernel.shape), 
                       padding=0).squeeze()
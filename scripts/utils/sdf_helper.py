#!/usr/bin/env python3
import numpy as np
import torch
import skfmm
import cv2
import rospy
from .gpu_manager import GPUManager

class SDFHelper:
    def __init__(self, device):
        self.device = device
        
    def calculate_global_sdf(self, mask_tensor):
        """Calculate global SDF for scene understanding"""
        try:
            # Convert tensor to numpy for skfmm
            mask_np = mask_tensor.cpu().numpy()
            
            # Create binary mask (1 for no leaf, 0 for leaf)
            binary_mask = (mask_np == 0).astype(np.float32)
            
            # Calculate SDF
            sdf = skfmm.distance(binary_mask, dx=1)
            
            # Convert back to tensor
            sdf_tensor = torch.from_numpy(sdf).float().to(self.device)
            
            # Get global minima and maxima
            min_global = torch.argmin(sdf_tensor)
            max_global = torch.argmax(sdf_tensor)
            
            return sdf_tensor, min_global, max_global
            
        except Exception as e:
            rospy.logerr(f"Error in SDF calculation: {str(e)}")
            return None, None, None

    def calculate_leaf_sdf(self, leaf_mask):
        """Calculate SDF for individual leaf"""
        try:
            leaf_mask_np = leaf_mask.cpu().numpy()
            leaf_binary = (leaf_mask_np == 0).astype(np.float32)
            leaf_sdf = skfmm.distance(leaf_binary, dx=1)
            return torch.from_numpy(leaf_sdf).float().to(self.device)
        except Exception as e:
            rospy.logerr(f"Error in leaf SDF calculation: {str(e)}")
            return None

    def get_sdf_approach_vector(self, leaf_mask, grasp_point_2d):
        """Calculate approach vector using SDF gradients"""
        try:
            # Calculate SDF
            sdf = self.calculate_leaf_sdf(leaf_mask)
            if sdf is None:
                return None

            # Calculate gradients
            sobelx = cv2.Sobel(sdf.cpu().numpy(), cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(sdf.cpu().numpy(), cv2.CV_64F, 0, 1, ksize=3)
            
            # Get gradient at grasp point
            x, y = grasp_point_2d
            gradient_x = sobelx[y, x]
            gradient_y = sobely[y, x]
            
            # Normalize gradient
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            if magnitude > 0:
                gradient_x /= magnitude
                gradient_y /= magnitude
            
            return gradient_x, gradient_y
            
        except Exception as e:
            rospy.logerr(f"Error in approach vector calculation: {str(e)}")
            return None

    def calculate_sdf_isolation_score(self, leaf_mask, all_masks):
        """Calculate isolation score using SDF"""
        try:
            # Calculate SDF for all leaves
            global_sdf, _, _ = self.calculate_global_sdf(all_masks)
            if global_sdf is None:
                return None
                
            # Get SDF values for current leaf
            leaf_values = global_sdf[leaf_mask.bool()]
            
            # Calculate mean SDF value for the leaf
            mean_sdf = torch.mean(leaf_values)
            
            # Normalize score
            max_sdf = torch.max(global_sdf)
            isolation_score = mean_sdf / max_sdf
            
            return isolation_score
            
        except Exception as e:
            rospy.logerr(f"Error in isolation score calculation: {str(e)}")
            return None
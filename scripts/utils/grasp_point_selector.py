#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import rospy
import cv2
import os
from .gpu_manager import GPUManager
from .ml_grasp_optimizer.data_collector import EnhancedGraspDataCollector
from .ml_grasp_optimizer.model import GraspQualityPredictor
from .ml_grasp_optimizer.trainer import GraspPointCNN

class GraspPointSelector:
    def __init__(self, device):
        self.device = device
        
        # Main scoring weights
        self.flatness_weight = 0.25      # Reduced from 0.3
        self.isolation_weight = 0.4      # Reduced from 0.5
        self.edge_weight = 0.2          # Reduced from 0.20
        self.accessibility_weight = 0.15     # Edge awareness
        
        # Parameters for point selection
        self.min_flat_area_size = 15     # Minimum size for grasp area
        self.min_edge_distance = 20      # Minimum pixels from edge
        self.isolation_radius = 50       # Radius to check for nearby leaves
        
        # Camera parameters from calibration
        self.camera_cx = 707  # optical center x from K matrix
        self.camera_cy = 494  # optical center y from K matrix
        self.f_norm = None    # will be set from projection matrix
        self.erosion_kernel_size = 21  # Increased from 15
        self.erosion_iterations = 2  

        #self.data_collector = EnhancedGraspDataCollector()
        self.ml_predictor = None

        # Add ML model setup
        self.device = device
        self.ml_predictor = GraspPointCNN(in_channels=9).to(device)  # Change to ml_predictor
        self.load_ml_model()

    def load_ml_model(self):
        """Load trained ML model"""
        try:
            model_path = os.path.expanduser('~/leaf_grasp_output/ml_models/best_model.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.ml_predictor.load_state_dict(checkpoint['model_state_dict'])  # Use ml_predictor
                self.ml_predictor.eval()
                rospy.loginfo("Loaded ML grasp model successfully")
            else:
                rospy.logwarn("No ML model found, will use traditional scoring only")
                self.ml_predictor = None  # Set to None if no model found
        except Exception as e:
            rospy.logerr(f"Error loading ML model: {str(e)}")
            self.ml_predictor = None

    def get_ml_score(self, leaf_mask, depth_tensor, scores, point):
        """Get ML model prediction for a patch"""
        try:
            if self.ml_predictor is None:
                return None
                
            x, y = point
            patch_size = 32
            
            # Extract and validate patches
            depth_patch = self._extract_local_patch(depth_tensor, x, y, patch_size)
            if depth_patch is None:
                rospy.logwarn(f"Failed to extract depth patch at ({x}, {y})")
                return None
                
            mask_patch = self._extract_local_patch(leaf_mask, x, y, patch_size)
            if mask_patch is None:
                rospy.logwarn(f"Failed to extract mask patch at ({x}, {y})")
                return None
            
            # Move tensors to device and add dimensions
            depth_patch = torch.as_tensor(depth_patch, device=self.device).float()
            mask_patch = torch.as_tensor(mask_patch, device=self.device).float()
            
            # Normalize depth patch
            depth_min = depth_patch.min()
            depth_max = depth_patch.max()
            if depth_max > depth_min:
                depth_patch = (depth_patch - depth_min) / (depth_max - depth_min)
            
            # Add batch and channel dimensions
            depth_patch = depth_patch.unsqueeze(0).unsqueeze(0)
            mask_patch = mask_patch.unsqueeze(0).unsqueeze(0)
            
            # Extract score patches
            score_patches = []
            required_scores = [
                'sdf_score', 'approach_score', 'flatness_map',
                'isolation_map', 'distance_map', 'accessibility_map',
                'stem_penalty'
            ]
            
            for score_name in required_scores:
                if score_name not in scores:
                    rospy.logwarn(f"Missing required score: {score_name}")
                    return None
                    
                score_map = scores[score_name]
                score_patch = self._extract_local_patch(
                    torch.from_numpy(score_map) if isinstance(score_map, np.ndarray) else score_map,
                    x, y, patch_size
                )
                
                if score_patch is None:
                    rospy.logwarn(f"Failed to extract {score_name} patch")
                    return None
                    
                # Convert to tensor and normalize
                score_patch = torch.as_tensor(score_patch, device=self.device).float()
                score_min = score_patch.min()
                score_max = score_patch.max()
                if score_max > score_min:
                    score_patch = (score_patch - score_min) / (score_max - score_min)
                    
                score_patches.append(score_patch)
            
            # Stack all features
            score_tensor = torch.stack(score_patches).unsqueeze(0)
            features = torch.cat([depth_patch, mask_patch, score_tensor], dim=1)
            
            # Get prediction
            with torch.no_grad():
                self.ml_predictor.eval()
                output = self.ml_predictor(features)
                score = torch.sigmoid(output).item()
                
                # Apply non-linear scaling to spread scores
                score = np.tanh(score * 3.0) * 0.5 + 0.5
                
                rospy.loginfo(f"Point ({x}, {y}) - Raw ML score: {output.item():.3f}, Final score: {score:.3f}")
                return score
                
        except Exception as e:
            rospy.logerr(f"Error in ML scoring: {str(e)}")
            return None
    
    def set_camera_params(self, projection_matrix):
        """Set camera parameters from projection matrix"""
        self.f_norm = projection_matrix[0, 0]
        self.camera_cx = projection_matrix[0, 2]
        self.camera_cy = projection_matrix[1, 2]
        self.baseline = -projection_matrix[0, 3] / self.f_norm

    def get_3d_grasp_point(self, grasp_point_2d, depth_tensor, pcl_data=None):
        """Get 3D coordinates of grasp point using both depth and point cloud"""
        u, v = grasp_point_2d
        
        # Get depth value at grasp point
        depth_value = depth_tensor[v, u].item()
        
        # Calculate 3D point from depth
        X = (depth_value * (u - self.camera_cx)) / self.f_norm
        Y = (depth_value * (v - self.camera_cy)) / self.f_norm
        Z = depth_value
        
        # Verify with point cloud if available
        pcd_point = None
        if pcl_data is not None:
            index = v * self.width + u
            if 0 <= index < len(pcl_data):
                pcd_point = pcl_data[index]
                
                # Compare and log any significant discrepancy
                depth_point = np.array([X, Y, Z])
                pcd_point = np.array(pcd_point)
                diff = np.linalg.norm(depth_point - pcd_point)
                if diff > 0.01:  # 1cm threshold
                    rospy.logwarn(f"Discrepancy between depth and point cloud: {diff}m")
                    # Use point cloud data if significant difference
                    X, Y, Z = pcd_point
        
        return (X, Y, Z)
    
    
    # This Funtion combines the traditional scire and ML score (7:3)
    # def select_grasp_point(self, leaf_mask, depth_tensor, image_processor, pcl_data=None):
    #     """Select optimal grasp point using combined traditional and ML approach"""
    #     try:
    #         leaf_mask_np = leaf_mask.cpu().numpy().astype(np.uint8)
            
    #         # Calculate scores and valid regions
    #         scores = self._calculate_all_scores(leaf_mask_np, depth_tensor, image_processor)
    #         valid_regions = self._get_valid_regions(leaf_mask_np, scores)
            
    #         # Get candidate points
    #         candidate_points = self._get_candidate_points(
    #             scores['traditional_score'], 
    #             valid_regions,
    #             top_k=20,
    #             min_distance=10
    #         )
            
    #         if not candidate_points:
    #             rospy.logwarn("No valid candidate points found")
    #             return None, None, None
                
    #         best_point = candidate_points[0]  # Default to highest traditional score
    #         best_score = scores['traditional_score'][best_point[1], best_point[0]]
    #         ml_used = False
            
    #         # Try ML evaluation if available
    #         if self.ml_predictor is not None and len(candidate_points) > 1:
    #             rospy.loginfo(f"Evaluating {len(candidate_points)} points with ML model")
                
    #             for point in candidate_points:
    #                 x, y = point
    #                 trad_score = scores['traditional_score'][y, x]
                    
    #                 # Get ML score
    #                 ml_score = self.get_ml_score(leaf_mask, depth_tensor, scores, point)
                    
    #                 if ml_score is not None:
    #                     # Calculate confidence-based weights
    #                     ml_conf = 1.0 - abs(ml_score - 0.5) * 2
    #                     ml_weight = min(0.3, ml_conf * 0.6)  # Cap at 0.3
    #                     trad_weight = 1.0 - ml_weight
                        
    #                     combined_score = trad_weight * trad_score + ml_weight * ml_score
                        
    #                     rospy.loginfo(f"Point ({x}, {y}):")
    #                     rospy.loginfo(f"  Traditional: {trad_score:.3f}")
    #                     rospy.loginfo(f"  ML Score: {ml_score:.3f} (conf: {ml_conf:.3f})")
    #                     rospy.loginfo(f"  Combined: {combined_score:.3f}")
                        
    #                     if combined_score > best_score:
    #                         best_score = combined_score
    #                         best_point = point
    #                         ml_used = True
    #                         rospy.loginfo("  New best point!")
            
    #         # Calculate 3D points
    #         grasp_point_2d = best_point
    #         grasp_point_3d = self.get_3d_grasp_point(grasp_point_2d, depth_tensor, pcl_data)
    #         pre_grasp_point = self.calculate_pre_grasp_point(grasp_point_3d, leaf_mask_np)
            
    #         if ml_used:
    #             rospy.loginfo("Final point selected with ML influence")
    #         else:
    #             rospy.loginfo("Final point selected using traditional scoring only")
                
    #         return grasp_point_2d, grasp_point_3d, pre_grasp_point
            
    #     except Exception as e:
    #         rospy.logerr(f"Error in grasp point selection: {str(e)}")
    #         return None, None, None
            
    
    def _calculate_all_scores(self, leaf_mask_np, depth_tensor, image_processor):
        """Calculate all scores needed for grasp point selection"""
        scores = {
            'sdf_score': self.calculate_sdf_score(leaf_mask_np),
            'approach_score': self.calculate_approach_vector_score(leaf_mask_np, depth_tensor),
            'flatness_map': self._calculate_flatness_map(
                depth_tensor * torch.from_numpy(leaf_mask_np).float().to(self.device), 
                image_processor
            ).cpu().numpy(),
            'isolation_map': self._calculate_isolation_score(leaf_mask_np),
            'distance_map': cv2.distanceTransform(leaf_mask_np, cv2.DIST_L2, 5),
            'accessibility_map': self._calculate_accessibility_score(leaf_mask_np),
            'stem_penalty': self._calculate_stem_penalty(leaf_mask_np).astype(np.float32)
        }
        
        # Calculate combined traditional score
        traditional_score = (
            0.4 * scores['approach_score'] +
            0.3 * scores['sdf_score'] +
            0.2 * scores['flatness_map'] +
            0.1 * scores['accessibility_map']
        ) * (1 - scores['stem_penalty'])
        
        scores['traditional_score'] = traditional_score
        return scores

    def _get_valid_regions(self, leaf_mask_np, scores):
        """Get valid regions for grasp point selection"""
        return (
            (scores['distance_map'] > self.min_edge_distance) &
            (leaf_mask_np > 0) &
            (scores['stem_penalty'] < 0.8)
        )
                
    # This Function give grasping pint based on only ML model       
    def select_grasp_point(self, leaf_mask, depth_tensor, image_processor, pcl_data=None):
        """Select grasp point using ML model only"""
        try:
            leaf_mask_np = leaf_mask.cpu().numpy().astype(np.uint8)
            
            # Calculate scores for ML features
            scores = {
                'sdf_score': self.calculate_sdf_score(leaf_mask_np),
                'approach_score': self.calculate_approach_vector_score(leaf_mask_np, depth_tensor),
                'flatness_map': self._calculate_flatness_map(
                    depth_tensor * leaf_mask.float(), image_processor).cpu().numpy(),
                'isolation_map': self._calculate_isolation_score(leaf_mask_np),
                'distance_map': cv2.distanceTransform(leaf_mask_np, cv2.DIST_L2, 5),
                'accessibility_map': self._calculate_accessibility_score(leaf_mask_np),
                'stem_penalty': self._calculate_stem_penalty(leaf_mask_np).astype(np.float32)
            }

            if self.ml_predictor is not None:
                # Get points where leaf mask is True
                y_coords, x_coords = np.where(leaf_mask_np > 0)
                
                best_score = float('-inf')
                best_point = None
                
                rospy.loginfo("Evaluating points with ML model...")
                
                # Sample fewer points for efficiency
                step = max(1, len(y_coords) // 100)  # Sample ~100 points
                for i in range(0, len(y_coords), step):
                    x, y = x_coords[i], y_coords[i]
                    
                    # Check boundaries for patch extraction
                    patch_size = 32
                    half_size = patch_size // 2
                    
                    if (y < half_size or y >= leaf_mask_np.shape[0] - half_size or 
                        x < half_size or x >= leaf_mask_np.shape[1] - half_size):
                        continue
                    
                    # Extract patches with proper dimensionality
                    depth_patch = depth_tensor[y-half_size:y+half_size, x-half_size:x+half_size].clone()
                    mask_patch = leaf_mask[y-half_size:y+half_size, x-half_size:x+half_size].clone()
                    
                    # Skip if patches are empty
                    if depth_patch.numel() == 0 or mask_patch.numel() == 0:
                        continue
                    
                    # Extract score patches
                    score_patches = []
                    for score_name in scores:
                        score_map = scores[score_name]
                        patch = score_map[y-half_size:y+half_size, x-half_size:x+half_size]
                        score_patches.append(torch.from_numpy(patch).float().to(self.device))
                    
                    # Prepare input for ML model
                    model_input = {
                        'depth_patch': depth_patch.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                        'mask_patch': mask_patch.unsqueeze(0).unsqueeze(0),
                        'score_patches': torch.stack(score_patches).unsqueeze(0)
                    }
                    
                    # Get ML prediction
                    with torch.no_grad():
                        self.ml_predictor.eval()
                        score = self.ml_predictor(torch.cat([
                            model_input['depth_patch'],
                            model_input['mask_patch'],
                            model_input['score_patches']
                        ], dim=1))
                        score = torch.sigmoid(score).item()
                    
                    rospy.loginfo(f"Point ({x}, {y}) ML score: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_point = (x, y)
                
                if best_point is not None:
                    grasp_point_2d = best_point
                    rospy.loginfo(f"ML Model selected point {grasp_point_2d} with score {best_score:.3f}")
                else:
                    rospy.logwarn("ML failed to find valid point, using centroid")
                    grasp_point_2d = image_processor.calculate_centroid(leaf_mask)
                    grasp_point_2d = (int(grasp_point_2d[0]), int(grasp_point_2d[1]))
            else:
                rospy.logwarn("No ML model available!")
                grasp_point_2d = image_processor.calculate_centroid(leaf_mask)
                grasp_point_2d = (int(grasp_point_2d[0]), int(grasp_point_2d[1]))

            # Calculate 3D points
            grasp_point_3d = self.get_3d_grasp_point(grasp_point_2d, depth_tensor, pcl_data)
            pre_grasp_point = self.calculate_pre_grasp_point(grasp_point_3d, leaf_mask_np)

            return grasp_point_2d, grasp_point_3d, pre_grasp_point
                
        except Exception as e:
            rospy.logerr(f"Error in ML grasp point selection: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None, None, None
                
    def _extract_local_patch(self, tensor, x, y, size):
        """Extract local patch around point with exact size control"""
        try:
            half_size = size // 2
            h, w = tensor.shape[-2:] if isinstance(tensor, torch.Tensor) else tensor.shape
            
            # Calculate valid ranges
            x1 = max(0, x - half_size)
            x2 = min(w, x + half_size)  # Remove the +1 that was causing oversized patches
            y1 = max(0, y - half_size)
            y2 = min(h, y + half_size)
            
            # Extract base patch
            if isinstance(tensor, torch.Tensor):
                patch = tensor[y1:y2, x1:x2].clone()  # Use clone() to avoid reference issues
            else:
                patch = tensor[y1:y2, x1:x2].copy()  # Use copy() for numpy arrays
            
            # Calculate padding sizes
            pad_left = half_size - (x - x1)
            pad_right = half_size - (x2 - x)
            pad_top = half_size - (y - y1)
            pad_bottom = half_size - (y2 - y)
            
            # Verify we have valid padding values
            if any(p < 0 for p in [pad_left, pad_right, pad_top, pad_bottom]):
                rospy.logwarn(f"Invalid padding values: l={pad_left}, r={pad_right}, t={pad_top}, b={pad_bottom}")
                return None
            
            # Apply padding
            if pad_left + pad_right + pad_top + pad_bottom > 0:
                if isinstance(tensor, torch.Tensor):
                    patch = torch.nn.functional.pad(
                        patch.unsqueeze(0).unsqueeze(0),
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='replicate'
                    ).squeeze(0).squeeze(0)
                else:
                    patch = np.pad(
                        patch,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='edge'
                    )
            
            # Final size verification
            if patch.shape != (size, size):
                rospy.logwarn(f"Incorrect patch size {patch.shape}, expected ({size}, {size})")
                return None
                
            return patch
            
        except Exception as e:
            rospy.logerr(f"Error in patch extraction: {str(e)}")
            return None
    
    def _get_candidate_points(self, score_map, valid_regions, top_k=20, min_distance=10):
        """Get top-k candidate points with minimum spacing"""
        try:
            # Apply valid regions to score map
            valid_scores = score_map * valid_regions
            
            # Get indices sorted by score
            flat_indices = np.argsort(valid_scores.ravel())[::-1]
            
            candidates = []
            used_positions = np.zeros_like(valid_regions, dtype=bool)
            
            for idx in flat_indices:
                if len(candidates) >= top_k:
                    break
                    
                # Convert flat index to 2D
                y, x = np.unravel_index(idx, score_map.shape)
                
                # Check if this point is too close to any existing candidate
                if not used_positions[
                    max(0, y - min_distance):min(y + min_distance + 1, score_map.shape[0]),
                    max(0, x - min_distance):min(x + min_distance + 1, score_map.shape[1])
                ].any():
                    candidates.append((int(x), int(y)))
                    # Mark region as used
                    used_positions[
                        max(0, y - min_distance):min(y + min_distance + 1, score_map.shape[0]),
                        max(0, x - min_distance):min(x + min_distance + 1, score_map.shape[1])
                    ] = True
            
            return candidates
            
        except Exception as e:
            rospy.logerr(f"Error getting candidate points: {str(e)}")
            return []
        
    def _extract_patches(self, x, y, patch_size, leaf_mask, depth_tensor, scores):
        """Extract feature patches around a point"""
        half_size = patch_size // 2
        patches = {}
        
        # Extract all relevant patches
        for name, score_map in scores.items():
            if isinstance(score_map, np.ndarray):
                patch = score_map[y-half_size:y+half_size, x-half_size:x+half_size]
                if patch.shape == (patch_size, patch_size):
                    patches[f'{name}_patch'] = torch.from_numpy(patch).float()
        
        # Add mask and depth patches
        patches['mask_patch'] = leaf_mask[y-half_size:y+half_size, x-half_size:x+half_size]
        patches['depth_patch'] = depth_tensor[y-half_size:y+half_size, x-half_size:x+half_size]
        
        return patches
        
    def _calculate_accessibility_score(self, leaf_mask_np):
        """Calculate accessibility score based on distance from camera origin"""
        height, width = leaf_mask_np.shape
        
        # Create distance map from camera origin
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_from_origin = np.sqrt(
            (x_grid - self.camera_cx)**2 + 
            (y_grid - self.camera_cy)**2
        )
        
        # Normalize distances
        max_dist = np.sqrt(width**2 + height**2)
        accessibility_map = 1 - (dist_from_origin / max_dist)
        
        # Create directional preference (favor points in front of camera)
        angle_map = np.arctan2(y_grid - self.camera_cy, x_grid - self.camera_cx)
        forward_preference = np.cos(angle_map)  # Higher score for points in front
        
        # Combine distance and direction scores
        accessibility_score = (0.7 * accessibility_map + 0.3 * forward_preference) * leaf_mask_np
        
        return accessibility_score    
    
    def calculate_sdf_score(self, leaf_mask_np):
        """Modified SDF score that penalizes deep interior points"""
        # Calculate distance transform
        dist_inside = cv2.distanceTransform(leaf_mask_np, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - leaf_mask_np, cv2.DIST_L2, 5)
        sdf = dist_inside - dist_outside
        
        # Create a penalty for deep interior points
        # This will create a "ridge" of high scores near the edges but not at them
        optimal_distance = 20  # pixels from edge
        interior_penalty = np.exp(-((dist_inside - optimal_distance) ** 2) / (2 * optimal_distance ** 2))
        
        # Normalize SDF
        sdf = sdf / np.max(np.abs(sdf))
        
        # Calculate approach vector score
        y_coords, x_coords = np.indices(leaf_mask_np.shape)
        vectors_to_camera = np.stack([
            x_coords - self.camera_cx,
            y_coords - self.camera_cy
        ], axis=-1)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors_to_camera, axis=-1)
        norms[norms == 0] = 1
        vectors_to_camera = vectors_to_camera / norms[..., np.newaxis]
        
        # Calculate angle with leaf orientation
        angle, _, _, center = self.estimate_leaf_orientation(leaf_mask_np)
        if angle is not None:
            leaf_direction = np.array([np.cos(angle), np.sin(angle)])
            # Prefer grasp points where approach vector is perpendicular to leaf direction
            alignment_score = np.abs(np.cross(vectors_to_camera, leaf_direction))
        else:
            alignment_score = np.ones_like(sdf)
        
        # Combine scores
        final_score = (0.4 * interior_penalty + 
                    0.4 * alignment_score + 
                    0.2 * sdf) * leaf_mask_np
        
        return final_score
    
    def calculate_approach_vector_score(self, leaf_mask_np, depth_tensor):
        """Calculate score based on approach vector quality"""
        height, width = leaf_mask_np.shape
        y_coords, x_coords = np.indices((height, width))
        
        # Calculate vectors from camera to each point
        vectors_to_point = np.stack([
            (x_coords - self.camera_cx),
            (y_coords - self.camera_cy),
            np.full((height, width), self.f_norm)  # Z component
        ], axis=-1)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors_to_point, axis=-1)
        norms[norms == 0] = 1
        vectors_to_point = vectors_to_point / norms[..., np.newaxis]
        
        # Calculate angle with vertical (prefer more vertical approaches)
        vertical = np.array([0, 0, 1])
        angle_with_vertical = np.abs(np.dot(vectors_to_point, vertical))
        
        # Penalize points that require steep approach angles
        approach_score = angle_with_vertical * leaf_mask_np
        
        return approach_score

    def _calculate_isolation_score(self, leaf_mask):
        """Calculate isolation score with height preference"""
        try:
            height, width = leaf_mask.shape
            
            # Create smaller kernels for memory efficiency
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            kernel_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
            
            # Calculate base isolation score
            current_leaf = leaf_mask.astype(np.uint8)
            all_leaves = (leaf_mask > 0).astype(np.uint8)
            other_leaves = all_leaves - current_leaf
            
            # Close proximity score
            interference_close = cv2.dilate(other_leaves, kernel_close)
            dist_close = cv2.distanceTransform((1 - interference_close), cv2.DIST_L2, 3)
            score_close = dist_close / (np.max(dist_close) + 1e-6)
            
            # Wider area score
            interference_wide = cv2.dilate(other_leaves, kernel_wide)
            dist_wide = cv2.distanceTransform((1 - interference_wide), cv2.DIST_L2, 3)
            score_wide = dist_wide / (np.max(dist_wide) + 1e-6)
            
            # Combine isolation scores
            isolation_score = 0.7 * score_close + 0.3 * score_wide
            
            # Create height preference (favor upper parts of leaf)
            y_coords = np.linspace(1.0, 0.2, height)[:, np.newaxis]  # Less severe dropoff (0.2 instead of 0)
            height_preference = np.tile(y_coords, (1, width))
            
            # Combine isolation score with height preference
            final_score = isolation_score * height_preference * current_leaf
            
            return final_score
            
        except Exception as e:
            rospy.logerr(f"Error in isolation score calculation: {str(e)}")
            return np.zeros_like(leaf_mask, dtype=np.float32)

    def _calculate_flatness_map(self, depth_patch, image_processor):
        """
        Calculate flatness score using depth gradients
        """
        # Smooth depth data
        depth_patch = image_processor.smooth_depth(depth_patch, self.device)
        
        # Get sobel kernels for gradient calculation
        sobel_x = image_processor.get_kernel('sobel_x', self.device)
        sobel_y = image_processor.get_kernel('sobel_y', self.device)
        
        # Calculate depth gradients
        depth_input = depth_patch.unsqueeze(0).unsqueeze(0)
        padded_depth = F.pad(depth_input, (1,1,1,1), mode='reflect')
        
        dx = F.conv2d(padded_depth, sobel_x.unsqueeze(0).unsqueeze(0)).squeeze()
        dy = F.conv2d(padded_depth, sobel_y.unsqueeze(0).unsqueeze(0)).squeeze()
        
        # Calculate flatness score (inverse of gradient magnitude)
        gradient_magnitude = torch.sqrt(dx**2 + dy**2)
        flatness_score = torch.exp(-gradient_magnitude * 5)  # Scale factor of 5
        
        return flatness_score
    
    def _calculate_position_map(self, leaf_mask):
        """Calculate position score favoring points closer to camera view"""
        height = leaf_mask.shape[0]
        y_grid = torch.arange(height, device=self.device).float()
        y_grid = y_grid.view(-1, 1).expand(-1, leaf_mask.shape[1])
        
        # Create exponential decay from top of image
        position_score = torch.exp(-2.0 * y_grid / height)
        
        return position_score
    
    def _find_valid_grasp_regions(self, score_map, flatness_mask):
        """Find regions large enough for grasping"""
        kernel_size = self.min_flat_area_size
        kernel = torch.ones(kernel_size, kernel_size, device=self.device)
        
        # Pad the mask for convolution
        padded_mask = F.pad(flatness_mask.unsqueeze(0).unsqueeze(0),
                           (kernel_size//2,)*4, mode='constant')
        
        # Count flat pixels in neighborhood
        neighborhood_count = F.conv2d(padded_mask.float(), 
                                    kernel.view(1, 1, kernel_size, kernel_size)).squeeze()
        
        # Points with enough flat neighbors are valid
        valid_points = neighborhood_count >= (kernel_size * kernel_size * 0.8)
        
        return valid_points
    
    def _calculate_stem_penalty(self, leaf_mask):
        """Penalize points near stem junction"""
        # Find the bottom region of leaf (probable stem area)
        bottom_region = np.zeros_like(leaf_mask)
        h, w = leaf_mask.shape
        bottom_third = h // 3
        bottom_region[-bottom_third:, :] = 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        # Calculate intersection of leaf mask with bottom region
        masked_bottom = leaf_mask & bottom_region
        stem_area = cv2.dilate(masked_bottom, kernel) & leaf_mask
        
        return stem_area.astype(np.float32)

    def _calculate_tip_penalty(self, leaf_mask):
        """Penalize points at leaf tips"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded = cv2.erode(leaf_mask, kernel)
        tips = leaf_mask - eroded
        tip_area = cv2.dilate(tips, kernel) & leaf_mask
        
        # Create a gradient from tips to inside
        distance_from_tips = cv2.distanceTransform((~tip_area).astype(np.uint8), cv2.DIST_L2, 5)
        # Normalize distance
        max_dist = np.max(distance_from_tips) + 1e-6
        penalty = 1 - (distance_from_tips / max_dist)
        
        return penalty * leaf_mask
    
    def estimate_leaf_orientation(self, leaf_mask_np):
        """Estimate the orientation of the leaf using PCA and contours"""
        try:
            # Get contours
            contours, _ = cv2.findContours(
                leaf_mask_np.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
            
            if not contours:
                return None, None, None, None
                
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Get oriented bounding box
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            size = rect[1]
            angle = rect[2]
            
            # Convert angle to proper range
            if size[0] < size[1]:
                angle = angle + 90
                
            # Get major and minor axes
            major_axis = max(size[0], size[1])
            minor_axis = min(size[0], size[1])
            
            return np.deg2rad(angle), major_axis, minor_axis, center
            
        except Exception as e:
            rospy.logerr(f"Error in leaf orientation estimation: {str(e)}")
            return None, None, None, None
        
    def calculate_pre_grasp_point(self, grasp_point_3d, leaf_mask_np):
        """
        Calculate pre-grasp point that:
        1. Lies on line between camera origin and grasp point
        2. Is not on any leaf
        3. Maintains similar Z coordinate
        4. Has significant distance from grasp point
        """
        try:
            # Camera origin is at (0,0,0) in camera frame
            camera_origin = np.array([0, 0, 0])
            grasp_point = np.array(grasp_point_3d)
            
            # Get vector from camera to grasp point
            direction = grasp_point - camera_origin
            direction = direction / np.linalg.norm(direction)
            
            # Find all leaf contours to check for collision
            contours, _ = cv2.findContours(leaf_mask_np.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            # Dilate mask to create clearance from leaves
            clearance = 15  # pixels - increased clearance
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance*2+1, clearance*2+1))
            dilated_mask = cv2.dilate(leaf_mask_np, kernel)
            
            # Start from a much larger minimum distance
            min_distance = 0.05  # 15cm minimum distance
            max_distance = 0.10  # 25cm maximum distance
            step_size = 0.01    # 1cm steps
            
            # Start from larger distance and move towards camera
            for distance in np.arange(min_distance, max_distance, step_size):
                # Calculate test point
                test_point = (
                    grasp_point_3d[0] - direction[0] * distance,
                    grasp_point_3d[1] - direction[1] * distance,
                    grasp_point_3d[2]  # Keep similar Z
                )
                
                # Project to 2D image space
                test_2d = self._project_point_to_2d(test_point)
                
                # Skip if outside image bounds
                if not (0 <= test_2d[0] < leaf_mask_np.shape[1] and 
                        0 <= test_2d[1] < leaf_mask_np.shape[0]):
                    continue
                    
                # Check against dilated mask for clearance
                if dilated_mask[test_2d[1], test_2d[0]] == 0:
                    # Add extra verification for minimum distance
                    dist_to_grasp = np.linalg.norm(np.array(test_point) - np.array(grasp_point_3d))
                    if dist_to_grasp >= min_distance:
                        return test_point
                    
            # If no valid point found, return point at max distance
            return (
                grasp_point_3d[0] - direction[0] * max_distance,
                grasp_point_3d[1] - direction[1] * max_distance,
                grasp_point_3d[2]
            )
                
        except Exception as e:
            rospy.logerr(f"Error calculating pre-grasp point: {str(e)}")
            return None

    def _project_point_to_2d(self, point_3d):
        """Project 3D point to 2D image coordinates"""
        x, y, z = point_3d
        u = int((x * self.f_norm / z) + self.camera_cx)
        v = int((y * self.f_norm / z) + self.camera_cy)
        return (u, v) 
    

    def detect_midrib(self, leaf_mask_np, raw_image):
        """Improved midrib detection using multiple image processing techniques"""
        try:
            # 1. Extract leaf region and enhance features
            leaf_region = cv2.bitwise_and(raw_image, raw_image, mask=leaf_mask_np)
            
            # Convert to grayscale
            gray = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2GRAY)
            
            # 2. Enhance midrib features
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Create green channel mask (midrib usually appears different in green channel)
            green_channel = raw_image[:,:,1] * leaf_mask_np
            _, green_thresh = cv2.threshold(green_channel, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Edge detection with different methods
            # Gradient
            sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Canny edges
            edges = cv2.Canny(enhanced, 50, 150)
            
            # 4. Get leaf orientation for guidance
            angle, major_axis, minor_axis, center = self.estimate_leaf_orientation(leaf_mask_np)
            if angle is None:
                return None
                
            # 5. Create ridge detection mask
            ridge_mask = np.zeros_like(enhanced)
            center = (int(center[0]), int(center[1]))
            dx = int(major_axis/2 * np.cos(angle))
            dy = int(major_axis/2 * np.sin(angle))
            
            # Create search region along predicted midrib
            mask_width = int(minor_axis/6)  # Narrow search region
            cv2.line(ridge_mask, 
                    (center[0] - dx, center[1] - dy),
                    (center[0] + dx, center[1] + dy), 
                    255, mask_width)
            
            # 6. Find strongest ridge within search region
            window_width = mask_width
            max_intensity_points = []
            
            # Sample points along the predicted midrib line
            for t in np.linspace(0, 1, 20):
                x = int(center[0] - dx + 2*dx*t)
                y = int(center[1] - dy + 2*dy*t)
                
                if 0 <= x < leaf_mask_np.shape[1] and 0 <= y < leaf_mask_np.shape[0]:
                    # Get perpendicular line
                    perp_dx = -dy / np.sqrt(dx*dx + dy*dy) * window_width
                    perp_dy = dx / np.sqrt(dx*dx + dy*dy) * window_width
                    
                    # Sample intensity profile perpendicular to midrib
                    intensities = []
                    positions = []
                    
                    for s in np.linspace(-1, 1, window_width):
                        sample_x = int(x + s*perp_dx)
                        sample_y = int(y + s*perp_dy)
                        
                        if (0 <= sample_x < leaf_mask_np.shape[1] and 
                            0 <= sample_y < leaf_mask_np.shape[0]):
                            if leaf_mask_np[sample_y, sample_x]:
                                intensities.append(enhanced[sample_y, sample_x])
                                positions.append((sample_x, sample_y))
                    
                    if intensities:
                        # Find position of maximum intensity (ridge)
                        max_idx = np.argmax(intensities)
                        max_intensity_points.append(positions[max_idx])
            
            if len(max_intensity_points) < 2:
                return None
                
            # 7. Fit smooth curve through detected points
            points = np.array(max_intensity_points)
            
            # Get endpoints for the line
            start_point = tuple(map(int, points[0]))
            end_point = tuple(map(int, points[-1]))
            
            return start_point, end_point
            
        except Exception as e:
            rospy.logerr(f"Error in midrib detection: {str(e)}")
            return None
        
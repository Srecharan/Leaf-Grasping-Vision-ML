import numpy as np
import torch
import cv2
import skfmm
import rospy
from paretoset import paretoset
import torch.nn.functional as F
from .gpu_manager import GPUManager

class OptimalLeafSelector:
    def __init__(self, device):
        self.device = device
        
        # Camera parameters (will be set later)
        self.camera_cx = None
        self.camera_cy = None
        self.f_norm = None
        
    def set_camera_params(self, projection_matrix):
        """Set camera parameters from projection matrix"""
        self.f_norm = projection_matrix[0, 0]
        self.camera_cx = projection_matrix[0, 2]
        self.camera_cy = projection_matrix[1, 2]
    
    def select_optimal_leaf(self, mask_tensor, depth_tensor):
        """Enhanced leaf selection with tall leaf consideration"""
        try:
            # Convert tensors to numpy
            mask_np = mask_tensor.cpu().numpy()
            depth_np = depth_tensor.cpu().numpy()
            
            leaf_ids = torch.unique(mask_tensor)[1:]  # Skip background (0)
            candidates = []
            
            # Calculate tall leaves like in old pipeline
            depth_list = []
            leaf_masks = []
            depth_np = depth_tensor.cpu().numpy()
            
            # First pass - get depth statistics
            for leaf_id in leaf_ids:
                leaf_mask = (mask_tensor == leaf_id).cpu().numpy()
                leaf_masks.append(leaf_mask)
                leaf_depths = depth_np[leaf_mask]
                if len(leaf_depths) > 0:
                    # Use median instead of mean for robustness
                    depth_list.append(np.median(leaf_depths))
            
            if not depth_list:
                return None
                
            # Calculate depth statistics like old pipeline
            depth_array = np.array(depth_list)
            depth_mean = np.mean(depth_array)
            # depth_std = np.std(depth_array)
            # GRASPER_CLEARANCE = 0.02  # 2cm clearance
            
            # Find tall leaves - all leaves closer than average
            tall_leaves = []
            for i, depth in enumerate(depth_list):
                if depth < depth_mean:  # Closer than average
                    tall_leaves.append(leaf_ids[i].item())
                    
            rospy.loginfo(f"Found {len(tall_leaves)} tall leaves (average depth: {depth_mean:.3f}m)")
                
            # Calculate SDF for clutter once
            cleaned_masks = mask_np >= 1
            cleaned_masks = np.where(cleaned_masks, cleaned_masks == 0, 1)
            global_sdf = skfmm.distance(cleaned_masks, dx=1)
            min_global = np.unravel_index(global_sdf.argmin(), global_sdf.shape)
            max_global = np.unravel_index(global_sdf.argmax(), global_sdf.shape)
            
            # Process each leaf
            for idx, leaf_id in enumerate(leaf_ids):
                try:
                    leaf_mask = leaf_masks[idx]
                    
                    # Check minimum area requirement
                    leaf_area = np.sum(leaf_mask)
                    if leaf_area < 10000:  # Minimum area threshold
                        continue
                    
                    # Get centroid for distance calculations
                    y_indices, x_indices = np.where(leaf_mask)
                    if len(y_indices) == 0:
                        continue
                        
                    centroid = (np.mean(x_indices), np.mean(y_indices))
                    
                    # Calculate clutter score using distances from SDF extrema
                    dist_to_min = np.sqrt((centroid[0] - min_global[1])**2 + 
                                        (centroid[1] - min_global[0])**2)
                    dist_to_max = np.sqrt((centroid[0] - max_global[1])**2 + 
                                        (centroid[1] - max_global[0])**2)
                    
                    # Normalize distances
                    total_dist = dist_to_min + dist_to_max
                    if total_dist > 0:
                        clutter_score = dist_to_min / total_dist
                    else:
                        clutter_score = 0
                    
                    # Calculate 3D distance from camera
                    leaf_depths = depth_np[leaf_mask]
                    mean_depth = np.mean(leaf_depths)
                    
                    # Convert to 3D coordinates
                    X = (mean_depth * (x_indices - self.camera_cx)) / self.f_norm
                    Y = (mean_depth * (y_indices - self.camera_cy)) / self.f_norm
                    Z = np.full_like(X, mean_depth)
                    
                    # Calculate Euclidean distance
                    distances = np.sqrt(X**2 + Y**2 + Z**2)
                    mean_distance = np.mean(distances)
                    
                    # Normalize distance score (closer = better)
                    distance_score = np.exp(-mean_distance / 0.3)  # 30cm scale factor
                    
                    # Visibility score remains the same
                    visibility_score = self._calculate_visibility_score(leaf_mask)
                    
                    is_tall = leaf_id.item() in tall_leaves
                    
                    # Store scores for Pareto optimization
                    candidates.append({
                        'leaf_id': leaf_id.item(),
                        'scores': np.array([
                            clutter_score,     # Higher is better (more isolated)
                            distance_score,    # Higher is better (closer to camera)
                            visibility_score   # Higher is better
                        ], dtype=np.float64),
                        'raw_scores': {
                            'clutter': clutter_score,
                            'distance': mean_distance,
                            'visibility': visibility_score
                        },
                        'is_tall': is_tall
                    })
                    
                except Exception as e:
                    rospy.logerr(f"Error processing leaf {leaf_id.item()}: {str(e)}")
                    continue
            
            if not candidates:
                rospy.logwarn("No valid leaf candidates found")
                return None
            
            try:
                # Separate tall and regular leaves
                tall_candidates = [c for c in candidates if c['is_tall']]
                regular_candidates = [c for c in candidates if not c['is_tall']]
                
                # First try to select from tall leaves if available
                if tall_candidates:
                    scores = np.stack([c['scores'] for c in tall_candidates])
                    # Give preference to tall leaves
                    scores = scores * 1.1  # 20% bonus for tall leaves
                    pareto_mask = paretoset(scores, sense=['max', 'max', 'max'])
                    pareto_candidates = [c for i, c in enumerate(tall_candidates) if pareto_mask[i]]
                else:
                    # Otherwise use regular leaves
                    scores = np.stack([c['scores'] for c in regular_candidates])
                    pareto_mask = paretoset(scores, sense=['max', 'max', 'max'])
                    pareto_candidates = [c for i, c in enumerate(regular_candidates) if pareto_mask[i]]
                
                if not pareto_candidates:
                    pareto_candidates = tall_candidates if tall_candidates else regular_candidates
                    
                # Select best candidate based on weighted score
                weights = np.array([0.35, 0.35, 0.3])  # Clutter, distance, visibility
                best_score = float('-inf')
                best_leaf = None
                best_raw_scores = None
                self._tall_leaves = tall_leaves
                
                for candidate in pareto_candidates:
                    weighted_score = np.sum(weights * candidate['scores'])
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_leaf = candidate['leaf_id']
                        best_raw_scores = candidate['raw_scores']

                
                # Log selection results
                # Add debugging visualization
                if best_raw_scores:
                    rospy.loginfo("\nScoring Details:")
                    rospy.loginfo(f"Number of candidates: {len(candidates)}")
                    rospy.loginfo(f"Number of tall leaves: {len(tall_leaves)}")
                    rospy.loginfo(f"Depth range: {np.min(depth_list):.3f}m to {np.max(depth_list):.3f}m")
                    rospy.loginfo(f"Selected leaf depth: {best_raw_scores['distance']:.3f}m")
                
                return best_leaf
                
            except Exception as e:
                rospy.logerr(f"Error in Pareto optimization: {str(e)}")
                if candidates:
                    return max(candidates, key=lambda x: np.mean(x['scores']))['leaf_id']
                return None
                
        except Exception as e:
            rospy.logerr(f"Error in leaf selection: {str(e)}")
            return None
        
    def get_tall_leaves(self):
        """Return list of currently identified tall leaves"""
        return self._tall_leaves if hasattr(self, '_tall_leaves') else []    
        
    def _calculate_clutter_score(self, all_masks, leaf_mask):
        """Enhanced clutter score using SDF from old pipeline"""
        try:
            # First calculate global SDF
            cleaned_masks = all_masks >= 1
            cleaned_masks = np.where(cleaned_masks, cleaned_masks == 0, 1)
            global_sdf = skfmm.distance(cleaned_masks, dx=1)
            
            # Calculate local SDF for current leaf
            leaf_sdf = skfmm.distance(~leaf_mask, dx=1)
            
            # Get centroids and extrema
            y_indices, x_indices = np.where(leaf_mask)
            if len(y_indices) == 0:
                return 0.0
                
            centroid = (np.mean(x_indices), np.mean(y_indices))
            
            # Find local maxima/minima in leaf region
            leaf_region = leaf_mask * global_sdf
            local_min = np.unravel_index(np.argmin(leaf_region[leaf_mask]), leaf_region.shape)
            local_max = np.unravel_index(np.argmax(leaf_region[leaf_mask]), leaf_region.shape)
            
            # Calculate distances
            dist_to_min = np.sqrt((centroid[0] - local_min[1])**2 + 
                                (centroid[1] - local_min[0])**2)
            dist_to_max = np.sqrt((centroid[0] - local_max[1])**2 + 
                                (centroid[1] - local_max[0])**2)
            
            # Weight by global SDF value
            sdf_weight = np.mean(global_sdf[leaf_mask])
            
            score = (dist_to_min / (dist_to_min + dist_to_max)) * sdf_weight
            return score
            
        except Exception as e:
            rospy.logerr(f"Error in clutter score calculation: {str(e)}")
            return 0.0
    
    def _calculate_distance_score(self, depth_map, leaf_mask):
        """Enhanced distance score from old pipeline"""
        try:
            # Get coordinates and depths of leaf points
            y_indices, x_indices = np.where(leaf_mask)
            if len(y_indices) == 0:
                return float('inf')
                
            # Get depths for all leaf points
            leaf_depths = depth_map[leaf_mask]
            mean_depth = np.mean(leaf_depths)
            
            # Calculate 3D coordinates like in old pipeline
            X = (mean_depth * (x_indices - self.camera_cx)) / self.f_norm
            Y = (mean_depth * (y_indices - self.camera_cy)) / self.f_norm
            Z = np.full_like(X, mean_depth)
            
            # Calculate Euclidean distance from camera origin
            distances = np.sqrt(X**2 + Y**2 + Z**2)
            mean_distance = np.mean(distances)
            
            # Normalize to [0,1] range with exponential weighting
            score = np.exp(-mean_distance / 0.5)  # 0.5m scale factor
            return score
            
        except Exception as e:
            rospy.logerr(f"Error in distance score calculation: {str(e)}")
            return float('inf')
    
    def _calculate_visibility_score(self, leaf_mask):
        """Calculate visibility score based on mask completeness and border contact"""
        h, w = leaf_mask.shape
        y_indices, x_indices = np.where(leaf_mask)
        
        if len(y_indices) == 0:
            return 0.0
            
        # Calculate border contact
        border_pixels = (np.sum(leaf_mask[0,:]) + np.sum(leaf_mask[-1,:]) +
                        np.sum(leaf_mask[:,0]) + np.sum(leaf_mask[:,-1]))
        
        # Strong penalty for border contact
        if border_pixels > 0:
            return 0.0  # Completely reject leaves touching borders
        
        # If not touching borders, calculate position score
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        
        # Distance from center of image
        center_x = w / 2
        center_y = h / 2
        dist_from_center = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Higher score for leaves closer to image center
        position_score = 1.0 - (dist_from_center / max_dist)
        
        return position_score
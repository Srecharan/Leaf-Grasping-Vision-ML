import torch
import numpy as np
import os
import rospy
import random
import cv2
from collections import deque
import shutil

class EnhancedGraspDataCollector:
    def __init__(self, patch_size=32, resume=True):
        self.patch_size = patch_size
        self.samples = []
        
        # Define the directory for saving data
        self.data_dir = os.path.expanduser('~/leaf_grasp_output/ml_training_data')
        
        # If not resuming, clear the existing data directory
        if not resume and os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
            rospy.loginfo(f"Existing data at {self.data_dir} cleared because resume is set to False.")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Track collection statistics
        self.stats = {
            'positive_samples': 0,
            'negative_samples': 0,
            'augmented_samples': 0
        }
        
        # Load existing data only if resume is True
        if resume:
            self.load_existing_data()
            rospy.loginfo(f"Resumed with {len(self.samples)} existing samples")
            self._log_collection_progress()
        else:
            rospy.loginfo("Starting fresh data collection. Existing data (if any) was cleared.")
        
        rospy.loginfo(f"Enhanced data collector initialized. Saving to: {self.data_dir}")


    def load_existing_data(self):
        """Load existing training data if available"""
        try:
            save_path = os.path.join(self.data_dir, 'training_data.pt')
            if os.path.exists(save_path):
                rospy.loginfo(f"Found existing data at {save_path}")
                data = torch.load(save_path)
                
                # Reconstruct samples from saved tensors
                num_samples = len(data['labels'])
                for i in range(num_samples):
                    sample = {
                        'depth_patch': data['depth_patches'][i],
                        'mask_patch': data['mask_patches'][i],
                        'score_patches': data['score_patches'][i],
                        'total_score': data['total_scores'][i].item(),
                        'grasp_point': tuple(data['grasp_points'][i].tolist()),
                        'label': data['labels'][i].item(),
                        'is_augmented': data['is_augmented'][i].item()
                    }
                    self.samples.append(sample)
                
                # Update statistics
                self.stats['positive_samples'] = sum(1 for s in self.samples 
                                                   if s['label'] == 1 and not s['is_augmented'])
                self.stats['augmented_samples'] = sum(1 for s in self.samples 
                                                    if s['label'] == 1 and s['is_augmented'])
                self.stats['negative_samples'] = sum(1 for s in self.samples 
                                                   if s['label'] == 0)
                
                rospy.loginfo("Loaded existing data")
                rospy.loginfo(f"Continuing from {self.stats['positive_samples']} positive samples")
            else:
                rospy.loginfo("No existing data found. Starting fresh collection.")
        except Exception as e:
            rospy.logerr(f"Error loading existing data: {str(e)}")
            rospy.logerr("Starting fresh collection.")
            self.samples = []
            self.stats = {'positive_samples': 0, 'negative_samples': 0, 'augmented_samples': 0}

    def _check_boundaries(self, x, y, shape, half_size):
        """Check if point is within valid boundaries"""
        if (y < half_size or y >= shape[0] - half_size or 
            x < half_size or x >= shape[1] - half_size):
            rospy.logwarn(f"Point ({x},{y}) too close to edge. Image shape: {shape}")
            return False
        return True

    def _extract_patches(self, x, y, leaf_mask, depth_tensor, scores):
        """Extract and validate all patches with improved error checking"""
        try:
            # Input validation
            if not torch.is_tensor(depth_tensor) or not torch.is_tensor(leaf_mask):
                rospy.logwarn("Invalid input tensors")
                return None
                
            half_size = self.patch_size // 2
            
            # Extract basic patches with bounds checking
            try:
                depth_patch = depth_tensor[y-half_size:y+half_size, x-half_size:x+half_size].clone()
                mask_patch = leaf_mask[y-half_size:y+half_size, x-half_size:x+half_size].clone()
            except IndexError:
                rospy.logwarn("Patch extraction out of bounds")
                return None
            
            # Validate patch sizes
            if depth_patch.numel() == 0 or mask_patch.numel() == 0:
                rospy.logwarn("Empty patches extracted")
                return None
                
            if (depth_patch.shape != (self.patch_size, self.patch_size) or 
                mask_patch.shape != (self.patch_size, self.patch_size)):
                rospy.logwarn(f"Invalid patch size: {depth_patch.shape}, {mask_patch.shape}")
                return None
                
            # Validate depth values
            if torch.isnan(depth_patch).any() or torch.isinf(depth_patch).any():
                rospy.logwarn("Invalid depth values in patch")
                return None
                
            # Validate mask coverage
            if not mask_patch.any():
                rospy.logwarn("Empty mask patch")
                return None

            # Extract and validate score patches
            score_patches = []
            required_scores = ['sdf_score', 'approach_score', 'flatness_map',
                            'isolation_map', 'distance_map', 'accessibility_map',
                            'stem_penalty']
                            
            for score_name in required_scores:
                if score_name not in scores:
                    rospy.logwarn(f"Missing required score: {score_name}")
                    return None
                    
                score_map = scores[score_name]
                
                # Handle different types
                if isinstance(score_map, np.ndarray):
                    patch = score_map[y-half_size:y+half_size, x-half_size:x+half_size]
                    # Validate numpy array values
                    if np.isnan(patch).any() or np.isinf(patch).any():
                        rospy.logwarn(f"Invalid values in {score_name}")
                        return None
                    score_patches.append(torch.from_numpy(patch).float())
                    
                elif isinstance(score_map, torch.Tensor):
                    patch = score_map[y-half_size:y+half_size, x-half_size:x+half_size]
                    # Validate tensor values
                    if torch.isnan(patch).any() or torch.isinf(patch).any():
                        rospy.logwarn(f"Invalid values in {score_name}")
                        return None
                    score_patches.append(patch.float())
                else:
                    rospy.logwarn(f"Invalid score map type for {score_name}")
                    return None
            
            # Verify we got all required scores
            if len(score_patches) != len(required_scores):
                rospy.logwarn(f"Expected {len(required_scores)} score channels, got {len(score_patches)}")
                return None
                
            return depth_patch, mask_patch, torch.stack(score_patches)
            
        except Exception as e:
            rospy.logerr(f"Error extracting patches: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

    def collect_sample(self, leaf_mask, depth_tensor, rgb_image, scores, grasp_point_2d, total_score):
        """Collect both positive and negative samples during runtime"""
        try:
            # Validate input tensors
            if not torch.is_tensor(depth_tensor) or not torch.is_tensor(leaf_mask):
                rospy.logerr("Invalid input tensors")
                return False

            # Validate grasp point coordinates
            x, y = map(int, grasp_point_2d)
            if x < 0 or y < 0 or x >= leaf_mask.shape[1] or y >= leaf_mask.shape[0]:
                rospy.logerr(f"Invalid grasp point coordinates: ({x}, {y})")
                return False

            # Validate scores dictionary
            required_scores = ['sdf_score', 'approach_score', 'flatness_map', 
                            'isolation_map', 'distance_map', 'accessibility_map', 
                            'stem_penalty']
            if not all(score in scores for score in required_scores):
                rospy.logerr("Missing required score maps")
                return False

            half_size = self.patch_size // 2
            
            # Log collection details
            rospy.loginfo(f"Collecting positive sample at point ({x}, {y}) with score {total_score:.3f}")
            
            # Check boundaries
            if not self._check_boundaries(x, y, leaf_mask.shape, half_size):
                return False
                
            # Extract patches with validation
            patches = self._extract_patches(x, y, leaf_mask, depth_tensor, scores)
            if not patches:
                return False
                
            depth_patch, mask_patch, score_patches = patches
            
            # Validate patch data ranges
            if torch.isnan(depth_patch).any() or torch.isinf(depth_patch).any():
                rospy.logerr("Invalid values in depth patch")
                return False
                
            if not mask_patch.any():
                rospy.logerr("Empty mask patch")
                return False
                
            # Add positive sample
            success = self._add_sample(depth_patch, mask_patch, score_patches, total_score,
                                    grasp_point_2d, label=1, is_augmented=False)
            if not success:
                return False
                
            # Generate augmented positive samples
            self._generate_augmented_samples(depth_patch, mask_patch, score_patches,
                                        total_score, grasp_point_2d)
            
            # Collect negative samples with enhanced validation
            self._collect_validated_negative_samples(leaf_mask, depth_tensor, scores)
            
            # Log collection progress
            self._log_collection_progress()
            
            # Save periodically
            if (self.stats['positive_samples'] + self.stats['negative_samples']) % 5 == 0:
                self.save_samples()
                
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in sample collection: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False

    def _generate_augmented_samples(self, depth_patch, mask_patch, score_patches, total_score, grasp_point_2d):
        """Generate augmented samples with improved tensor handling"""
        try:
            # Convert tensors to float and handle bool masks
            depth_patch = depth_patch.float()
            mask_patch = mask_patch.float()
            score_patches = [s.float() for s in score_patches]
            
            # List of augmentation angles
            angles = [90, 180, 270]  # 90-degree rotations
            
            for angle in angles:
                try:
                    # Rotate patches
                    rot_depth = self._rotate_tensor(depth_patch, angle)
                    rot_mask = self._rotate_tensor(mask_patch, angle)
                    rot_scores = torch.stack([self._rotate_tensor(score, angle) 
                                        for score in score_patches])
                    
                    # Convert mask back to boolean after rotation if needed
                    rot_mask = (rot_mask > 0.5).float()
                    
                    # Add noise to depth (1-2% random noise)
                    noise_factor = random.uniform(0.01, 0.02)
                    depth_noise = torch.randn_like(rot_depth) * (noise_factor * rot_depth.mean())
                    noisy_depth = torch.clamp(rot_depth + depth_noise, min=0.0)
                    
                    # Calculate new grasp point after rotation
                    new_point = self._rotate_point(grasp_point_2d, angle, self.patch_size)
                    
                    # Add augmented sample
                    success = self._add_sample(noisy_depth, rot_mask, rot_scores, 
                                            total_score * random.uniform(0.95, 1.0),
                                            new_point, label=1, is_augmented=True)
                    
                    if not success:
                        rospy.logwarn(f"Failed to add augmented sample for angle {angle}")
                        
                except Exception as e:
                    rospy.logwarn(f"Error in augmentation for angle {angle}: {str(e)}")
                    continue
                    
        except Exception as e:
            rospy.logwarn(f"Error in augmentation: {str(e)}")

    def _collect_validated_negative_samples(self, leaf_mask, depth_tensor, scores):
        """Collect negative samples with improved validation and limits"""
        try:
            leaf_mask_np = leaf_mask.cpu().numpy()
            
            # Control parameters
            max_attempts = 10
            attempts = 0
            collected = 0
            max_negative_samples = 3  # Limit per positive sample
            
            while collected < max_negative_samples and attempts < max_attempts:
                try:
                    # Generate candidate points
                    negative_points = []
                    
                    # Get points from different regions
                    tip_points = self._get_tip_points(leaf_mask_np)
                    stem_points = self._get_stem_points(leaf_mask_np)
                    edge_points = self._get_edge_points(leaf_mask_np)
                    
                    # Add points with validation
                    if tip_points:
                        negative_points.extend(random.sample(tip_points, 
                                            min(1, len(tip_points))))
                    if stem_points:
                        negative_points.extend(random.sample(stem_points, 
                                            min(1, len(stem_points))))
                    if edge_points:
                        negative_points.extend(random.sample(edge_points, 
                                            min(1, len(edge_points))))
                    
                    # Process negative points
                    for x, y in negative_points:
                        if collected >= max_negative_samples:
                            break
                            
                        patches = self._extract_patches(x, y, leaf_mask, depth_tensor, scores)
                        if patches:
                            depth_patch, mask_patch, score_patches = patches
                            success = self._add_sample(depth_patch, mask_patch, score_patches, 
                                                0.0, (x, y), label=0, is_augmented=False)
                            if success:
                                collected += 1
                    
                    attempts += 1
                    
                except Exception as e:
                    rospy.logwarn(f"Error processing negative samples: {str(e)}")
                    attempts += 1
                    continue
                    
        except Exception as e:
            rospy.logerr(f"Error in negative sample collection: {str(e)}")

    def _add_sample(self, depth_patch, mask_patch, score_patches, total_score, grasp_point, label, is_augmented):
        """Add a sample with improved validation"""
        try:
            # Validate inputs
            if not all(isinstance(x, torch.Tensor) for x in [depth_patch, mask_patch]):
                rospy.logwarn("Invalid tensor types in sample")
                return False
                
            if not all(x.shape == (self.patch_size, self.patch_size) 
                    for x in [depth_patch, mask_patch]):
                rospy.logwarn("Invalid patch shapes")
                return False
                
            # Convert any boolean tensors to float
            if mask_patch.dtype == torch.bool:
                mask_patch = mask_patch.float()
                
            # Create sample with type checking
            sample = {
                'depth_patch': depth_patch,
                'mask_patch': mask_patch,
                'score_patches': score_patches,
                'total_score': float(total_score),  # Ensure scalar
                'grasp_point': tuple(map(int, grasp_point)),  # Ensure integers
                'label': int(label),
                'is_augmented': bool(is_augmented)
            }
            
            # Add to samples list
            self.samples.append(sample)
            
            # Update statistics
            if label == 1:
                if is_augmented:
                    self.stats['augmented_samples'] += 1
                else:
                    self.stats['positive_samples'] += 1
            else:
                self.stats['negative_samples'] += 1
                
            return True
            
        except Exception as e:
            rospy.logwarn(f"Error adding sample: {str(e)}")
            return False

    def _rotate_tensor(self, tensor, angle):
        """Rotate a tensor by the specified angle"""
        k = angle // 90
        return torch.rot90(tensor, k, dims=(-2, -1))
        
    def _rotate_point(self, point, angle, size):
        """Rotate a point around the patch center"""
        x, y = point
        center = size // 2
        angle_rad = np.radians(angle)
        
        # Translate to origin
        x -= center
        y -= center
        
        # Rotate
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate back
        new_x += center
        new_y += center
        
        return (int(new_x), int(new_y))

    def _get_tip_points(self, mask):
        """Get points in leaf tip regions"""
        try:
            # Calculate distance transform
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            
            # Find local maxima in distance transform
            kernel = np.ones((5,5), np.uint8)
            local_max = cv2.dilate(dist, kernel) == dist
            
            # Get tip candidates
            y_indices, x_indices = np.where((local_max) & (mask > 0))
            points = list(zip(x_indices, y_indices))
            
            # Sort by distance value and take top 25%
            points.sort(key=lambda p: dist[p[1], p[0]], reverse=True)
            return points[:max(1, len(points)//4)]
            
        except Exception as e:
            rospy.logwarn(f"Error getting tip points: {str(e)}")
            return []

    def _get_stem_points(self, mask):
        """Get points in stem regions using morphology"""
        try:
            height = mask.shape[0]
            stem_region = mask.copy()
            stem_region[:int(0.75*height)] = 0  # Keep bottom 25%
            
            # Erode to get stem-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            eroded = cv2.erode(stem_region.astype(np.uint8), kernel, iterations=2)
            
            y_indices, x_indices = np.where(eroded > 0)
            return list(zip(x_indices, y_indices))
            
        except Exception as e:
            rospy.logwarn(f"Error getting stem points: {str(e)}")
            return []

    def _get_edge_points(self, mask):
        """Get points along high-curvature edges"""
        try:
            # Get contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_NONE)
            
            edge_points = []
            if contours:
                contour = max(contours, key=cv2.contourArea)
                # Calculate curvature using contour angles
                for i in range(len(contour)):
                    prev = contour[i-1][0]
                    curr = contour[i][0]
                    next_pt = contour[(i+1)%len(contour)][0]
                    
                    # Calculate angles
                    v1 = prev - curr
                    v2 = next_pt - curr
                    angle = np.abs(np.arctan2(np.cross(v1,v2), np.dot(v1,v2)))
                    
                    if angle < np.pi/4:  # High curvature points
                        edge_points.append((curr[0], curr[1]))
                        
            return edge_points
            
        except Exception as e:
            rospy.logwarn(f"Error getting edge points: {str(e)}")
            return []

    def _log_collection_progress(self):
        """Log current collection statistics"""
        rospy.loginfo("\n=== Data Collection Progress ===")
        rospy.loginfo(f"Original positive samples: {self.stats['positive_samples']}")
        rospy.loginfo(f"Augmented positive samples: {self.stats['augmented_samples']}")
        rospy.loginfo(f"Negative samples: {self.stats['negative_samples']}")
        rospy.loginfo(f"Total samples: {len(self.samples)}")
        
    def save_samples(self):
        """Save samples with backup mechanism and enhanced quality metrics"""
        try:
            if not self.samples:
                rospy.logwarn("No samples to save")
                return
                
            # Setup paths
            save_path = os.path.join(self.data_dir, 'training_data.pt')
            backup_path = save_path + '.backup'
            
            # Create backup of previous save
            if os.path.exists(save_path):
                shutil.copy2(save_path, backup_path)
                
            try:
                # Prepare data tensors
                data = {
                    'depth_patches': torch.stack([s['depth_patch'] for s in self.samples]),
                    'mask_patches': torch.stack([s['mask_patch'] for s in self.samples]),
                    'score_patches': torch.stack([s['score_patches'] for s in self.samples]),
                    'labels': torch.tensor([s['label'] for s in self.samples]),
                    'total_scores': torch.tensor([s['total_score'] for s in self.samples]),
                    'grasp_points': torch.tensor([s['grasp_point'] for s in self.samples]),
                    'is_augmented': torch.tensor([s['is_augmented'] for s in self.samples])
                }
                
                # Calculate quality metrics
                quality_metrics = {
                    'depth_range': [data['depth_patches'].min().item(), 
                                data['depth_patches'].max().item()],
                    'mask_coverage': (data['mask_patches'] > 0).float().mean().item(),
                    'positive_ratio': (data['labels'] == 1).float().mean().item(),
                    'augmented_ratio': data['is_augmented'].float().mean().item(),
                    'score_statistics': {
                        'mean': data['total_scores'].mean().item(),
                        'std': data['total_scores'].std().item(),
                        'min': data['total_scores'].min().item(),
                        'max': data['total_scores'].max().item()
                    }
                }
                
                # Log quality metrics
                rospy.loginfo("\n=== Data Quality Metrics ===")
                rospy.loginfo(f"Depth range: {quality_metrics['depth_range']}")
                rospy.loginfo(f"Mask coverage: {quality_metrics['mask_coverage']:.3f}")
                rospy.loginfo(f"Positive sample ratio: {quality_metrics['positive_ratio']:.3f}")
                rospy.loginfo(f"Augmented sample ratio: {quality_metrics['augmented_ratio']:.3f}")
                rospy.loginfo("\nScore Statistics:")
                for key, value in quality_metrics['score_statistics'].items():
                    rospy.loginfo(f"{key}: {value:.3f}")
                
                # Save data
                torch.save(data, save_path)
                
                # Save metadata
                metadata_path = os.path.join(self.data_dir, 'collection_metadata.txt')
                with open(metadata_path, 'w') as f:
                    f.write("=== Data Collection Statistics ===\n")
                    f.write(f"Original positive samples: {self.stats['positive_samples']}\n")
                    f.write(f"Augmented positive samples: {self.stats['augmented_samples']}\n")
                    f.write(f"Negative samples: {self.stats['negative_samples']}\n")
                    f.write(f"Total samples: {len(self.samples)}\n\n")
                    
                    f.write("=== Tensor Shapes ===\n")
                    for key, tensor in data.items():
                        f.write(f"{key}: {tensor.shape}\n")
                        
                    f.write("\n=== Quality Metrics ===\n")
                    f.write(f"Depth range: {quality_metrics['depth_range']}\n")
                    f.write(f"Mask coverage: {quality_metrics['mask_coverage']:.3f}\n")
                    f.write(f"Positive ratio: {quality_metrics['positive_ratio']:.3f}\n")
                    f.write(f"Augmented ratio: {quality_metrics['augmented_ratio']:.3f}\n")
                    f.write("\nScore Statistics:\n")
                    for key, value in quality_metrics['score_statistics'].items():
                        f.write(f"{key}: {value:.3f}\n")
                
                rospy.loginfo(f"Saved {len(self.samples)} samples to {save_path}")
                rospy.loginfo(f"Saved metadata to {metadata_path}")
                
                # Remove backup if save was successful
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
            except Exception as e:
                rospy.logerr(f"Error during data preparation and saving: {str(e)}")
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, save_path)
                    rospy.loginfo("Restored from backup")
                raise

            frame_count_path = os.path.join(self.data_dir, 'collection_progress.txt')
            with open(frame_count_path, 'w') as f:
                f.write(f"last_frame: {self.stats['positive_samples']}\n")
                
        except Exception as e:
            rospy.logerr(f"Error saving samples: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())

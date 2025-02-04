#!/usr/bin/env python3

# Optimized Leaf_Grasp_Node with better GPU memory management

import sys
import os
import rospy
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from skimage import measure
import torch.nn.functional as F
import colorsys
from scipy.spatial import KDTree
from raftstereo.msg import depth
from yoloV8_seg.msg import masks
from std_msgs.msg import String

class LeafGraspNode:
    def __init__(self):
        rospy.init_node('leaf_grasp_node', anonymous=False)
        
        # CUDA setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_cuda()
        
        # Image dimensions
        self.height = 1080
        self.width = 1440
        
        # Selection criteria weights
        self.initialize_weights()
        
        # Parameters
        self.min_leaf_area = 3500  # Minimum area in pixels
        self.kernel_size = 21  # For isolation calculation
        self.depth_threshold = 0.7  # Maximum acceptable depth
        self.gaussian_kernel_size = 5  # For depth smoothing
        
        # Create output directory
        self.output_dir = os.path.join(os.path.expanduser('~'), 'leaf_grasp_output')
        self.vis_path = os.path.join(self.output_dir, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        
        # Initialize kernels and state
        self.initialize_kernels()
        self.initialize_state()
        
        # ROS setup
        self.setup_ros()
        
        rospy.loginfo("Optimized Leaf grasp node initialized")

    def setup_cuda(self):
        """Enhanced CUDA setup with memory management"""
        if torch.cuda.is_available():
            # Clear any existing cache
            torch.cuda.empty_cache()
            
            # Set device
            torch.cuda.set_device(0)
            
            # Limit GPU memory growth to 30%
            torch.cuda.set_per_process_memory_fraction(0.3)
            
            # Enable cudnn benchmarking for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            
            # Log GPU info
            rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
            rospy.loginfo(f"Initial GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        else:
            rospy.logwarn("CUDA not available, using CPU")

    def initialize_weights(self):
        """Initialize weights on CPU first"""
        # Create weights on CPU
        self.w_height = torch.tensor(0.35)
        self.w_isolation = torch.tensor(0.35)
        self.w_flatness = torch.tensor(0.30)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.w_height = self.w_height.to(self.device)
            self.w_isolation = self.w_isolation.to(self.device)
            self.w_flatness = self.w_flatness.to(self.device)

    def initialize_kernels(self):
        """Initialize kernels with memory optimization"""
        self.kernels = {}
        
        # Create kernels on CPU first
        isolation_kernel = torch.ones(self.kernel_size, self.kernel_size)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        gaussian_kernel = self.create_gaussian_kernel(self.gaussian_kernel_size)
        
        # Store in dictionary
        self.kernels = {
            'isolation': isolation_kernel,
            'sobel_x': sobel_x,
            'sobel_y': sobel_x.t(),
            'gaussian': gaussian_kernel
        }

    def initialize_state(self):
        """Initialize state variables"""
        self.latest_mask = None
        self.latest_depth = None
        self.processing = False
        self.depth_updated = False
        self.mask_updated = False
        self.last_processed_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.1)  # 10Hz processing rate
        self.frame_count = 0
        self.color_map = {}

    def setup_ros(self):
        """Setup ROS subscribers and publishers"""
        self.mask_sub = rospy.Subscriber('/leaves_masks', masks, self.mask_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/depth_image', depth, self.depth_callback, queue_size=1)
        self.grasp_pub = rospy.Publisher('/optimal_leaf_grasp', String, queue_size=1)
        self.vis_pub = rospy.Publisher('/visualization_topic', String, queue_size=1)

    def get_kernel(self, name):
        """Lazy loading of kernels to GPU"""
        if name in self.kernels:
            if self.kernels[name].device != self.device:
                self.kernels[name] = self.kernels[name].to(self.device)
        return self.kernels[name]

    def clear_gpu_memory(self):
        """Enhanced GPU memory cleanup"""
        if torch.cuda.is_available():
            # Move kernels back to CPU
            for name in self.kernels:
                if hasattr(self.kernels[name], 'device') and self.kernels[name].device.type == 'cuda':
                    self.kernels[name] = self.kernels[name].cpu()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory status
            rospy.logdebug(f"GPU Memory after cleanup: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

    def create_gaussian_kernel(self, size):
        """Create 2D Gaussian kernel for depth smoothing"""
        sigma = size / 6.0
        center = size // 2
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return torch.tensor(kernel, dtype=torch.float32)

    def process_with_gpu_management(self, func):
        """Decorator for GPU memory management"""
        def wrapper(*args, **kwargs):
            try:
                # Clear GPU before processing
                self.clear_gpu_memory()
                
                # Execute function
                result = func(*args, **kwargs)
                
                return result
            finally:
                # Clear GPU after processing
                self.clear_gpu_memory()
        return wrapper

    @process_with_gpu_management
    def smooth_depth(self, depth_patch):
        """Apply Gaussian smoothing to depth data"""
        # Get Gaussian kernel and ensure it's on GPU
        gaussian_kernel = self.get_kernel('gaussian')
        
        # Move depth patch to GPU if needed
        if depth_patch.device != self.device:
            depth_patch = depth_patch.to(self.device)
            
        # Apply padding and convolution
        padded_depth = F.pad(depth_patch.unsqueeze(0).unsqueeze(0), 
                           (self.gaussian_kernel_size//2,)*4, mode='reflect')
        smoothed = F.conv2d(padded_depth, 
                          gaussian_kernel.view(1, 1, *gaussian_kernel.shape), 
                          padding=0).squeeze()
                          
        return smoothed

    @process_with_gpu_management
    def calculate_flatness(self, depth_patch, leaf_mask):
        """Calculate flatness score using surface normals approximation"""
        try:
            # Smooth depth data
            depth_patch = self.smooth_depth(depth_patch)
            
            # Get Sobel kernels
            sobel_x = self.get_kernel('sobel_x')
            sobel_y = self.get_kernel('sobel_y')
            
            # Move tensors to GPU if needed
            if depth_patch.device != self.device:
                depth_patch = depth_patch.to(self.device)
            
            # Apply Sobel operators for gradients
            padded_depth = F.pad(depth_patch.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='reflect')
            dx = F.conv2d(padded_depth, sobel_x.unsqueeze(0).unsqueeze(0))
            dy = F.conv2d(padded_depth, sobel_y.unsqueeze(0).unsqueeze(0))
            
            # Calculate normal vectors
            normal_z = torch.ones_like(dx)
            magnitude = torch.sqrt(dx**2 + dy**2 + normal_z**2)
            
            # Normalize vectors
            dx = dx / (magnitude + 1e-6)
            dy = dy / (magnitude + 1e-6)
            normal_z = normal_z / (magnitude + 1e-6)
            
            # Calculate variance of normal vectors
            normals = torch.cat([dx, dy, normal_z], dim=1)
            normals_var = torch.var(normals.reshape(3, -1), dim=1).sum()
            
            # Calculate planarity score (inverse of variance)
            planarity = 1.0 / (normals_var + 1e-6)
            
            return planarity
            
        finally:
            # Clean up intermediate tensors
            del dx, dy, normal_z, magnitude, normals
            torch.cuda.empty_cache()

    @process_with_gpu_management
    def calculate_isolation(self, leaf_mask, all_masks):
        """Calculate isolation score using distance transform and convolution"""
        try:
            # Get isolation kernel
            isolation_kernel = self.get_kernel('isolation')
            
            # Move masks to GPU if needed
            if all_masks.device != self.device:
                all_masks = all_masks.to(self.device)
            if leaf_mask.device != self.device:
                leaf_mask = leaf_mask.to(self.device)
            
            # Create binary mask of other leaves
            other_leaves = (all_masks > 0) & (all_masks != leaf_mask)
            
            # Calculate distance score using convolution
            isolation_score = F.conv2d(
                other_leaves.float().unsqueeze(0).unsqueeze(0),
                isolation_kernel.unsqueeze(0).unsqueeze(0),
                padding=self.kernel_size//2
            )
            
            # Calculate mean isolation within leaf area
            leaf_area = leaf_mask.float().unsqueeze(0).unsqueeze(0)
            mean_isolation = (isolation_score * leaf_area).sum() / (leaf_area.sum() + 1e-6)
            
            # Convert to isolation score (higher is better)
            isolation = 1.0 / (mean_isolation + 1e-6)
            
            return isolation
            
        finally:
            # Clean up intermediate tensors
            del other_leaves, isolation_score, leaf_area
            torch.cuda.empty_cache()

    @process_with_gpu_management
    def calculate_leaf_scores(self, depth_tensor, mask_tensor, leaf_mask, leaf_id):
        """Calculate all scores for a single leaf"""
        try:
            # Calculate depth score
            leaf_depth = depth_tensor[leaf_mask]
            mean_depth = leaf_depth.mean()
            height_score = 1.0 / (mean_depth + 1e-6)
            
            # Calculate isolation score
            isolation_score = self.calculate_isolation(leaf_mask, mask_tensor)
            
            # Calculate flatness score
            flatness_score = self.calculate_flatness(depth_tensor * leaf_mask.float(), leaf_mask)
            
            # Normalize scores
            height_score = height_score / (1.0 / (self.depth_threshold + 1e-6))
            isolation_score = torch.clamp(isolation_score / 1000.0, 0, 1)
            flatness_score = torch.clamp(flatness_score / 1000.0, 0, 1)
            
            # Calculate weighted total score
            total_score = (self.w_height * height_score + 
                         self.w_isolation * isolation_score + 
                         self.w_flatness * flatness_score)
            
            return {
                'height': height_score.item(),
                'isolation': isolation_score.item(),
                'flatness': flatness_score.item(),
                'total': total_score.item()
            }
            
        finally:
            # Clean up intermediate tensors
            del leaf_depth
            torch.cuda.empty_cache()

    def calculate_centroid(self, leaf_mask):
        """Calculate centroid of leaf mask"""
        y_indices, x_indices = torch.where(leaf_mask)
        centroid_x = float(x_indices.float().mean())
        centroid_y = float(y_indices.float().mean())
        return (centroid_x, centroid_y)

    def generate_color(self, leaf_id):
        """Generate distinct colors for leaf visualization"""
        if leaf_id not in self.color_map:
            golden_ratio = 0.618033988749895
            hue = (leaf_id * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self.color_map[leaf_id] = tuple(int(255 * x) for x in rgb)
        return self.color_map[leaf_id]

    @process_with_gpu_management
    def select_optimal_leaf(self):
        """Memory-optimized leaf selection"""
        if self.latest_mask is None or self.latest_depth is None or self.processing:
            return
        
        self.processing = True
        try:
            # Move data to GPU in batches if needed
            depth_tensor = self.latest_depth.to(self.device)
            mask_tensor = self.latest_mask.to(self.device)
            
            # Process in smaller batches if needed
            leaf_ids = torch.unique(mask_tensor)[1:]
            best_score = -float('inf')
            optimal_leaf_id = None
            optimal_centroid = None
            optimal_scores = None
            
            # Process each leaf with memory management
            for leaf_id in leaf_ids:
                # Clear intermediate results
                torch.cuda.empty_cache()
                
                # Create mask for current leaf
                leaf_mask = (mask_tensor == leaf_id)
                
                if leaf_mask.sum() < self.min_leaf_area:
                    continue
                
                # Get leaf depth values
                leaf_depth = depth_tensor[leaf_mask]
                mean_depth = leaf_depth.mean()
                
                # Skip if leaf is too far
                if mean_depth > self.depth_threshold:
                    continue
                
                # Calculate scores with memory optimization
                scores = self.calculate_leaf_scores(depth_tensor, mask_tensor, leaf_mask, leaf_id)
                
                if scores['total'] > best_score:
                    best_score = scores['total']
                    optimal_leaf_id = leaf_id
                    optimal_scores = scores
                    optimal_centroid = self.calculate_centroid(leaf_mask)
            
            # Process results
            if optimal_leaf_id is not None:
                self.publish_results(optimal_leaf_id, optimal_centroid, optimal_scores)
                self.visualize_result(optimal_leaf_id, optimal_centroid)
        
        finally:
            self.processing = False
            self.mask_updated = False
            self.depth_updated = False
            self.clear_gpu_memory()

    @process_with_gpu_management
    def visualize_result(self, leaf_id, centroid):
        """Generate and save visualization of leaf selection"""
        try:
            # Move tensors to CPU for visualization
            all_masks = self.latest_mask.cpu().numpy()
            depth = self.latest_depth.cpu().numpy()
            optimal_mask = (self.latest_mask == leaf_id).cpu().numpy().astype(np.uint8)
            
            # Create visualization for all leaves
            vis_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw all leaves with dynamic colors
            unique_ids = np.unique(all_masks)[1:]
            for id in unique_ids:
                mask = (all_masks == id)
                vis_img[mask] = self.generate_color(int(id))
            
            # Process optimal leaf contour
            optimal_contour = (optimal_mask * 255).astype(np.uint8)
            optimal_contour = cv2.GaussianBlur(optimal_contour, (5,5), 0)
            contours, _ = cv2.findContours(optimal_contour, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_TC89_KCOS)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Smooth contour
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(smoothed_contour) >= 5:
                    # Fit and draw ellipse
                    ellipse = cv2.fitEllipse(smoothed_contour)
                    
                    # Create semi-transparent overlay
                    overlay = vis_img.copy()
                    cv2.ellipse(overlay, ellipse, (0, 255, 255), -1)
                    vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)
                    
                    # Draw ellipse boundary
                    cv2.ellipse(vis_img, ellipse, (0, 255, 255), 2)
            
            # Create depth visualization
            depth_vis = np.zeros_like(vis_img)
            masked_depth = depth * optimal_mask
            if optimal_mask.sum() > 0:
                # Normalize depth values for visualization
                valid_depths = masked_depth[optimal_mask > 0]
                min_depth = valid_depths.min()
                max_depth = valid_depths.max()
                normalized_depth = np.zeros_like(masked_depth)
                normalized_depth[optimal_mask > 0] = ((valid_depths - min_depth) / 
                                                    (max_depth - min_depth + 1e-6) * 255)
                depth_vis = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
                depth_vis[optimal_mask == 0] = 0  # Set background to black
            
            # Draw centroid markers
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(vis_img, (cx, cy), 20, (255, 255, 255), 4)  # White border
            cv2.circle(vis_img, (cx, cy), 18, (0, 0, 255), -1)    # Red fill
            cv2.circle(depth_vis, (cx, cy), 20, (255, 255, 255), 4)
            cv2.circle(depth_vis, (cx, cy), 18, (0, 0, 255), -1)
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, "Optimal Leaf", (cx-60, cy-20), 
                       font, 1, (255, 255, 255), 2)
            
            # Combine visualizations side by side
            combined_vis = np.hstack((vis_img, depth_vis))
            
            # Save visualization
            vis_filename = os.path.join(self.vis_path, 
                                      f'optimal_leaf_{self.frame_count}.png')
            cv2.imwrite(vis_filename, combined_vis)
            self.frame_count += 1
            
            rospy.loginfo(f"Saved visualization to {vis_filename}")
            
        except Exception as e:
            rospy.logerr(f"Error in visualization: {str(e)}")

    def publish_results(self, leaf_id, centroid, scores):
        """Publish grasp point and scores"""
        rospy.loginfo(f"Selected Leaf {leaf_id.item()}:")
        for score_name, score_value in scores.items():
            rospy.loginfo(f"  {score_name}: {score_value:.3f}")
        
        result_msg = f"{centroid[0]},{centroid[1]}"
        self.grasp_pub.publish(result_msg)

    def mask_callback(self, msg):
        """Memory-optimized mask callback"""
        try:
            # Process on CPU first
            mask_data = np.array(msg.imageData, dtype=np.int16)
            self.latest_mask = torch.tensor(mask_data).reshape(self.height, self.width)
            self.mask_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in mask callback: {str(e)}")
            self.clear_gpu_memory()

    def depth_callback(self, msg):
        """Memory-optimized depth callback"""
        try:
            # Process on CPU first
            depth_data = np.array(msg.imageData, dtype=np.float32)
            self.latest_depth = torch.tensor(depth_data).reshape(self.height, self.width)
            self.depth_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {str(e)}")
            self.clear_gpu_memory()

    def check_and_process(self):
        """Synchronize processing of mask and depth data"""
        current_time = rospy.Time.now()
        if (self.mask_updated and self.depth_updated and 
            (current_time - self.last_processed_time) > self.process_interval):
            self.select_optimal_leaf()
            self.last_processed_time = current_time

if __name__ == '__main__':
    try:
        node = LeafGraspNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

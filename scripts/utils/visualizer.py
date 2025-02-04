#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy
import matplotlib.pyplot as plt
from .gpu_manager import GPUManager

class LeafVisualizer:
    def __init__(self, height, width, vis_path, grasp_selector=None):
        self.height = height
        self.width = width
        self.vis_path = vis_path
        self.grasp_selector = grasp_selector
        os.makedirs(self.vis_path, exist_ok=True)
        self.frame_count = 0
        self.camera_origin_color = (255, 255, 0)  # Yellow
        self.axis_length = 50  # pixels
        self.raft_output_dir = os.path.join(os.path.expanduser('~'), 'leaf_grasp_output/Raft_outputs')

    def draw_camera_frame(self, img, camera_params):
        """Draw camera coordinate frame"""
        cx = int(camera_params['cx'])
        cy = int(camera_params['cy'])
        
        # Draw camera center
        cv2.circle(img, (cx, cy), 5, self.camera_origin_color, -1)
        
        # Draw coordinate axes
        cv2.arrowedLine(img, (cx, cy), (cx + self.axis_length, cy), 
                       (0, 0, 255), 2)  # X-axis (Red)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - self.axis_length), 
                       (0, 255, 0), 2)  # Y-axis (Green)
        cv2.arrowedLine(img, (cx, cy), (cx + 20, cy + 20), 
                       (255, 0, 0), 2)  # Z-axis (Blue, projected)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Camera Origin', (cx - 60, cy - 20), 
                   font, 0.5, self.camera_origin_color, 2)
        cv2.putText(img, f'({cx},{cy})', (cx - 40, cy + 30), 
                   font, 0.5, self.camera_origin_color, 2)
        
        return img

    def visualize_result(self, leaf_id, grasp_point_2d, grasp_point_3d, pre_grasp_point,
                    mask_tensor, depth_tensor, color_generator, camera_params, 
                    left_image=None, tall_leaves=None):  # Added tall_leaves parameter
        try:
            # Generate filenames
            filename = f"optimal_leaf_{self.frame_count}.png"
            raft_image_path = os.path.join(self.raft_output_dir, f"left_rect{self.frame_count}.png")
            
            rospy.loginfo(f"Processing frame {self.frame_count}")
            rospy.loginfo(f"Looking for RAFT image: {raft_image_path}")
            
            # Create left side visualization
            all_masks = mask_tensor.cpu().numpy()
            optimal_mask = (mask_tensor == leaf_id).cpu().numpy().astype(np.uint8)
            vis_img = self._create_leaf_visualization(
                all_masks, 
                color_generator,
                tall_leaves=tall_leaves,  # Pass tall_leaves to visualization
                optimal_leaf_id=leaf_id
            )
            vis_img = self._add_contour(vis_img, optimal_mask)
            
            # Add legend
            legend_y = 120  # Start y position for legend
            cv2.putText(vis_img, "Tall Leaves", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis_img, "Regular Leaves", (10, legend_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, "Selected Leaf", (10, legend_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add orientation visualization
            angle, major_axis, minor_axis, center = self.grasp_selector.estimate_leaf_orientation(optimal_mask)
            if angle is not None:
                dx = int(major_axis/2 * np.cos(angle))
                dy = int(major_axis/2 * np.sin(angle))
                center = (int(center[0]), int(center[1]))
                end_point = (center[0] + dx, center[1] + dy)
                start_point = (center[0] - dx, center[1] - dy)
                
                cv2.line(vis_img, start_point, end_point, (255, 255, 255), 2)
                cv2.arrowedLine(vis_img, center, end_point, (255, 255, 255), 2)
                
                angle_deg = np.rad2deg(angle)
                cv2.putText(vis_img, f'Leaf Angle: {angle_deg:.1f}°', 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            
            # Add grasp points
            gx, gy = int(grasp_point_2d[0]), int(grasp_point_2d[1])
            
            if pre_grasp_point is not None:
                pre_grasp_2d = self._project_3d_to_2d(pre_grasp_point, camera_params)
                pgx, pgy = int(pre_grasp_2d[0]), int(pre_grasp_2d[1])
                
                cv2.line(vis_img, (pgx, pgy), (gx, gy), (0, 255, 255), 2)
                cv2.circle(vis_img, (pgx, pgy), 12, (0, 255, 255), 2)
                cv2.circle(vis_img, (pgx, pgy), 8, (0, 255, 255), -1)
                cv2.line(vis_img, (pgx-7, pgy), (pgx+7, pgy), (0, 0, 0), 2)
                cv2.line(vis_img, (pgx, pgy-7), (pgx, pgy+7), (0, 0, 0), 2)
                cv2.putText(vis_img, "Pre-Grasp", (pgx-60, pgy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw grasp point
            cv2.circle(vis_img, (gx, gy), 12, (255, 255, 255), 2)
            cv2.circle(vis_img, (gx, gy), 8, (0, 255, 0), -1)
            cv2.line(vis_img, (gx-7, gy), (gx+7, gy), (0, 0, 0), 2)
            cv2.line(vis_img, (gx, gy-7), (gx, gy+7), (0, 0, 0), 2)
            cv2.putText(vis_img, "Grasp Point", (gx-60, gy-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add 3D coordinates
            x, y, z = grasp_point_3d
            cv2.putText(vis_img, f'3D: ({x:.3f}, {y:.3f}, {z:.3f}m)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if pre_grasp_point is not None:
                px, py, pz = pre_grasp_point
                cv2.putText(vis_img, f'Pre-grasp 3D: ({px:.3f}, {py:.3f}, {pz:.3f}m)',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw camera frame
            vis_img = self.draw_camera_frame(vis_img, camera_params)
            
            # Load and process RAFT image
            if os.path.exists(raft_image_path):
                right_img = cv2.imread(raft_image_path)
                if right_img is not None:
                    # Convert BGR to RGB since RAFT saves in RGB
                    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to match left image size
                    right_img = cv2.resize(right_img, (self.width, self.height))
                    
                    # Draw midrib if detected
                    if self.grasp_selector is not None:
                        midrib_points = self.grasp_selector.detect_midrib(optimal_mask, right_img)
                        if midrib_points is not None:
                            start_point, end_point = midrib_points
                            cv2.line(right_img, start_point, end_point, (0, 0, 255), 3)
                            cv2.putText(right_img, "Midrib", start_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Save combined visualization
                    combined_vis = np.hstack((vis_img, right_img))
                    save_path = os.path.join(self.vis_path, filename)
                    # Convert back to BGR for saving
                    cv2.imwrite(save_path, cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
                    rospy.loginfo(f"Saved visualization to {save_path}")
                else:
                    rospy.logwarn(f"Failed to load RAFT image {raft_image_path}")
            else:
                rospy.logwarn(f"RAFT image not found: {raft_image_path}")
                        
            self.frame_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error in visualization: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def _project_3d_to_2d(self, point_3d, camera_params):
        x, y, z = point_3d
        u = (x * camera_params['f_norm'] / z) + camera_params['cx']
        v = (y * camera_params['f_norm'] / z) + camera_params['cy']
        return (u, v)

    def _create_leaf_visualization(self, all_masks, color_generator, tall_leaves=None, optimal_leaf_id=None):
        """
        Create visualization with different colors for:
        - Tall leaves (blue)
        - Regular leaves (green)
        - Optimal leaf (red)
        """
        vis_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        unique_ids = np.unique(all_masks)[1:]  # Skip background
        
        # Define colors
        TALL_LEAF_COLOR = (255, 0, 0)     # Blue
        REGULAR_LEAF_COLOR = (0, 255, 0)   # Green
        OPTIMAL_LEAF_COLOR = (0, 0, 255)   # Red
        
        if tall_leaves is None:
            tall_leaves = []
            
        # Color each leaf
        for leaf_id in unique_ids:
            mask = (all_masks == leaf_id)
            
            if leaf_id == optimal_leaf_id:
                vis_img[mask] = OPTIMAL_LEAF_COLOR
            elif leaf_id in tall_leaves:
                vis_img[mask] = TALL_LEAF_COLOR
            else:
                vis_img[mask] = REGULAR_LEAF_COLOR
                
        return vis_img

    def _add_contour(self, vis_img, optimal_mask):
        """Add contours and ellipse to visualization"""
        # Convert to proper format for contour detection
        optimal_contour = (optimal_mask * 255).astype(np.uint8)
        optimal_contour = cv2.GaussianBlur(optimal_contour, (5,5), 0)
        
        # Draw original mask contour in white
        contours, _ = cv2.findContours(optimal_contour,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_TC89_KCOS)
        
        if contours and len(max(contours, key=cv2.contourArea)) >= 5:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw original contour in white with increased thickness
            cv2.drawContours(vis_img, [largest_contour], -1, (255, 255, 255), 3)
            
            # Create and draw eroded contour in yellow with more aggressive erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
            eroded_mask = cv2.erode(optimal_contour, kernel, iterations=2)
            eroded_contours, _ = cv2.findContours(eroded_mask,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            if eroded_contours:
                largest_eroded = max(eroded_contours, key=cv2.contourArea)
                # Draw eroded contour in yellow with increased thickness
                cv2.drawContours(vis_img, [largest_eroded], -1, (0, 255, 255), 3)
                
                # Fill the area between original and eroded contours with semi-transparent color
                mask = np.zeros_like(optimal_contour)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                cv2.drawContours(mask, [largest_eroded], -1, 0, -1)
                
                # Create yellow overlay for safety margin zone
                overlay = vis_img.copy()
                overlay[mask == 255] = [0, 255, 255]  # Yellow color
                cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                
                # Add legend with larger font
                cv2.putText(vis_img, "Safety Margin", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Fit and draw ellipse last
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                center, axes, angle = ellipse
                axes = (axes[0] * 0.9, axes[1] * 0.9)
                ellipse = (center, axes, angle)
                cv2.ellipse(vis_img, ellipse, (0, 255, 255), 2)
        
        return vis_img
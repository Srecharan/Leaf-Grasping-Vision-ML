#!/usr/bin/env python3
import cv2
import numpy as np
import os
import colorsys

class LeafAngleQuantifier:
    def __init__(self):
        self.height = 1080
        self.width = 1440
        self.points = []
        self.current_image = None
        self.current_masks = None
        self.ellipses = []
        self.selected_leaf_id = None
        self.color_map = {}
        self.display_image = None
        self.id_map = {}  # For mapping original IDs to 1-n
        
    def generate_color(self, leaf_id):
        """Generate distinct color for leaf visualization"""
        if leaf_id not in self.color_map:
            golden_ratio = 0.618033988749895
            hue = (leaf_id * golden_ratio) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self.color_map[leaf_id] = tuple(int(255 * x) for x in rgb)
        return self.color_map[leaf_id]
        
    def normalize_angle(self, angle):
        """Normalize angle to be between -90 and 90 degrees"""
        angle = angle % 180
        if angle > 90:
            angle -= 180
        return angle
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                # Draw point
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
                if len(self.points) == 2:
                    # Draw line
                    cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
                    
                    # Calculate line angle
                    dx = self.points[1][0] - self.points[0][0]
                    dy = self.points[1][1] - self.points[0][1]
                    line_angle = np.degrees(np.arctan2(-dy, dx))  # Negative dy for image coordinates
                    line_angle = self.normalize_angle(line_angle)
                    
                    # Find closest leaf
                    center_point = ((self.points[0][0] + self.points[1][0])//2, 
                                  (self.points[0][1] + self.points[1][1])//2)
                    leaf_id = self.current_masks[center_point[1], center_point[0]]
                    
                    if leaf_id > 0:
                        self.selected_leaf_id = leaf_id
                        # Find corresponding ellipse
                        for orig_id, ellipse in self.ellipses:
                            if orig_id == self.selected_leaf_id:
                                # Calculate ellipse angle
                                angle = ellipse[2]
                                if ellipse[1][0] < ellipse[1][1]:
                                    angle += 90
                                ellipse_angle = self.normalize_angle(angle)
                                
                                mapped_id = self.id_map[orig_id]
                                print(f"\nLeaf {mapped_id}:")
                                print(f"Manual line angle: {line_angle:.1f}°")
                                print(f"Ellipse angle: {-ellipse_angle:.1f}°")
                                print(f"Angle difference: {abs(line_angle + ellipse_angle):.1f}°")
                    
                    # Reset points for next line
                    self.points = []
                
                cv2.imshow('Leaf Angle Quantification', self.display_image)

    def create_mask_visualization(self, all_masks):
        """Create colored visualization of all masks"""
        vis_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        unique_ids = np.unique(all_masks)[1:]  # Skip background
        
        # Create mapping from original IDs to 1-n
        self.id_map = {id: i+1 for i, id in enumerate(sorted(unique_ids))}
        
        for orig_id in unique_ids:
            mask = (all_masks == orig_id)
            vis_img[mask] = self.generate_color(self.id_map[orig_id])
        return vis_img

    def process_image(self, image_path):
        try:
            # Read mask image
            self.current_masks = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self.current_masks is None:
                print(f"Error: Could not read image at {image_path}")
                return False
            
            # Create colored mask visualization
            self.display_image = self.create_mask_visualization(self.current_masks)
            
            # Find and draw ellipses for all leaves
            self.ellipses = []
            unique_ids = np.unique(self.current_masks)[1:]  # Skip background
            
            for orig_id in unique_ids:
                leaf_mask = (self.current_masks == orig_id).astype(np.uint8)
                contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours and len(max(contours, key=cv2.contourArea)) >= 5:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:  # Need at least 5 points for ellipse fitting
                        ellipse = cv2.fitEllipse(largest_contour)
                        self.ellipses.append((orig_id, ellipse))
                        
                        # Draw ellipse
                        cv2.ellipse(self.display_image, ellipse, (255, 255, 255), 2)
                        
                        # Draw leaf ID and ellipse angle
                        center = tuple(map(int, ellipse[0]))
                        angle = ellipse[2]
                        if ellipse[1][0] < ellipse[1][1]:
                            angle += 90
                        angle = self.normalize_angle(angle)
                        mapped_id = self.id_map[orig_id]
                        cv2.putText(self.display_image, f'{mapped_id} ({abs(angle):.0f})', 
                                    (center[0] - 40, center[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return True
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return False

def main():
    # Initialize
    quantifier = LeafAngleQuantifier()
    base_path = '/home/buggspray/leaf_grasp_output/Yolo_outputs'
    image_path = os.path.join(base_path, 'aggrigated_masks9.png')
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    # Process image
    if quantifier.process_image(image_path):
        # Set up display window
        cv2.namedWindow('Leaf Angle Quantification')
        cv2.setMouseCallback('Leaf Angle Quantification', quantifier.mouse_callback)
        
        print("\nInstructions:")
        print("1. Click two points to draw a line along the leaf midrib")
        print("2. The script will display the line angle and corresponding ellipse angle")
        print("3. Press 'q' to quit")
        print("4. Press 'n' for next image (if available)")
        
        current_image_index = 0
        
        while True:
            cv2.imshow('Leaf Angle Quantification', quantifier.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Try to load next image
                current_image_index += 1
                next_image_path = os.path.join(base_path, f'aggrigated_masks{current_image_index}.png')
                if os.path.exists(next_image_path):
                    if quantifier.process_image(next_image_path):
                        print(f"\nLoaded image {current_image_index}")
                    else:
                        print(f"Error loading image {current_image_index}")
                        current_image_index -= 1
                else:
                    print("No more images available")
                    current_image_index -= 1
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
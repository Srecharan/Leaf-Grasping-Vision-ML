#!/usr/bin/env python3
import sys
import os
import rospy
import numpy as np
import torch

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from scripts.utils.gpu_manager import GPUManager
from scripts.utils.image_processor import ImageProcessor
from scripts.utils.leaf_scorer import OptimalLeafSelector
from scripts.utils.visualizer import LeafVisualizer
from scripts.utils.grasp_point_selector import GraspPointSelector

# ROS message imports
from raftstereo.msg import depth
from yoloV8_seg.msg import masks
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

class LeafGraspNode:
    def __init__(self):
        rospy.init_node('leaf_grasp_node', anonymous=False)
        rospy.set_param('/leaf_grasp_done', False)
        self.processing_lock = False
        
        # Initialize dimensions and parameters
        self.height = 1080
        self.width = 1440
        self.min_leaf_area = 3500
        self.kernel_size = 21
        self.depth_threshold = 0.7
        self.gaussian_kernel_size = 5
        
        
        # Setup CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GPUManager.setup(self.device)

        self.grasp_selector = GraspPointSelector(self.device)
        
        # Initialize components
        self.setup_components()
        
        # Initialize state
        self.initialize_state()
        
        # Setup ROS
        self.setup_ros()
        
        rospy.loginfo("Optimized Leaf grasp node initialized")

    def setup_components(self):
        """Initialize all processing components"""
        # Create output directories
        self.output_dir = os.path.join(os.path.expanduser('~'), 'leaf_grasp_output')
        self.vis_path = os.path.join(self.output_dir, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        
        # Initialize processors
        self.image_processor = ImageProcessor(self.height, self.width, 
                                            self.kernel_size, self.gaussian_kernel_size)
        self.leaf_scorer = OptimalLeafSelector(self.device)
        self.visualizer = LeafVisualizer(self.height, self.width, 
                                        self.vis_path,
                                        self.grasp_selector)

    def initialize_state(self):
        """Initialize state variables"""
        self.latest_mask = None
        self.latest_depth = None
        self.processing = False
        self.depth_updated = False
        self.mask_updated = False
        self.last_processed_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.1)  # 10Hz processing rate

    def setup_ros(self):
        """Setup ROS subscribers and publishers"""
        self.mask_sub = rospy.Subscriber('/leaves_masks', masks, self.mask_callback, queue_size=1)
        self.left_img_sub = rospy.Subscriber('/theia/left/image_rect_color', Image, self.left_image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/depth_image', depth, 
                                        self.depth_callback, queue_size=1)
        self.grasp_pub = rospy.Publisher('/optimal_leaf_grasp', String, queue_size=1)
        self.cam_info_sub = rospy.Subscriber("/theia/right/camera_info", CameraInfo, 
                                    self.camera_info_callback, queue_size=1)

    def camera_info_callback(self, msg):
        """Handle camera calibration info"""
        try:
            projection_matrix = np.array(msg.P).reshape(3, 4)
            self.grasp_selector.set_camera_params(projection_matrix)
            self.leaf_scorer.set_camera_params(projection_matrix)  # Add this line
        except Exception as e:
            rospy.logerr(f"Error setting camera parameters: {str(e)}")    

    def select_optimal_leaf(self):
        """Memory-optimized leaf selection"""
        if self.latest_mask is None or self.latest_depth is None or self.processing:
            return
        
        self.processing = True
        rospy.set_param('/leaf_grasp_done', False)
        try:
            depth_tensor = self.latest_depth.to(self.device)
            mask_tensor = self.latest_mask.to(self.device)
            
            # Use new optimal leaf selector
            optimal_leaf_id = self.leaf_scorer.select_optimal_leaf(mask_tensor, depth_tensor)
            
            if optimal_leaf_id is not None:
                optimal_mask = (mask_tensor == optimal_leaf_id)
                grasp_point_2d, grasp_point_3d, pre_grasp_point = self.grasp_selector.select_grasp_point(
                    optimal_mask, depth_tensor, self.image_processor, pcl_data=None)
                
                camera_params = {
                    'cx': self.grasp_selector.camera_cx,
                    'cy': self.grasp_selector.camera_cy,
                    'f_norm': self.grasp_selector.f_norm
                }
                
                # Create dummy scores for compatibility
                scores = {
                    'total': 1.0,
                    'selected_leaf': optimal_leaf_id
                }
                
                self.publish_results(optimal_leaf_id, grasp_point_2d, grasp_point_3d, pre_grasp_point, scores)
                
                tall_leaves = self.leaf_scorer.get_tall_leaves()
                
                self.visualizer.visualize_result(
                    optimal_leaf_id, 
                    grasp_point_2d,
                    grasp_point_3d,
                    pre_grasp_point,
                    mask_tensor, 
                    depth_tensor,
                    self.image_processor.generate_color,
                    camera_params,
                    self.latest_left_image,
                    tall_leaves=tall_leaves 
                )
        
        finally:
            self.processing = False
            self.mask_updated = False
            self.depth_updated = False
            GPUManager.clear_memory()
            
            # Signal completion
            rospy.set_param('/leaf_grasp_done', True)
            rospy.loginfo("Leaf grasp processing complete, ready for next frame")

    def publish_results(self, leaf_id, grasp_point_2d, grasp_point_3d, pre_grasp_point, scores):
        """Publish grasp point and scores"""
        # Change from:
        # rospy.loginfo(f"Selected Leaf {leaf_id.item()}:")
        # To:
        rospy.loginfo(f"Selected Leaf {leaf_id}:")  # leaf_id is already an integer
        
        for score_name, score_value in scores.items():
            rospy.loginfo(f"  {score_name}: {score_value:.3f}")
        
        # Publish coordinates
        if pre_grasp_point is not None:
            result_msg = (f"{grasp_point_2d[0]},{grasp_point_2d[1]},"
                        f"{grasp_point_3d[0]},{grasp_point_3d[1]},{grasp_point_3d[2]},"
                        f"{pre_grasp_point[0]},{pre_grasp_point[1]},{pre_grasp_point[2]}")
        else:
            result_msg = f"{grasp_point_2d[0]},{grasp_point_2d[1]},{grasp_point_3d[0]},{grasp_point_3d[1]},{grasp_point_3d[2]}"
        
        self.grasp_pub.publish(result_msg)
        
        # Log points
        rospy.loginfo(f"Grasp Point 3D: ({grasp_point_3d[0]:.3f}m, {grasp_point_3d[1]:.3f}m, {grasp_point_3d[2]:.3f}m)")
        if pre_grasp_point is not None:
            rospy.loginfo(f"Pre-Grasp Point 3D: ({pre_grasp_point[0]:.3f}m, {pre_grasp_point[1]:.3f}m, {pre_grasp_point[2]:.3f}m)")

    def mask_callback(self, msg):
        """Memory-optimized mask callback"""
        try:
            mask_data = np.array(msg.imageData, dtype=np.int16)
            self.latest_mask = torch.tensor(mask_data).reshape(self.height, self.width)
            self.mask_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in mask callback: {str(e)}")
            GPUManager.clear_memory()

    def depth_callback(self, msg):
        """Memory-optimized depth callback"""
        try:
            depth_data = np.array(msg.imageData, dtype=np.float32)
            self.latest_depth = torch.tensor(depth_data).reshape(self.height, self.width)
            self.depth_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {str(e)}")
            GPUManager.clear_memory()

    def check_and_process(self):
        """Synchronize processing of mask and depth data"""
        current_time = rospy.Time.now()
        if (self.mask_updated and self.depth_updated and 
            (current_time - self.last_processed_time) > self.process_interval):
            
            # Wait for YOLO and RAFT to complete
            while not (rospy.get_param('/yolo_done') and rospy.get_param('/raft_done')):
                rospy.sleep(0.1)
                
            self.select_optimal_leaf()
            self.last_processed_time = current_time

    def left_image_callback(self, msg):
        """Handle left camera image"""
        try:
            self.latest_left_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(self.height, self.width, 3)
        except Exception as e:
            rospy.logerr(f"Error in left image callback: {str(e)}")        

    def check_collected_data():
        data_path = os.path.expanduser('~/leaf_grasp_output/ml_training_data/training_data.pt')
        if os.path.exists(data_path):
            data = torch.load(data_path)
            rospy.loginfo(f"Collected samples: {len(data['labels'])}")
            return True
        return False

if __name__ == '__main__':
    try:
        node = LeafGraspNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
#!/usr/bin/env python3
import sys
import os
import rospy
import numpy as np
import torch
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.gpu_manager import GPUManager
from scripts.utils.image_processor import ImageProcessor
from scripts.utils.leaf_scorer import OptimalLeafSelector
from scripts.utils.visualizer import LeafVisualizer
from scripts.utils.grasp_point_selector import GraspPointSelector
from scripts.utils.vla_integration import LLaVAProcessor, HybridSelector, ConfidenceManager

from raftstereo.msg import depth
from yoloV8_seg.msg import masks
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

class VLAEnhancedGraspNode:
    def __init__(self):
        rospy.init_node('vla_enhanced_grasp_node', anonymous=False)
        rospy.set_param('/leaf_grasp_done', False)
        self.processing_lock = False
        
        self.height = 1080
        self.width = 1440
        self.min_leaf_area = 3500
        self.kernel_size = 21
        self.depth_threshold = 0.7
        self.gaussian_kernel_size = 5
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GPUManager.setup(self.device)

        self.setup_components()
        self.initialize_state()
        self.setup_ros()
        
        rospy.loginfo("VLA Enhanced Leaf grasp node initialized")

    def setup_components(self):
        self.output_dir = os.path.join(os.path.expanduser('~'), 'leaf_grasp_output')
        self.vis_path = os.path.join(self.output_dir, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        
        self.traditional_selector = GraspPointSelector(self.device)
        self.image_processor = ImageProcessor(self.height, self.width, 
                                            self.kernel_size, self.gaussian_kernel_size)
        self.leaf_scorer = OptimalLeafSelector(self.device)
        self.visualizer = LeafVisualizer(self.height, self.width, 
                                        self.vis_path, self.traditional_selector)
        
        try:
            self.vla_processor = LLaVAProcessor(self.device)
            self.hybrid_selector = HybridSelector(self.device)
            self.confidence_manager = ConfidenceManager()
            self.vla_enabled = True
            rospy.loginfo("VLA components initialized")
        except Exception as e:
            rospy.logwarn(f"VLA initialization failed: {e}")
            self.vla_enabled = False

    def initialize_state(self):
        self.latest_mask = None
        self.latest_depth = None
        self.latest_left_image = None
        self.processing = False
        self.depth_updated = False
        self.mask_updated = False
        self.last_processed_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.1)

    def setup_ros(self):
        self.mask_sub = rospy.Subscriber('/leaves_masks', masks, self.mask_callback, queue_size=1)
        self.left_img_sub = rospy.Subscriber('/theia/left/image_rect_color', Image, 
                                           self.left_image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/depth_image', depth, 
                                        self.depth_callback, queue_size=1)
        self.grasp_pub = rospy.Publisher('/optimal_leaf_grasp', String, queue_size=1)
        self.cam_info_sub = rospy.Subscriber("/theia/right/camera_info", CameraInfo, 
                                           self.camera_info_callback, queue_size=1)

    def camera_info_callback(self, msg):
        try:
            projection_matrix = np.array(msg.P).reshape(3, 4)
            self.traditional_selector.set_camera_params(projection_matrix)
            self.leaf_scorer.set_camera_params(projection_matrix)
        except Exception as e:
            rospy.logerr(f"Error setting camera parameters: {str(e)}")

    def select_optimal_leaf_vla(self, instruction="Select the best leaf for grasping"):
        if self.latest_mask is None or self.latest_depth is None or self.processing:
            return
        
        self.processing = True
        rospy.set_param('/leaf_grasp_done', False)
        
        try:
            depth_tensor = self.latest_depth.to(self.device)
            mask_tensor = self.latest_mask.to(self.device)
            
            candidates = self._generate_candidates(mask_tensor, depth_tensor)
            
            if not candidates:
                rospy.logwarn("No valid candidates found")
                return
                
            geometric_scores = [c['geometric_score'] for c in candidates]
            
            if self.vla_enabled and self.latest_left_image is not None:
                try:
                    vla_scores = self.vla_processor.evaluate_candidates(
                        self.latest_left_image, candidates, instruction
                    )
                    
                    vla_confidence = self.confidence_manager.calculate_confidence(
                        vla_scores, geometric_scores
                    )
                    
                    best_candidate = self.hybrid_selector.select_best_candidate(
                        candidates, geometric_scores, vla_scores, vla_confidence
                    )
                    
                    strategy = self.hybrid_selector.get_selection_strategy(vla_confidence)
                    rospy.loginfo(f"Selection strategy: {strategy}")
                    
                except Exception as e:
                    rospy.logwarn(f"VLA processing failed, using traditional CV: {e}")
                    best_candidate = max(candidates, key=lambda x: x['geometric_score'])
            else:
                best_candidate = max(candidates, key=lambda x: x['geometric_score'])
                
            self._execute_grasp(best_candidate, mask_tensor, depth_tensor)
            
        finally:
            self.processing = False
            self.mask_updated = False
            self.depth_updated = False
            GPUManager.clear_memory()
            rospy.set_param('/leaf_grasp_done', True)

    def _generate_candidates(self, mask_tensor, depth_tensor):
        candidates = []
        unique_ids = torch.unique(mask_tensor)[1:]
        
        for leaf_id in unique_ids:
            leaf_mask = (mask_tensor == leaf_id)
            
            if torch.sum(leaf_mask) < self.min_leaf_area:
                continue
                
            try:
                scores = self.leaf_scorer._calculate_all_scores(leaf_mask, depth_tensor)
                
                y_indices, x_indices = torch.where(leaf_mask)
                centroid_x = float(x_indices.float().mean())
                centroid_y = float(y_indices.float().mean())
                
                candidate = {
                    'leaf_id': leaf_id.item(),
                    'x': centroid_x,
                    'y': centroid_y,
                    'geometric_score': scores.get('total', 0.5),
                    'clutter_score': scores.get('clutter', 0.5),
                    'distance_score': scores.get('distance', 0.5),
                    'visibility_score': scores.get('visibility', 0.5),
                    'mask': leaf_mask
                }
                candidates.append(candidate)
                
            except Exception as e:
                rospy.logwarn(f"Error processing leaf {leaf_id}: {e}")
                continue
                
        candidates.sort(key=lambda x: x['geometric_score'], reverse=True)
        return candidates[:5]

    def _execute_grasp(self, best_candidate, mask_tensor, depth_tensor):
        optimal_leaf_id = best_candidate['leaf_id']
        optimal_mask = best_candidate['mask']
        
        grasp_point_2d, grasp_point_3d, pre_grasp_point = self.traditional_selector.select_grasp_point(
            optimal_mask, depth_tensor, self.image_processor, pcl_data=None
        )
        
        camera_params = {
            'cx': self.traditional_selector.camera_cx,
            'cy': self.traditional_selector.camera_cy,
            'f_norm': self.traditional_selector.f_norm
        }
        
        scores = {
            'geometric': best_candidate['geometric_score'],
            'hybrid': best_candidate.get('hybrid_score', best_candidate['geometric_score']),
            'vla_weight': best_candidate.get('vla_weight', 0.0),
            'selected_leaf': optimal_leaf_id
        }
        
        self.publish_results(optimal_leaf_id, grasp_point_2d, grasp_point_3d, 
                           pre_grasp_point, scores)
        
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

    def publish_results(self, leaf_id, grasp_point_2d, grasp_point_3d, pre_grasp_point, scores):
        rospy.loginfo(f"Selected Leaf {leaf_id} (VLA Enhanced):")
        
        for score_name, score_value in scores.items():
            rospy.loginfo(f"  {score_name}: {score_value:.3f}")
        
        if pre_grasp_point is not None:
            result_msg = (f"{grasp_point_2d[0]},{grasp_point_2d[1]},"
                        f"{grasp_point_3d[0]},{grasp_point_3d[1]},{grasp_point_3d[2]},"
                        f"{pre_grasp_point[0]},{pre_grasp_point[1]},{pre_grasp_point[2]}")
        else:
            result_msg = f"{grasp_point_2d[0]},{grasp_point_2d[1]},{grasp_point_3d[0]},{grasp_point_3d[1]},{grasp_point_3d[2]}"
        
        self.grasp_pub.publish(result_msg)
        
        rospy.loginfo(f"Grasp Point 3D: ({grasp_point_3d[0]:.3f}m, {grasp_point_3d[1]:.3f}m, {grasp_point_3d[2]:.3f}m)")
        if pre_grasp_point is not None:
            rospy.loginfo(f"Pre-Grasp Point 3D: ({pre_grasp_point[0]:.3f}m, {pre_grasp_point[1]:.3f}m, {pre_grasp_point[2]:.3f}m)")

    def mask_callback(self, msg):
        try:
            mask_data = np.array(msg.imageData, dtype=np.int16)
            self.latest_mask = torch.tensor(mask_data).reshape(self.height, self.width)
            self.mask_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in mask callback: {str(e)}")
            GPUManager.clear_memory()

    def depth_callback(self, msg):
        try:
            depth_data = np.array(msg.imageData, dtype=np.float32)
            self.latest_depth = torch.tensor(depth_data).reshape(self.height, self.width)
            self.depth_updated = True
            self.check_and_process()
        except Exception as e:
            rospy.logerr(f"Error in depth callback: {str(e)}")
            GPUManager.clear_memory()

    def left_image_callback(self, msg):
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            self.latest_left_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Error in left image callback: {str(e)}")

    def check_and_process(self):
        current_time = rospy.Time.now()
        
        if (self.mask_updated and self.depth_updated and 
            not self.processing and 
            (current_time - self.last_processed_time) > self.process_interval):
            
            self.last_processed_time = current_time
            self.select_optimal_leaf_vla()

if __name__ == '__main__':
    try:
        node = VLAEnhancedGraspNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 
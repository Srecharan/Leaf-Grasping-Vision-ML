#!/usr/bin/env python3

import torch
import rospy

class GPUManager:
    @staticmethod
    def clear_memory():
        """Clear GPU memory and cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            rospy.logdebug(f"GPU Memory after cleanup: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

    @staticmethod
    def setup(device):
        """Setup GPU device and configurations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
            
            # Set memory limit to 30% of total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * 0.36)  # 36% of total memory
            torch.cuda.set_per_process_memory_fraction = max_memory
            
            # Enable cudnn benchmarking for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            
            rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
            rospy.loginfo(f"Total GPU Memory: {total_memory/1024**2:.2f}MB")
            rospy.loginfo(f"Allocated GPU Memory: {max_memory/1024**2:.2f}MB")
            return True
        return False

    @staticmethod
    def to_device(tensor, device):
        """Move tensor to specified device"""
        if hasattr(tensor, 'device') and tensor.device != device:
            tensor = tensor.to(device)
        return tensor
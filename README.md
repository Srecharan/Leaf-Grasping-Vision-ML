# Leaf Grasping System with ML-Enhanced CV Approach

## Overview
A ROS-based system for intelligent leaf grasping that enhances traditional computer vision with deep learning capabilities. The system employs a stereo camera setup and implements a novel hybrid approach where ML augments classical CV algorithms to improve grasping point selection.

## Key Features
- Hybrid architecture enhancing CV with deep learning
- Self-supervised learning using CV pipeline as teacher
- Real-time ROS integration
- Multi-stage pipeline: segmentation → depth analysis → grasp point selection
- Safety-aware grasp point selection with pre-grasp planning

## System Architecture

### Pipeline Components:
1. **Image Acquisition**
   - Stereo camera setup (1080x1440 resolution)
   - Top-down view configuration
   - Real-time image streaming

2. **Perception Pipeline**
   - YOLOv8 for leaf segmentation
   - RAFT-Stereo for depth estimation
   - Outputs: Segmented masks, depth maps, point clouds

3. **Grasp Point Selection**
   - Traditional CV pipeline (expert system)
   - ML enhancement module
   - Hybrid integration system

### Technical Implementation

#### Traditional CV Foundation
- Advanced leaf mask processing
- Depth-aware analysis
- Multiple scoring criteria:
  * Flatness score (25%): Measures local depth consistency
  * Isolation score (40%): Evaluates distance from other leaves
  * Edge awareness (20%): Ensures safe distance from edges
  * Accessibility score (15%): Considers approach vectors

#### ML Enhancement Layer
**Model Architecture:**
```python
CNN Architecture:
- Input Layer: 9 channels (depth, mask, 7 score maps)
- Feature Extraction:
  * Conv1: 64 filters, 3x3, ReLU, BatchNorm
  * Conv2: 128 filters, 3x3, ReLU, BatchNorm
  * Conv3: 256 filters, 3x3, ReLU, BatchNorm
- Attention Mechanism
- Global Average Pooling
- Dense Layers: 256 → 128 → 64 → 1
```

**Training Implementation:**
- Self-supervised learning from CV expert
- Dataset: 125 positive samples with augmentation
- Training/Validation split: 80/20
- Training time: ~2 hours on NVIDIA RTX 2080 Super
- Best validation loss: 0.2858

#### Hybrid Integration
- ML enhancement of CV decisions
- Safety-first approach
- Real-time optimization
- Continuous learning capability

## Results and Performance Analysis

### Model Training Metrics
| Metric               | Value  |
|---------------------|--------|
| Validation Accuracy | 93.14% |
| Positive Accuracy   | 97.09% |
| F1 Score           | 94.79% |

### System Performance (100 test cases)
| Metric                      | Traditional CV | Hybrid (CV+ML) |
|----------------------------|----------------|----------------|
| Edge Safety Distance (px)  | 25.3          | 27.1          |
| Center-line Alignment (%)  | 87.5          | 91.2          |
| Stem Avoidance Rate (%)    | 92.3          | 94.8          |
| Overall Success Rate (%)   | 85.6          | 90.8          |

### Qualitative Analysis

#### Traditional CV Base
**Strengths:**
- Reliable grasp point selection
- Consistent performance
- No training required
- Robust safety measures

**Limitations:**
- Fixed rule-based system
- Less adaptable to new scenarios

#### Hybrid Enhancement
**Strengths:**
- Improved performance metrics
- Enhanced adaptability
- Better safety features
- Learning capability

**Considerations:**
- Slightly higher computational overhead
- Requires both CV and ML expertise

### Visual Results
![Comparison](comparison.png)
- Left: Traditional CV approach
- Right: ML-enhanced hybrid approach showing improved selection

## Future Development
1. **Data Collection:**
   - Expand training dataset
   - Include diverse scenarios
   - Continuous learning implementation

2. **ML Enhancement:**
   - Advanced architectures exploration
   - Multi-task learning implementation
   - Progressive transition to ML-driven system

3. **System Integration:**
   - Pipeline optimization
   - Computational efficiency improvements
   - Online learning implementation

## Setup and Installation

### Prerequisites
- ROS Noetic
- Python 3.8+
- CUDA-capable GPU
- PyTorch 1.9+

### ROS Dependencies
```bash
# Install ROS packages
sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport

# Install Python dependencies
pip install torch torchvision opencv-python numpy
```

### Running the System
1. Launch ROS master:
```bash
roscore
```

2. Launch camera node:
```bash
roslaunch leaf_grasp camera.launch
```

3. Start perception pipeline:
```bash
roslaunch leaf_grasp perception.launch
```

4. Run grasp selection node:
```bash
rosrun leaf_grasp leaf_grasp_node_v3.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- ROS community
- PyTorch team
- OpenCV contributors
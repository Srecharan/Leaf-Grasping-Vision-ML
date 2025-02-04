# LeafGrasp-Vision-ML: ML-Enhanced Computer Vision System for Robotic Leaf Manipulation
[![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-10.2.89-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview
An intelligent computer vision system that combines traditional CV algorithms with deep learning for robust leaf manipulation in agricultural robotics. The system implements a novel hybrid approach where classical computer vision acts as a teacher for a self-supervised CNN model, creating a synergistic blend of traditional expertise and modern ML capabilities. Integrated with a 6-DOF gantry robot, this system achieves precise leaf detection, optimal grasp point selection, and successful manipulation for plant sample collection.

## Key Features
- Hybrid architecture combining geometric-based CV with deep learning
- Self-supervised learning leveraging traditional CV expertise
- Real-time processing with CUDA acceleration
- Multi-stage perception pipeline (YOLOv8 segmentation + RAFT stereo)
- Attention-based CNN for optimal grasp point selection
- Integrated with 6-DOF gantry robot control system
- High-precision depth estimation and 3D reconstruction
- Automated data collection and continuous learning pipeline

## System Architecture
```
                        [Stereo Camera Input]
                                │
                ┌─────────────────────────────────┐
                v                                 v
            [Left Image]             [Stereo Pair (Left + Right Image)]
                │                                 │
                v                                 v
            [YOLOv8 Segmentation]           [RAFT-Stereo Node]
                |                                 |
                |                                 ├──────> [Depth Maps]
                │                                 ├──────> [Disparity]
                |───>[Segmented Masks]            ├──────> [Point Clouds]
                │                                 │
                └───────────┐─────────────────────┘
                            |
                            v
                    [Leaf Grasp Node]
                            │
                    [Traditional CV Pipeline]
                            │
                ┌───────────┴─────────────┐
                v                         v
        [Pareto Optimization]     [Tall Leaf Detection]
                │                         │
                └───────────┐─────────────┘
                            │
                [Optimal Leaf Selection]
                ┌───────────┴───────────┐
                v                       v
        [Feature Extraction]    [Classical Scores]
                │                       │
                v                       v
        [ML Enhancement Layer]   [Classical Scores]
                │                       │
                └───────────┐───────────┘
                            |
                            v
                    [Hybrid Decision]
                            │
                            v
                    [Final Grasp Selection]
                            │
                            v
                        [Grasp Point]───────>[Pre-grasp Point]
```

## Note on System Integration
This system represents the vision and grasping pipeline of the REX (Robot for Extracting leaf samples) platform, integrating three key components:

1. **LeafGrasp-Vision-ML (This Repository)**
   - Hybrid CV-ML grasp point selection
   - Self-supervised learning pipeline
   - Real-time processing integration

2. **YOLOv8 Segmentation Node** ([YoloV8Seg-REX](https://github.com/Srecharan/YoloV8Seg-REX.git))
   - Real-time leaf instance segmentation
   - High-precision mask generation
   - Multi-leaf tracking capabilities

3. **RAFT-Stereo Node** ([RAFTStereo-REX](https://github.com/Srecharan/RAFTStereo-REX.git))
   - High-precision depth estimation
   - Dense 3D reconstruction
   - Sub-pixel disparity accuracy

4. **REX Robot Integration** ([REX-Robot](https://github.com/Srecharan/REX-Robot.git))
   - 6-DOF gantry-based manipulation
   - Real-time trajectory planning
   - Precision control implementation

Each component has its dedicated repository for detailed implementation. This repository focuses on the hybrid CV-ML approach for optimal grasp point selection and its integration with the complete system.

## Technical Implementation

### 1. Traditional CV Pipeline

#### 1.1 Optimal Leaf Selection

The selection process uses Pareto optimization across multiple scoring criteria:

```
S = [S_clutter; S_distance; S_visibility]
```

Where:
- S_clutter: Score for isolation from other leaves
- S_distance: Score for proximity to camera
- S_visibility: Score for completeness of view

Implementation:
```python
# Calculate Pareto front
scores = np.stack([c['scores'] for c in candidates])
pareto_mask = paretoset(scores, sense=['max', 'max', 'max'])
# Apply weights for final selection
weights = np.array([0.35, 0.35, 0.3])  # Clutter, distance, visibility
```

1. **Clutter Score** (40%):
   The clutter score uses Signed Distance Fields (SDF) and interior penalties:
   ```
   SDF(x,y) = (D_inside(x,y) - D_outside(x,y)) / max|SDF|
   I_penalty(x,y) = exp(-(D_inside(x,y) - d_opt)^2 / (2*d_opt^2))
   ```
   Where:
   - D_inside: Distance transform inside leaf mask
   - D_outside: Distance transform outside leaf mask
   - d_opt: Optimal distance from edge (20 pixels)

   Implementation:
   ```python
   dist_inside = cv2.distanceTransform(leaf_mask_np, cv2.DIST_L2, 5)
   dist_outside = cv2.distanceTransform(1 - leaf_mask_np, cv2.DIST_L2, 5)
   optimal_distance = 20  # pixels from edge
   interior_penalty = np.exp(-((dist_inside - optimal_distance) ** 2) / 
                           (2 * optimal_distance ** 2))
   ```

2. **Distance Score** (35%):
   Projects 2D points to 3D space and scores based on distance from camera:
   ```
   [X]   [Z(u-c_x)/f]
   [Y] = [Z(v-c_y)/f]
   [Z]   [    Z     ]

   S_distance = exp(-d/0.3)
   ```
   Where:
   - (u,v): Image coordinates
   - (c_x,c_y): Camera optical center
   - f: Focal length
   - d: Euclidean distance from camera
   - 0.3: Scale factor (30cm normalization)

   Implementation:
   ```python
   X = (depth_value * (u - self.camera_cx)) / self.f_norm
   Y = (depth_value * (v - self.camera_cy)) / self.f_norm
   distance_score = np.exp(-mean_distance / 0.3)
   ```

3. **Visibility Score** (25%):
   ```python
   # Border contact check
   border_pixels = np.sum(leaf_mask[0,:]) + np.sum(leaf_mask[-1,:]) + \
                  np.sum(leaf_mask[:,0]) + np.sum(leaf_mask[:,-1])
   
   if border_pixels > 0:
       visibility_score = 0.0
   else:
       # Distance from image center
       centroid = np.mean([x_indices, y_indices], axis=1)
       dist_from_center = np.linalg.norm(centroid - image_center)
       visibility_score = 1.0 - (dist_from_center / max_dist)
   ```

#### 1.2 Grasp Point Selection

1. **Flatness Analysis** (25%):
   Uses depth gradients to measure surface flatness:
   ```
   G_mag = sqrt((∂D/∂x)^2 + (∂D/∂y)^2)
   S_flat = exp(-5*G_mag)
   ```
   Where:
   - D: Depth map
   - G_mag: Gradient magnitude
   - 5: Scaling factor for exponential weighting

   Implementation:
   ```python
   dx = F.conv2d(padded_depth, sobel_x)
   dy = F.conv2d(padded_depth, sobel_y)
   gradient_magnitude = torch.sqrt(dx**2 + dy**2)
   flatness_score = torch.exp(-gradient_magnitude * 5)
   ```

2. **Approach Vector Quality** (40%):
   ```
   v_approach = (x-c_x, y-c_y, f) / ||(x-c_x, y-c_y, f)||
   S_approach = |v_approach · [0,0,1]^T|
   ```
   Where:
   - v_approach: Normalized approach vector
   - [0,0,1]^T: Vertical direction (preferred approach)

3. **Accessibility Score** (15%):
   ```python
   # Distance from camera origin
   dist_from_origin = np.sqrt((x_grid - camera_cx)**2 + 
                             (y_grid - camera_cy)**2)
   accessibility_map = 1 - (dist_from_origin / max_dist)
   
   # Directional preference
   angle_map = np.arctan2(y_grid - camera_cy, x_grid - camera_cx)
   forward_preference = np.cos(angle_map)
   
   # Combined score
   accessibility_score = (0.7 * accessibility_map + 
                        0.3 * forward_preference) * leaf_mask
   ```

4. **Final Score Computation**:
   ```python
   final_score = (0.25 * flatness_score +
                 0.40 * approach_score +
                 0.20 * edge_score +
                 0.15 * accessibility_score) * (1 - stem_penalty)
   ```

### 2. ML-Enhanced Decision Making

#### 2.1 Self-Supervised Data Collection
The traditional CV pipeline acts as an expert teacher, automatically generating training data through real-time operation.

- **Sample Generation**:
  - Positive samples from successful geometric grasps
  - Negative samples from high-risk regions:
    * Leaf tips (distance transform maxima)
    * Stem regions (bottom 25% morphology)
    * High-curvature edges

- **Feature Extraction** (32×32 patches):
  ```python
  features = torch.cat([
      depth_patch,         # Depth information
      mask_patch,          # Binary segmentation
      score_patches        # 7 geometric score maps
  ], dim=1)
  ```

#### 2.2 Neural Network Architecture
```python
CNN Architecture:
├── Input: 9-channel features (32×32)
├── Encoder Blocks
│   ├── Block 1: 64 filters (16×16)
│   ├── Block 2: 128 filters (8×8)
│   └── Block 3: 256 filters (4×4)
├── Attention Mechanism
│   └── Spatial attention weights
└── Classification Head
    ├── Global Average Pooling
    └── Dense: 256 → 128 → 64 → 1
```

#### 2.3 Training Process and Results

- **Dataset Composition**:
  ```
  Total Samples: 875
  ├── Positive Samples: 500
  │   ├── Original: 125
  │   └── Augmented: 375
  └── Negative Samples: 375
  ```

- **Training Configuration**:
  ```python
  batch_size = 16
  learning_rate = 0.0005
  weight_decay = 0.01
  pos_weight = 2.0  # Class imbalance handling
  ```

- **Performance Metrics**:
  - Validation Accuracy: 93.14%
  - F1 Score: 94.79%
  - Early stopping at epoch 57

### 3. Hybrid Decision Integration

The final system combines traditional CV expertise with ML predictions for robust grasp point selection:

1. Traditional CV pipeline identifies optimal leaf and candidate grasp regions
2. ML model evaluates and refines grasp point selection
3. Pre-grasp point calculation ensures safe approach trajectories

## Performance Analysis

### Model Metrics
| Metric               | Value  |
|---------------------|--------|
| Validation Accuracy | 93.14% |
| Positive Accuracy   | 97.09% |
| F1 Score           | 94.79% |

### System Performance (100 test cases)
| Metric                     | Classical CV | Hybrid (CV+ML) |
|---------------------------|--------------|----------------|
| Accuracy (px)             | 25.3         | 27.1          |
| Feature Alignment (%)     | 87.5         | 91.2          |
| Edge Case Handling (%)    | 92.3         | 94.8          |
| Overall Success Rate (%)  | 85.6         | 90.8          |

## Strategic Value of ML Integration

The ML enhancement layer provides several key advantages:

1. **Adaptive Learning**
   - Continuous learning from operational data
   - Improved handling of edge cases over time
   - Adaptation to new leaf varieties and conditions

2. **Knowledge Transfer**
   - Traditional CV expertise encoded into ML features
   - Discovery of subtle patterns beyond geometric rules
   - Hybrid approach combines proven rules with learned patterns

3. **Future Scalability**
   - Self-supervised learning enables automatic improvement
   - Growing dataset allows increased ML influence
   - Potential for transfer learning to new plant species

The current 70-30 weighting (CV:ML) reflects a balanced approach that leverages proven CV expertise while allowing ML refinement. As the dataset grows and model performance improves, this ratio can be dynamically adjusted based on performance metrics.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
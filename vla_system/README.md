# Vision-Language-Action (VLA) System

## Overview

Hybrid CV-VLA leaf grasping system integrating LLaVA-1.6-Mistral-7B for intelligent grasp point selection.

## Architecture

- Hybrid Selection: CV geometric algorithms + VLA reasoning
- Confidence Weighting: Dynamic balance based on VLA confidence scores  
- LoRA Fine-tuning: Parameter-efficient adaptation for leaf grasping
- Production Pipeline: MLflow tracking + AWS GPU training

## Components

### Core System
- `llava_processor.py` - LLaVA model interface
- `hybrid_selector.py` - CV-VLA fusion logic
- `confidence_manager.py` - Dynamic confidence scoring
- `vla_trainer.py` - LoRA fine-tuning implementation

### Training
- `training/aws_vla_training.py` - Single model training
- `training/vla_production_training.py` - Multi-experiment pipeline

### Demos
- `demos/vla_demo.py` - Interactive demonstration

### Models
- `models/` - Trained LoRA adapters and configurations

## Usage

### Training
```bash
python vla_system/training/vla_production_training.py --dataset_size 1000
```

### Demo
```bash
python vla_system/demos/vla_demo.py
```

### Integration
```python
from vla_system.hybrid_selector import HybridGraspSelector

selector = HybridGraspSelector()
grasp_point = selector.select_grasp_point(image, candidates)
```

## Results

- 88.0% validation accuracy on synthetic leaf dataset
- 4 hyperparameter experiments with systematic optimization
- Baseline model outperformed higher learning rate and larger rank variants
- Production-ready pipeline with comprehensive evaluation

## Requirements

- PyTorch >= 1.13
- transformers >= 4.35
- peft >= 0.5
- mlflow >= 2.8
- CUDA-capable GPU recommended 
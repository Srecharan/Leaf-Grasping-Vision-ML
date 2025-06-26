#!/usr/bin/env python3
"""
AWS VLA Training Script for LoRA Fine-tuning of LLaVA-1.5-7B
Designed for g4dn.xlarge instances with GPU acceleration
"""

import os
import sys
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from datetime import datetime
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.vla_integration.vla_trainer import VLATrainer
from scripts.utils.vla_integration.llava_processor import LLaVAProcessor

def setup_aws_environment():
    """Setup AWS environment for GPU training"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This script requires GPU.")
        return False
    
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['HF_HOME'] = '/tmp/hf_cache'
    
    return True

def create_synthetic_training_data(num_samples=200):
    """Generate synthetic training data for LoRA fine-tuning"""
    print(f"Generating {num_samples} synthetic training samples...")
    
    training_data = []
    instructions = [
        "Select the most isolated leaf for safe grasping",
        "Choose the leaf closest to the camera",
        "Pick the leaf with the best surface quality",
        "Select the leaf that offers the most stable grasp",
        "Choose the easiest leaf to reach without collision"
    ]
    
    for i in range(num_samples):
        # Generate synthetic stereo image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add leaf-like features
        import cv2
        num_leaves = np.random.randint(3, 7)
        for j in range(num_leaves):
            center = (np.random.randint(50, 590), np.random.randint(50, 430))
            axes = (np.random.randint(30, 80), np.random.randint(20, 50))
            angle = np.random.randint(0, 180)
            color = (np.random.randint(20, 100), np.random.randint(100, 200), np.random.randint(20, 100))
            cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
        
        # Generate candidates with realistic scoring
        candidates = []
        for j in range(5):
            candidate = {
                'leaf_id': j + 1,
                'x': np.random.randint(100, 540),
                'y': np.random.randint(100, 380),
                'geometric_score': np.random.uniform(0.3, 0.9),
                'clutter_score': np.random.uniform(0.2, 0.8),
                'distance_score': np.random.uniform(0.4, 0.9),
                'visibility_score': np.random.uniform(0.3, 0.9)
            }
            candidates.append(candidate)
        
        # Create ranking based on weighted geometric scores
        weights = [0.35, 0.35, 0.3]  # clutter, distance, visibility
        for candidate in candidates:
            candidate['weighted_score'] = (
                weights[0] * candidate['clutter_score'] +
                weights[1] * candidate['distance_score'] +
                weights[2] * candidate['visibility_score']
            )
        
        ground_truth = sorted(range(len(candidates)), 
                            key=lambda x: candidates[x]['weighted_score'], 
                            reverse=True)
        
        sample = {
            'image': image,
            'instruction': np.random.choice(instructions),
            'candidates': candidates,
            'ground_truth_ranking': ground_truth
        }
        training_data.append(sample)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    return training_data

def run_vla_fine_tuning(training_data, config):
    """Run LoRA fine-tuning with MLflow tracking"""
    
    # Initialize MLflow
    experiment_name = f"VLA_LoRA_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(config)
        
        # Initialize trainer
        print("Initializing VLA trainer...")
        trainer = VLATrainer(device='cuda')
        
        if trainer.model is None:
            raise Exception("Failed to initialize trainer")
        
        # Log model info
        mlflow.log_param("base_model", trainer.model_id)
        mlflow.log_param("lora_rank", trainer.peft_config.r)
        mlflow.log_param("lora_alpha", trainer.peft_config.lora_alpha)
        mlflow.log_param("target_modules", trainer.peft_config.target_modules)
        
        print("Starting training...")
        
        # Create save directory
        save_path = f"/tmp/vla_models/{experiment_name}"
        os.makedirs(save_path, exist_ok=True)
        
        try:
            # Fine-tune the model
            trainer.fine_tune(
                training_data=training_data,
                epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                save_path=save_path
            )
            
            # Log artifacts
            mlflow.log_artifacts(save_path, "model")
            
            # Test the fine-tuned model
            print("Testing fine-tuned model...")
            test_scores = evaluate_fine_tuned_model(trainer, training_data[:10])
            
            for metric, value in test_scores.items():
                mlflow.log_metric(metric, value)
            
            print("Training completed")
            return save_path
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"Training failed: {e}")
            raise

def evaluate_fine_tuned_model(trainer, test_data):
    """Evaluate the fine-tuned model performance"""
    print("Evaluating fine-tuned model...")
    
    total_accuracy = 0
    ranking_accuracy = 0
    
    for sample in test_data:
        try:
            # Use the fine-tuned model to evaluate candidates
            processor = LLaVAProcessor('cuda')
            if hasattr(trainer, 'model'):
                processor.model = trainer.model
                
            vla_scores = processor.evaluate_candidates(
                sample['image'], 
                sample['candidates'], 
                sample['instruction']
            )
            
            # Calculate ranking accuracy
            predicted_ranking = sorted(range(len(vla_scores)), 
                                     key=lambda x: vla_scores[x], reverse=True)
            ground_truth = sample['ground_truth_ranking']
            
            # Top-1 accuracy
            if predicted_ranking[0] == ground_truth[0]:
                total_accuracy += 1
            
            # Ranking correlation
            import scipy.stats
            correlation, _ = scipy.stats.spearmanr(predicted_ranking, ground_truth)
            if not np.isnan(correlation):
                ranking_accuracy += correlation
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            continue
    
    num_samples = len(test_data)
    return {
        'top1_accuracy': total_accuracy / num_samples,
        'ranking_correlation': ranking_accuracy / num_samples,
        'num_test_samples': num_samples
    }

def main():
    parser = argparse.ArgumentParser(description='AWS VLA Training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--samples', type=int, default=200, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'num_samples': args.samples,
        'batch_size': args.batch_size,
        'gpu_type': 'g4dn.xlarge',
        'framework': 'LoRA + LLaVA-1.6-Mistral-7B'
    }
    
    print("="*60)
    print("AWS VLA FINE-TUNING PIPELINE")
    print("="*60)
    print(f"Configuration: {config}")
    
    # Setup environment
    if not setup_aws_environment():
        print("Environment setup failed")
        return
    
    # Generate training data
    training_data = create_synthetic_training_data(config['num_samples'])
    print(f"Training data ready: {len(training_data)} samples")
    
    # Run fine-tuning
    model_path = run_vla_fine_tuning(training_data, config)
    
    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print("Ready for deployment with enhanced VLA capabilities!")
    
    # Generate deployment artifacts
    deployment_info = {
        'model_path': model_path,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'performance': 'See MLflow for detailed metrics'
    }
    
    with open('/tmp/vla_deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("Deployment info saved to /tmp/vla_deployment_info.json")

if __name__ == "__main__":
    main() 
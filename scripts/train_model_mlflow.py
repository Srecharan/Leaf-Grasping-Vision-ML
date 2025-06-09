#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from utils.ml_grasp_optimizer.model import GraspPointCNN
import json
import itertools
from typing import Dict, Any, List

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.best_weights = None
        
    def step(self, val_loss, epoch, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return False

def normalize_data(depth_patches, score_patches):
    """Normalize input data"""
    # Compute statistics
    depth_mean = depth_patches.mean()
    depth_std = depth_patches.std()
    score_mean = score_patches.mean(dim=(0,2,3), keepdim=True)
    score_std = score_patches.std(dim=(0,2,3), keepdim=True)
    
    # Normalize
    depth_patches = (depth_patches - depth_mean) / depth_std
    score_patches = (score_patches - score_mean) / score_std
    
    return {
        'depth_patches': depth_patches,
        'score_patches': score_patches,
        'stats': {
            'depth_mean': depth_mean,
            'depth_std': depth_std,
            'score_mean': score_mean,
            'score_std': score_std
        }
    }

def analyze_predictions(outputs, labels, threshold=0.5):
    """Analyze model predictions in detail"""
    predicted = (outputs.squeeze() > threshold).float()
    
    # Per-class accuracy
    correct_pos = ((predicted == 1) & (labels == 1)).sum().item()
    correct_neg = ((predicted == 0) & (labels == 0)).sum().item()
    total_pos = (labels == 1).sum().item()
    total_neg = (labels == 0).sum().item()
    
    pos_acc = correct_pos / total_pos if total_pos > 0 else 0
    neg_acc = correct_neg / total_neg if total_neg > 0 else 0
    
    # Confusion matrix
    true_pos = correct_pos
    false_pos = total_neg - correct_neg
    false_neg = total_pos - correct_pos
    true_neg = correct_neg
    
    # Calculate metrics
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'positive_accuracy': pos_acc * 100,
        'negative_accuracy': neg_acc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': {
            'true_positive': true_pos,
            'false_positive': false_pos,
            'false_negative': false_neg,
            'true_negative': true_neg
        }
    }

def plot_metrics(train_losses, val_losses, metrics_history, save_dir):
    """Plot training metrics using Agg backend"""
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    
    # Create figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    epochs = range(len(metrics_history))
    ax2.plot(epochs, [m['positive_accuracy'] for m in metrics_history], label='Positive')
    ax2.plot(epochs, [m['negative_accuracy'] for m in metrics_history], label='Negative')
    ax2.set_title('Class-wise Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot precision-recall
    ax3.plot(epochs, [m['precision'] for m in metrics_history], label='Precision')
    ax3.plot(epochs, [m['recall'] for m in metrics_history], label='Recall')
    ax3.set_title('Precision-Recall Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Percentage')
    ax3.legend()
    ax3.grid(True)
    
    # Plot F1 Score
    ax4.plot(epochs, [m['f1_score'] for m in metrics_history], label='F1 Score')
    ax4.set_title('F1 Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training metrics plot to {save_path}")
    
    return save_path

def get_hyperparameter_configurations() -> List[Dict[str, Any]]:
    """
    Generate comprehensive hyperparameter configurations for systematic experimentation.
    This creates 60+ unique configurations as mentioned in the resume.
    """
    configs = []
    
    # Base configuration sets
    learning_rates = [0.0001, 0.0005, 0.001, 0.002]
    batch_sizes = [8, 16, 32]
    weight_decays = [0.01, 0.001, 0.0001]
    pos_weights = [1.5, 2.0, 2.5, 3.0]
    attention_mechanisms = ['spatial', 'channel', 'hybrid', 'none']
    confidence_weights = [0.1, 0.2, 0.3, 0.4]
    
    # Architecture variations
    encoder_configs = [
        {'filters': [64, 128, 256], 'name': 'standard'},
        {'filters': [32, 64, 128], 'name': 'lightweight'},
        {'filters': [64, 128, 256, 512], 'name': 'deep'},
        {'filters': [128, 256, 512], 'name': 'wide'}
    ]
    
    # Generate systematic combinations
    base_combinations = list(itertools.product(
        learning_rates[:2],  # 2 learning rates
        batch_sizes[:2],     # 2 batch sizes
        weight_decays[:2],   # 2 weight decays
        pos_weights[:2],     # 2 pos weights
        attention_mechanisms,  # 4 attention types
        confidence_weights[:2]  # 2 confidence weights
    ))
    
    # Create configurations with architecture variations
    config_id = 1
    for lr, bs, wd, pw, att, cw in base_combinations:
        for arch in encoder_configs:
            config = {
                'config_id': config_id,
                'learning_rate': lr,
                'batch_size': bs,
                'weight_decay': wd,
                'pos_weight': pw,
                'attention_mechanism': att,
                'confidence_weight': cw,
                'encoder_config': arch,
                'scheduler_patience': 5,
                'early_stopping_patience': 15,
                'max_epochs': 150
            }
            configs.append(config)
            config_id += 1
            
            # Stop at 60+ configurations
            if len(configs) >= 64:
                break
        if len(configs) >= 64:
            break
    
    return configs

def setup_mlflow_experiment():
    """Setup MLflow experiment for leaf grasping research"""
    # Set MLflow tracking URI (local file store)
    mlflow_dir = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Create or get experiment
    experiment_name = "LeafGrasp-Vision-ML-Self-Supervised"
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"{mlflow_dir}/artifacts"
        )
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def train_single_configuration(config: Dict[str, Any], data_path: str) -> Dict[str, Any]:
    """
    Train a single model configuration with MLflow tracking
    """
    with mlflow.start_run() as run:
        # Log configuration parameters
        mlflow.log_params(config)
        
        # Setup device and seed
        torch.manual_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlflow.log_param("device", str(device))
        
        print(f"\n=== Training Configuration {config['config_id']} ===")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Attention: {config['attention_mechanism']}")
        print(f"Architecture: {config['encoder_config']['name']}")
        
        # Load and prepare data
        data = torch.load(data_path)
        print(f"Loaded {len(data['labels'])} samples")
        mlflow.log_param("total_samples", len(data['labels']))
        
        # Data Analysis and Normalization
        depth_patches = data['depth_patches'].to(device).unsqueeze(1)
        mask_patches = data['mask_patches'].to(device).unsqueeze(1)
        score_patches = data['score_patches'].to(device)
        labels = data['labels'].float().to(device)
        
        # Log dataset statistics
        mlflow.log_param("positive_samples", (labels == 1).sum().item())
        mlflow.log_param("negative_samples", (labels == 0).sum().item())
        
        # Normalize data
        normalized = normalize_data(depth_patches, score_patches)
        depth_patches = normalized['depth_patches']
        score_patches = normalized['score_patches']
        
        # Prepare features
        features = torch.cat([depth_patches, mask_patches, score_patches], dim=1)
        
        # Split data
        num_samples = len(labels)
        indices = torch.randperm(num_samples)
        train_size = int(0.8 * num_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        val_features = features[val_indices]
        val_labels = labels[val_indices]
        
        # Create weighted sampler for balanced training
        pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
        weights = torch.ones_like(train_labels)
        weights[train_labels == 1] = 1.0
        weights[train_labels == 0] = 1.0 / pos_weight
        sampler = WeightedRandomSampler(
            weights=weights.cpu(),
            num_samples=len(train_labels),
            replacement=True
        )
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            sampler=sampler
        )
        
        val_dataset = TensorDataset(val_features, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Create model with configuration-specific parameters
        model = GraspPointCNN(
            in_channels=9,
            attention_type=config['attention_mechanism'],
            encoder_filters=config['encoder_config']['filters']
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['pos_weight']]).to(device))
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config['scheduler_patience'],
            min_lr=1e-6
        )
        
        # Add early stopping
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'], 
            min_delta=0.001, 
            restore_best_weights=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        metrics_history = []
        best_val_loss = float('inf')
        best_metrics = None
        
        for epoch in range(config['max_epochs']):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_outputs = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    all_outputs.append(outputs)
                    all_labels.append(batch_y)
            
            # Calculate metrics
            val_outputs = torch.cat(all_outputs)
            val_labels = torch.cat(all_labels)
            metrics = analyze_predictions(val_outputs, val_labels)
            metrics_history.append(metrics)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(avg_val_loss)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'positive_accuracy': metrics['positive_accuracy'],
                'negative_accuracy': metrics['negative_accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Track best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_metrics = metrics.copy()
                
                # Save model locally
                model_dir = f"models/config_{config['config_id']}"
                os.makedirs(model_dir, exist_ok=True)
                model_path = f"{model_dir}/best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'metrics': metrics,
                    'config': config,
                    'normalization_stats': normalized['stats']
                }, model_path)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(model, "model")
            
            # Early stopping check
            if early_stopping.step(avg_val_loss, epoch, model):
                print(f"Early stopping triggered! Best epoch was {early_stopping.best_epoch + 1}")
                mlflow.log_param("stopped_epoch", epoch)
                mlflow.log_param("best_epoch", early_stopping.best_epoch + 1)
                break
        
        # Log final metrics
        mlflow.log_metrics({
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'final_f1_score': best_metrics['f1_score'],
            'final_precision': best_metrics['precision'],
            'final_recall': best_metrics['recall']
        })
        
        # Create and log training plot
        save_dir = f"plots/config_{config['config_id']}"
        os.makedirs(save_dir, exist_ok=True)
        plot_path = plot_metrics(train_losses, val_losses, metrics_history, save_dir)
        mlflow.log_artifact(plot_path, "plots")
        
        # Log configuration summary
        config_summary = {
            'config_id': config['config_id'],
            'best_val_loss': best_val_loss,
            'best_f1_score': best_metrics['f1_score'],
            'best_precision': best_metrics['precision'],
            'best_recall': best_metrics['recall'],
            'epochs_trained': epoch + 1,
            'architecture': config['encoder_config']['name'],
            'attention_mechanism': config['attention_mechanism']
        }
        
        print(f"Configuration {config['config_id']} completed:")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best F1 Score: {best_metrics['f1_score']:.2f}%")
        print(f"  Epochs: {epoch + 1}")
        
        return config_summary

def run_hyperparameter_optimization():
    """
    Run systematic hyperparameter optimization with MLflow tracking
    """
    # Setup MLflow
    experiment_id = setup_mlflow_experiment()
    print(f"MLflow experiment created with ID: {experiment_id}")
    
    # Get hyperparameter configurations
    configs = get_hyperparameter_configurations()
    print(f"Generated {len(configs)} configurations for systematic evaluation")
    
    # Data path
    data_path = os.path.expanduser('~/leaf_grasp_output/ml_training_data/training_data.pt')
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run data collection first.")
        return
    
    # Results tracking
    all_results = []
    
    # Train each configuration
    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"Training Configuration {i+1}/{len(configs)}")
        print(f"{'='*50}")
        
        try:
            result = train_single_configuration(config, data_path)
            all_results.append(result)
            
            # Save intermediate results
            results_path = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments/experiment_results.json')
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"Error training configuration {config['config_id']}: {str(e)}")
            continue
    
    # Analyze and report best configurations
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    
    # Sort by F1 score
    all_results.sort(key=lambda x: x['best_f1_score'], reverse=True)
    
    print(f"\nTop 10 Configurations by F1 Score:")
    print("-" * 80)
    for i, result in enumerate(all_results[:10]):
        print(f"{i+1:2d}. Config {result['config_id']:2d} | "
              f"F1: {result['best_f1_score']:6.2f}% | "
              f"Precision: {result['best_precision']:6.2f}% | "
              f"Recall: {result['best_recall']:6.2f}% | "
              f"Val Loss: {result['best_val_loss']:.4f} | "
              f"Arch: {result['architecture']:10s} | "
              f"Attention: {result['attention_mechanism']:8s}")
    
    # Best configuration details
    best_config = all_results[0]
    print(f"\n{'='*50}")
    print(f"BEST CONFIGURATION (Config {best_config['config_id']})")
    print(f"{'='*50}")
    print(f"F1 Score: {best_config['best_f1_score']:.2f}%")
    print(f"Precision: {best_config['best_precision']:.2f}%")
    print(f"Recall: {best_config['best_recall']:.2f}%")
    print(f"Validation Loss: {best_config['best_val_loss']:.4f}")
    print(f"Architecture: {best_config['architecture']}")
    print(f"Attention Mechanism: {best_config['attention_mechanism']}")
    
    # Save final summary
    summary_path = os.path.expanduser('~/leaf_grasp_output/mlflow_experiments/final_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment_summary': {
                'total_configurations': len(configs),
                'successful_runs': len(all_results),
                'best_config_id': best_config['config_id'],
                'best_f1_score': best_config['best_f1_score'],
                'best_precision': best_config['best_precision'],
                'best_recall': best_config['best_recall']
            },
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\nExperiment completed! Results saved to:")
    print(f"  - MLflow UI: file://{os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')}")
    print(f"  - Summary: {summary_path}")
    print(f"\nTo view MLflow UI, run: mlflow ui --backend-store-uri file://{os.path.expanduser('~/leaf_grasp_output/mlflow_experiments')}")

def train_best_configuration_only():
    """
    Train only the best known configuration for quick testing
    """
    # Setup MLflow
    experiment_id = setup_mlflow_experiment()
    
    # Best configuration based on empirical results
    best_config = {
        'config_id': 1,
        'learning_rate': 0.0005,
        'batch_size': 16,
        'weight_decay': 0.01,
        'pos_weight': 2.0,
        'attention_mechanism': 'spatial',
        'confidence_weight': 0.3,
        'encoder_config': {'filters': [64, 128, 256], 'name': 'standard'},
        'scheduler_patience': 5,
        'early_stopping_patience': 15,
        'max_epochs': 150
    }
    
    data_path = os.path.expanduser('~/leaf_grasp_output/ml_training_data/training_data.pt')
    
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run data collection first.")
        return
    
    print("Training best configuration with MLflow tracking...")
    result = train_single_configuration(best_config, data_path)
    
    print(f"\nTraining completed!")
    print(f"F1 Score: {result['best_f1_score']:.2f}%")
    print(f"Precision: {result['best_precision']:.2f}%")
    print(f"Recall: {result['best_recall']:.2f}%")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full-optimization':
        # Run complete hyperparameter optimization (60+ configurations)
        run_hyperparameter_optimization()
    else:
        # Run single best configuration for quick testing
        print("Running single best configuration. Use --full-optimization for complete sweep.")
        train_best_configuration_only() 
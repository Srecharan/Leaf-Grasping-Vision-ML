#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.ml_grasp_optimizer.model import GraspPointCNN

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


def train_grasp_model():
    # Setup device and seed
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    data_path = os.path.expanduser('~/leaf_grasp_output/ml_training_data/training_data.pt')
    data = torch.load(data_path)
    print(f"Loaded {len(data['labels'])} samples")
    
    # Data Analysis and Normalization
    depth_patches = data['depth_patches'].to(device).unsqueeze(1)
    mask_patches = data['mask_patches'].to(device).unsqueeze(1)
    score_patches = data['score_patches'].to(device)
    labels = data['labels'].float().to(device)
    
    # Normalize data
    normalized = normalize_data(depth_patches, score_patches)
    depth_patches = normalized['depth_patches']
    score_patches = normalized['score_patches']
    
    # Prepare features
    features = torch.cat([depth_patches, mask_patches, score_patches], dim=1)
    print(f"Combined features shape: {features.shape}")
    
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
    
    # Create data loaders with smaller batch size
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        sampler=sampler
    )
    
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # Create model with higher pos_weight
    model = GraspPointCNN(in_channels=9).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Add early stopping with best weights restoration
    early_stopping = EarlyStopping(patience=15, min_delta=0.001, restore_best_weights=True)
    
    # Training loop
    num_epochs = 150  # Increased epochs
    train_losses = []
    val_losses = []
    metrics_history = []
    best_val_loss = float('inf')
    
    print("\n=== Starting Training ===")
    
    for epoch in range(num_epochs):
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
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Positive Accuracy: {metrics["positive_accuracy"]:.2f}%')
        print(f'Negative Accuracy: {metrics["negative_accuracy"]:.2f}%')
        print(f'F1 Score: {metrics["f1_score"]:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = os.path.expanduser('~/leaf_grasp_output/ml_models')
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': metrics,
                'normalization_stats': normalized['stats'],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'metrics_history': metrics_history
            }, save_path)
            print(f"\nSaved best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping check
        if early_stopping.step(avg_val_loss, epoch, model):
            print(f"\nEarly stopping triggered! Best epoch was {early_stopping.best_epoch + 1}")
            break
    
    # Plot and save metrics
    save_dir = os.path.expanduser('~/leaf_grasp_output/ml_models')
    plot_metrics(train_losses, val_losses, metrics_history, save_dir)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'metrics': metrics,
        'normalization_stats': normalized['stats'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics_history': metrics_history
    }, final_model_path)
    
    # Final summary
    print("\n=== Training Summary ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final metrics:")
    print("Class-wise Accuracy:")
    print(f"  Positive: {metrics['positive_accuracy']:.2f}%")
    print(f"  Negative: {metrics['negative_accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Positives: {metrics['confusion_matrix']['true_positive']}")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positive']}")
    print(f"  True Negatives: {metrics['confusion_matrix']['true_negative']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negative']}")
    
    print(f"\nBest model saved to: {save_dir}/best_model.pth")
    print(f"Final model saved to: {save_dir}/final_model.pth")
    print(f"Training metrics plot saved to: {save_dir}/training_metrics.png")
    
    # Save training configuration
    config_path = os.path.join(save_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Total epochs: {epoch + 1}\n")
        f.write(f"Best epoch: {early_stopping.best_epoch + 1}\n")
        f.write(f"Initial learning rate: 0.0005\n")
        f.write(f"Batch size: 16\n")
        f.write(f"Positive weight: 2.0\n")
        f.write(f"Early stopping patience: 15\n")
        f.write(f"Weight decay: 0.01\n")
        f.write("\n=== Dataset Statistics ===\n")
        f.write(f"Total samples: {num_samples}\n")
        f.write(f"Training samples: {train_size}\n")
        f.write(f"Validation samples: {num_samples - train_size}\n")
        f.write(f"Positive samples: {(labels == 1).sum().item()}\n")
        f.write(f"Negative samples: {(labels == 0).sum().item()}\n")

if __name__ == '__main__':
    try:
        train_grasp_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())
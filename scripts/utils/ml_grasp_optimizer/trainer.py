import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import rospy
from .model import GraspPointCNN

class GraspModelTrainer:
    def __init__(self, device):
        self.device = device
        self.model = GraspPointCNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))  # Weight positive class more
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create directory for saving models
        self.model_dir = os.path.expanduser('~/leaf_grasp_output/ml_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def prepare_data(self, data_dict):
        """Prepare data for training"""
        # Concatenate all features
        features = torch.cat([
            data_dict['rgb_patches'],
            data_dict['depth_patches'].unsqueeze(1),
            data_dict['mask_patches'].unsqueeze(1),
            data_dict['score_patches']
        ], dim=1)
        
        # Create dataset
        dataset = TensorDataset(features, data_dict['labels'].float())
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=50):
        """Train the model"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs.squeeze(), batch_labels)
                    val_loss += loss.item()
                    
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            rospy.loginfo(f'Learning Rate: {current_lr:.6f}')

            # Print progress
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            rospy.loginfo(f'Epoch [{epoch+1}/{epochs}]')
            rospy.loginfo(f'Train Loss: {avg_train_loss:.4f}')
            rospy.loginfo(f'Val Loss: {avg_val_loss:.4f}')
            rospy.loginfo(f'Accuracy: {accuracy:.2f}%')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('best_model.pth')
    
    def save_model(self, filename):
        """Save model weights"""
        save_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
        rospy.loginfo(f'Saved model to {save_path}')
    
    def load_model(self, filename):
        """Load model weights"""
        load_path = os.path.join(self.model_dir, filename)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rospy.loginfo(f'Loaded model from {load_path}')
            return True
        return False
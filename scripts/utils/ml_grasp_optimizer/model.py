import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspPointCNN(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        
        # Feature extraction layers with increased capacity
        self.encoder = nn.ModuleList([
            # First block: 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3)
            ),
            
            # Second block: 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3)
            ),
            
            # Third block: 8x8 -> 4x4
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3)
            )
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with increased capacity
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(64, 1)  # Single output for binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction through encoder blocks
        for block in self.encoder:
            x = block(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
class GraspQualityPredictor:
    def __init__(self, device):
        self.device = device
        self.model = GraspPointCNN().to(device)
        
    def predict(self, patch_data):
        """
        Predict grasp quality for a patch
        Args:
            patch_data: Dictionary containing:
                - rgb_patch: (32,32,3)
                - depth_patch: (32,32,1)
                - mask_patch: (32,32,1)
                - score_patches: (32,32,7)
        Returns:
            Predicted grasp quality score (0-1)
        """
        self.model.eval()
        with torch.no_grad():
            # Concatenate all features
            features = torch.cat([
                patch_data['rgb_patch'],
                patch_data['depth_patch'],
                patch_data['mask_patch'],
                patch_data['score_patches']
            ], dim=1)
            
            # Add batch dimension
            features = features.unsqueeze(0)
            
            # Predict
            score = self.model(features)
            return score.item()
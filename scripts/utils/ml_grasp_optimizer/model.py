import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspPointCNN(nn.Module):
    def __init__(self, in_channels=9, attention_type='spatial', encoder_filters=[64, 128, 256]):
        super().__init__()
        
        self.attention_type = attention_type
        self.encoder_filters = encoder_filters
        
        # Build configurable encoder blocks
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        
        for i, filters in enumerate(encoder_filters):
            block = nn.Sequential(
                nn.Conv2d(current_channels, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3)
            )
            self.encoder.append(block)
            current_channels = filters
        
        # Configurable attention mechanism
        final_filters = encoder_filters[-1]
        if attention_type == 'spatial':
            self.attention = nn.Sequential(
                nn.Conv2d(final_filters, 1, kernel_size=1),
                nn.Sigmoid()
            )
        elif attention_type == 'channel':
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(final_filters, final_filters // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_filters // 16, final_filters, kernel_size=1),
                nn.Sigmoid()
            )
        elif attention_type == 'hybrid':
            # Spatial attention
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(final_filters, 1, kernel_size=1),
                nn.Sigmoid()
            )
            # Channel attention
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(final_filters, final_filters // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_filters // 16, final_filters, kernel_size=1),
                nn.Sigmoid()
            )
        else:  # 'none'
            self.attention = None
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with configurable capacity
        final_filters = encoder_filters[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_filters, final_filters),
            nn.BatchNorm1d(final_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(final_filters, final_filters // 2),
            nn.BatchNorm1d(final_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(final_filters // 2, final_filters // 4),
            nn.BatchNorm1d(final_filters // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(final_filters // 4, 1)  # Single output for binary classification
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
        
        # Apply attention based on type
        if self.attention_type == 'spatial':
            attention_weights = self.attention(x)
            x = x * attention_weights
        elif self.attention_type == 'channel':
            attention_weights = self.attention(x)
            x = x * attention_weights
        elif self.attention_type == 'hybrid':
            # Apply both spatial and channel attention
            spatial_weights = self.spatial_attention(x)
            channel_weights = self.channel_attention(x)
            x = x * spatial_weights * channel_weights
        # else: no attention (attention_type == 'none')
        
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
"""
Lightweight CNN model for chest X-ray classification.
Simple architecture with convolution and pooling layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for binary classification (Normal vs Pneumonia).
    
    Architecture:
    - Input: 3-channel images (RGB) or 1-channel (Grayscale)
    - Conv layers with ReLU activation
    - Max pooling for downsampling
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes=2, input_channels=1):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes (default: 2 for Normal/Pneumonia)
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Assuming input image size of 224x224, after 3 pooling layers: 224/8 = 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits for classification
        """
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Conv block 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(num_classes=2, input_channels=1, pretrained_path=None):
    """
    Factory function to create and optionally load a pretrained model.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        pretrained_path: Path to pretrained model weights (optional)
        
    Returns:
        Initialized model
    """
    model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        model.eval()
    
    return model


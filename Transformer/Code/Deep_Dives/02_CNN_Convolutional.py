"""
02. Convolutional Neural Networks (CNN)
Deep Dive Implementation

This file demonstrates the explicit architecture of a CNN, highlighting 
the Convolution -> ReLU -> MaxPool pattern. We also calculate the 
flattened dimensions manually to bridge the gap between 2D Feature Maps 
and 1D Dense Layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """The Detective's Magnifying Glass Architecture."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # --- Feature Extraction (The Magnifying Glasses) ---
        
        # Conv Layer 1: Takes 1 input channel (Grayscale), applies 32 filters of size 3x3
        # Padding=1 ensures the output image size stays the same as input
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max Pooling 1: Reduces spatial dimensions by half (e.g. 28x28 -> 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: Takes 32 input feature maps, applies 64 filters of size 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Max Pooling 2: Reduces spatial dimensions by half again (e.g. 14x14 -> 7x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Classification (The Final Vote) ---
        
        # After two 2x2 pools on a 28x28 image, the spatial dimensions are 7x7.
        # We have 64 feature maps (channels). 
        # So flattened size = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Print shape tracking for educational purposes
        # Input x shape: [Batch, Channels, Height, Width] -> [B, 1, 28, 28]
        
        # Layer 1
        x = self.conv1(x)         # [B, 32, 28, 28]
        x = F.relu(x)             # [B, 32, 28, 28]
        x = self.pool1(x)         # [B, 32, 14, 14]
        
        # Layer 2
        x = self.conv2(x)         # [B, 64, 14, 14]
        x = F.relu(x)             # [B, 64, 14, 14]
        x = self.pool2(x)         # [B, 64, 7, 7]
        
        # Flatten the 2D spatial maps into a 1D vector for the Dense MLP
        # The -1 tells PyTorch to figure out the Batch dimension automatically
        x = x.view(-1, 64 * 7 * 7) # [B, 3136]
        
        # Fully Connected Classifier
        x = self.fc1(x)           # [B, 128]
        x = F.relu(x)             # [B, 128]
        logits = self.fc2(x)      # [B, 10]
        
        return logits

if __name__ == "__main__":
    print("--- 02. Convolutional Neural Network ---")
    
    # Simulate a batch of 4 Grayscale images of size 28x28 (e.g., MNIST dataset)
    batch_size = 4
    channels = 1
    height = 28
    width = 28
    
    # Create the network
    cnn = CNN(num_classes=10)
    
    # Generate dummy image data: Shape [Batch, Channels, Height, Width]
    image_batch = torch.randn(batch_size, channels, height, width)
    
    # Forward Pass
    predictions = cnn(image_batch)
    
    print(f"Input Image Shape: {image_batch.shape} [Batch, Channels, Height, Width]")
    print(f"Output Predictions Shape: {predictions.shape} [Batch, Classes]")
    print("Success: CNN Forward Pass complete, spatial dimensions flattened correctly.")

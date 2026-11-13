"""
01. Multi-Layer Perceptron (MLP) & Feedforward Neural Network
Deep Dive Implementation

This file explicitly demonstrates the forward pass and mathematics of an MLP.
Instead of just using `nn.Linear`, we implement the explicit matrix multiplication
y = x @ W^T + b to match the mathematical equations in the report.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitLinearLayer(nn.Module):
    """An explicit mathematical representation of a single Dense layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize Weights (W) and Biases (b) explicitly
        # We use kaiming/he initialization scaled manually
        stdv = 1. / (in_features ** 0.5)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # The core equation: Z = XW^T + b
        # F.linear(x, self.weight, self.bias) does exactly: x @ self.weight.T + self.bias
        # We write it out explicitly for educational purposes:
        return torch.matmul(x, self.weight.t()) + self.bias

class DeepMLP(nn.Module):
    """A deep feedforward network acting as our 'Assembly Line'."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Input Layer -> Hidden Layer 1
        self.layer1 = ExplicitLinearLayer(input_dim, hidden_dim)
        
        # Hidden Layer 1 -> Hidden Layer 2
        self.layer2 = ExplicitLinearLayer(hidden_dim, hidden_dim)
        
        # Hidden Layer 2 -> Output Layer
        self.layer3 = ExplicitLinearLayer(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass tracking the activations (A)
        # Z^[1] = X @ W^[1]T + b^[1]
        z1 = self.layer1(x)
        # A^[1] = ReLU(Z^[1])
        a1 = F.relu(z1)
        
        # Z^[2] = A^[1] @ W^[2]T + b^[2]
        z2 = self.layer2(a1)
        # A^[2] = ReLU(Z^[2])
        a2 = F.relu(z2)
        
        # Z^[3] = A^[2] @ W^[3]T + b^[3]
        logits = self.layer3(a2)
        
        # Output is Raw Logits (We apply Sigmoid/Softmax in the loss function)
        return logits

if __name__ == "__main__":
    print("--- 01. MLP & Feedforward Network ---")
    # Simulate a batch of 8 images, each 28x28 pixels flattened (784)
    batch_size = 8
    input_features = 784 
    hidden_units = 128
    num_classes = 10 # E.g., classifying digits 0-9
    
    # Create the network
    mlp = DeepMLP(input_features, hidden_units, num_classes)
    
    # Generate dummy data
    X_dummy = torch.randn(batch_size, input_features)
    
    # Forward Pass
    predictions = mlp(X_dummy)
    
    print(f"Input Shape: {X_dummy.shape} [Batch Size, Features]")
    print(f"Output Shape: {predictions.shape} [Batch Size, Classes]")
    
    # To get final probabilities, we apply Softmax
    probabilities = F.softmax(predictions, dim=1)
    print(f"Sample probabilities for Image 0: \n{probabilities[0].detach().numpy()}")
    print("Success: Explicit MLP Forward Pass complete.")

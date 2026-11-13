"""
20. Kolmogorov-Arnold Networks (KAN)
Deep Dive Implementation

This file explicitly demonstrates the difference between an MLP and a KAN.
Instead of `Weight Matrix * Input + Bias`, the KAN layer applies a learnable 
mathematical function (we approximate B-splines using radial basis functions for simplicity)
directly on the edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedKANLayer(nn.Module):
    """
    A simplified KAN layer.
    Instead of full B-splines (which require complex piecewise polynomial math),
    we use a sum of Gaussian Radial Basis Functions (RBFs) to approximate
    learnable, smooth, non-linear curves on the edges.
    """
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # In a KAN, we don't have a single weight W.
        # We have a set of coefficients for our curves on EVERY edge.
        # Total curves = in_features * out_features.
        # Each curve needs `grid_size` control points.
        self.curve_coefficients = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) / grid_size
        )
        
        # The grid centers for our Gaussian bumps
        # We spread them evenly between -1 and 1
        self.grid = nn.Parameter(
            torch.linspace(-1, 1, grid_size), requires_grad=False
        )
        
        # A standard bias term on the output node
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """
        x shape: [Batch, In_Features]
        """
        batch_size = x.size(0)
        
        # 1. Expand input to interact with our grid
        # x_expanded: [Batch, In_Features, 1]
        x_expanded = x.unsqueeze(-1)
        
        # 2. Calculate the Basis Functions (Gaussian bumps)
        # For every input value, how close is it to our grid centers?
        # basis: [Batch, In_Features, Grid_Size]
        sigma = 2.0 / self.grid_size # Width of the bump
        basis = torch.exp(-((x_expanded - self.grid) ** 2) / (sigma ** 2))
        
        # 3. Apply the learnable coefficients to warp the curves
        # This is where the "edge activation" happens!
        # We multiply the basis functions by our learnable curve coefficients.
        # We use einsum to do this efficiently across all in/out combinations.
        # Output is the value from evaluating the specific mathematical curve for each edge.
        # shape: [Batch, Out_Features, In_Features]
        edge_activations = torch.einsum('big,oig->boi', basis, self.curve_coefficients)
        
        # 4. Sum up the edges at the output nodes
        # The node itself does NO activation function. It just sums.
        # shape: [Batch, Out_Features]
        node_sums = torch.sum(edge_activations, dim=-1) + self.bias
        
        return node_sums

class DeepKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Note: No nn.ReLU() or nn.Sigmoid() between layers!
        # The non-linearity is already handled by the splines inside the KAN Layer.
        self.layer1 = SimplifiedKANLayer(input_dim, hidden_dim)
        self.layer2 = SimplifiedKANLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    print("--- 20. Kolmogorov-Arnold Network (KAN) ---")
    
    batch_size = 4
    input_dim = 2
    hidden_dim = 5
    output_dim = 1
    
    # Create the KAN
    kan = DeepKAN(input_dim, hidden_dim, output_dim)
    
    # Dummy data
    x_input = torch.randn(batch_size, input_dim)
    
    # Forward Pass
    print("Evaluating learnable splines on the network edges...")
    predictions = kan(x_input)
    
    print(f"Input Shape: {x_input.shape}")
    print(f"Predictions Shape: {predictions.shape}")
    print("\nNotice there are ZERO static weights (W) and ZERO activation functions (ReLU) used in this network!")
    print("The non-linearity was completely handled by warping the curve coefficients on the edges.")

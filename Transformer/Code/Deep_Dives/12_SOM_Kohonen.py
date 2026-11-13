"""
12. Self-Organizing Maps (SOM) / Kohonen Networks
Deep Dive Implementation

This file explicitly demonstrates unsupervised Competitive Learning.
There is no backpropagation. We calculate the Euclidean distance to find the Best 
Matching Unit (BMU), calculate the 2D neighborhood, and pull the weights toward the data.
"""

import torch
import torch.nn as nn

class ExplicitSOM(nn.Module):
    """A 2D Self-Organizing Map."""
    def __init__(self, map_size=(10, 10), input_dim=3):
        super().__init__()
        self.map_size = map_size
        self.input_dim = input_dim
        
        self.num_nodes = map_size[0] * map_size[1]
        
        # Initialize Weights randomly
        # Shape: [Num_Nodes, Input_Dim]
        self.weights = nn.Parameter(torch.randn(self.num_nodes, input_dim), requires_grad=False)
        
        # Create a grid of physical 2D coordinates for the nodes
        # e.g., Node 0 is at (0,0), Node 1 is at (0,1), etc.
        x_coords, y_coords = torch.meshgrid(
            torch.arange(map_size[0]), 
            torch.arange(map_size[1]), 
            indexing='ij'
        )
        self.locations = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()

    def find_bmu(self, x):
        """Find the Best Matching Unit (the node whose weight is closest to input x)"""
        # Expand x to match weight matrix: [1, Input_Dim] -> [Num_Nodes, Input_Dim]
        x_expanded = x.expand(self.num_nodes, -1)
        
        # Calculate Euclidean Distance between x and ALL weights
        distances = torch.sum((x_expanded - self.weights) ** 2, dim=1)
        
        # Find the index of the minimum distance
        bmu_index = torch.argmin(distances)
        return bmu_index

    def update_weights(self, x, bmu_index, current_radius, learning_rate):
        """Pull the BMU and its neighbors toward x."""
        # 1. Get the physical 2D location of the BMU
        bmu_loc = self.locations[bmu_index]
        
        # 2. Calculate physical distance from BMU to EVERY other node on the 2D grid
        # Shape: [Num_Nodes]
        dist_to_bmu_squared = torch.sum((self.locations - bmu_loc) ** 2, dim=1)
        
        # 3. Calculate the Neighborhood Function Theta (Gaussian)
        # Nodes closer to the BMU get a value close to 1. Nodes far away get ~0.
        theta = torch.exp(-dist_to_bmu_squared / (2 * (current_radius ** 2)))
        
        # Expand theta to match weight dimensions: [Num_Nodes] -> [Num_Nodes, 1]
        theta = theta.unsqueeze(1)
        
        # 4. The Kohonen Update Rule: W_new = W_old + Theta * LR * (x - W_old)
        # Difference between data point and current weights
        weight_delta = x.unsqueeze(0) - self.weights 
        
        # Update weights (in-place)
        self.weights += theta * learning_rate * weight_delta


if __name__ == "__main__":
    print("--- 12. Self-Organizing Map (SOM) ---")
    
    # E.g., Mapping 3D RGB Colors to a 10x10 2D Grid
    input_dim = 3
    map_size = (10, 10)
    som = ExplicitSOM(map_size=map_size, input_dim=input_dim)
    
    # A single training step with a dummy data point (e.g. the color Red: [1, 0, 0])
    data_point = torch.tensor([1.0, 0.0, 0.0])
    
    # Training Parameters
    initial_radius = max(map_size) / 2.0
    learning_rate = 0.5
    
    print("Finding Best Matching Unit (BMU)...")
    bmu_idx = som.find_bmu(data_point)
    bmu_location = som.locations[bmu_idx]
    print(f"BMU found at 2D grid coordinate: ({int(bmu_location[0])}, {int(bmu_location[1])})")
    
    print("Pulling magnetic mesh toward data point...")
    som.update_weights(data_point, bmu_idx, initial_radius, learning_rate)
    
    print("Success: Unsupervised Competitive Learning applied successfully.")

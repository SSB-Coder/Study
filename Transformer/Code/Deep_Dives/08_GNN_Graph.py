"""
08. Graph Neural Networks (GNN)
Deep Dive Implementation

This file explicitly demonstrates a Graph Convolutional Network (GCN) layer.
We manually define the Adjacency Matrix (A), the Degree Matrix (D), 
and perform the normalized Message Passing equation:
H_next = sigma( D^{-0.5} * A_tilde * D^{-0.5} * H_prev * W )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitGCNLayer(nn.Module):
    """A single layer of Graph Message Passing."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # The actual Neural Network part (W)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, H_prev, A_normalized):
        """
        H_prev: The feature matrix of all nodes [Num_Nodes, In_Features]
        A_normalized: The mathematically normalized Adjacency Matrix D^{-0.5} A_tilde D^{-0.5}
        """
        # Step 1: Feature Transformation (H * W)
        # Apply the neural network weights to every node's features
        support = torch.mm(H_prev, self.weight)
        
        # Step 2: Message Passing / Aggregation (A_norm * Support)
        # Matrix multiplication with the adjacency matrix inherently sums up the features 
        # of all neighboring nodes!
        output = torch.mm(A_normalized, support)
        
        return output

def calculate_normalized_adjacency(A):
    """
    Calculates D^{-0.5} A_tilde D^{-0.5}
    """
    num_nodes = A.size(0)
    
    # 1. Add Self-Loops (A_tilde = A + I)
    # This ensures a node doesn't forget its own features when aggregating from friends
    I = torch.eye(num_nodes)
    A_tilde = A + I
    
    # 2. Calculate Degree Matrix (D)
    # Sum across the rows to see how many edges each node has
    degrees = A_tilde.sum(dim=1)
    
    # 3. Calculate D^{-0.5}
    # We raise degrees to -0.5. We use clamp to avoid division by zero.
    D_inv_sqrt = torch.pow(degrees.clamp(min=1e-12), -0.5)
    
    # Create a diagonal matrix out of the D_inv_sqrt vector
    D_inv_sqrt_diag = torch.diag(D_inv_sqrt)
    
    # 4. Final Matrix Multiplication
    # norm_A = D^{-0.5} @ A_tilde @ D^{-0.5}
    norm_A = torch.mm(torch.mm(D_inv_sqrt_diag, A_tilde), D_inv_sqrt_diag)
    
    return norm_A


class DeepGCN(nn.Module):
    """A 2-Layer GCN."""
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super().__init__()
        self.gc1 = ExplicitGCNLayer(num_node_features, hidden_dim)
        self.gc2 = ExplicitGCNLayer(hidden_dim, num_classes)

    def forward(self, x, A_norm):
        # Layer 1 Message Passing
        x = self.gc1(x, A_norm)
        x = F.relu(x)
        
        # Layer 2 Message Passing
        x = self.gc2(x, A_norm)
        
        # Final Softmax to classify the nodes
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    print("--- 08. Graph Convolutional Network (GCN) ---")
    
    # Mock Graph: 5 Nodes (Users in a social network)
    num_nodes = 5
    in_features = 3 # E.g., [Age, AccountBalance, TimeOnline]
    
    # Feature Matrix H (Each row is a user's features)
    H = torch.randn(num_nodes, in_features)
    
    # Adjacency Matrix A (Who is friends with who?)
    # 1 means edge, 0 means no edge. Symmetric matrix for undirected graphs.
    A = torch.tensor([
        [0, 1, 0, 0, 1], # Node 0 is connected to 1 and 4
        [1, 0, 1, 1, 0], # Node 1 is connected to 0, 2, 3
        [0, 1, 0, 1, 0], # Node 2 is connected to 1, 3
        [0, 1, 1, 0, 0], # Node 3 is connected to 1, 2
        [1, 0, 0, 0, 0]  # Node 4 is only connected to 0
    ], dtype=torch.float32)
    
    print("Calculating Normalized Adjacency Matrix...")
    A_norm = calculate_normalized_adjacency(A)
    
    # Create GCN
    hidden_features = 8
    num_classes = 2 # E.g., Fraudulent vs Legit User
    gcn = DeepGCN(in_features, hidden_features, num_classes)
    
    # Forward Pass
    print("Executing Message Passing across the graph...")
    predictions = gcn(H, A_norm)
    
    print(f"Input Features Shape: {H.shape} [Nodes, Features]")
    print(f"Output Predictions Shape: {predictions.shape} [Nodes, Classes]")
    print(f"Class probabilities for Node 0:\n{torch.exp(predictions[0]).detach().numpy()}")
    print("Success: Graph Message Passing mathematics complete.")

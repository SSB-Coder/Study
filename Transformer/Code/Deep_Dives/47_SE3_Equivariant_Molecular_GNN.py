"""
47. E(3) / SE(3) Equivariant Graph Neural Networks (EGNN)
Deep Dive Implementation: Molecular Dynamics & Physical Symmetry Conservation

This file explicitly demonstrates Equivariant Graph Neural Networks for chemistry and materials science.
We prove:
1. Rotational Invariance: Molecular Energy E(R * x + t) == E(x)
2. Rotational Equivariance: Atomic Coordinates/Forces x_out(R * x + t) == R * x_out(x) + t
"""

import torch
import torch.nn as nn
import math

class EGNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network Layer (Satorras et al., 2021).
    Takes node scalar features h_i in R^d and 3D geometric coordinates x_i in R^3.
    """
    def __init__(self, node_dim=16, edge_dim=0, hidden_dim=32):
        super().__init__()
        self.node_dim = node_dim
        
        # Edge Message MLP: inputs [h_i, h_j, d_ij^2]
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate Update MLP: computes scalar weighting for directional vector (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Node Scalar Feature Update MLP: inputs [h_i, aggregated_messages]
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, h, x, edge_index):
        """
        h: Node scalar features [Num_Nodes, Node_Dim]
        x: Continuous 3D coordinates [Num_Nodes, 3]
        edge_index: Graph connectivity [2, Num_Edges] (row i -> col j)
        """
        row, col = edge_index[0], edge_index[1]
        
        # 1. Compute pairwise difference vectors and rotation-invariant squared distances
        # diff_x: [Num_Edges, 3]
        diff_x = x[row] - x[col]
        # dist_sq: [Num_Edges, 1] (Strictly invariant to 3D rotation and translation)
        dist_sq = torch.sum(diff_x ** 2, dim=-1, keepdim=True)
        
        # 2. Compute Edge Messages
        # edge_input: [Num_Edges, node_dim * 2 + 1]
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input) # [Num_Edges, hidden_dim]
        
        # 3. Equivariant Coordinate Update
        # coord_weights: scalar factor for each edge [Num_Edges, 1]
        coord_weights = self.coord_mlp(m_ij)
        # Normalize difference vector to prevent numerical explosions when atoms are close
        norm_diff = diff_x / (torch.sqrt(dist_sq) + 1e-6)
        # Scaled displacement: [Num_Edges, 3]
        coord_trans = norm_diff * coord_weights
        
        # Aggregate coordinate updates onto node i
        num_nodes = h.size(0)
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_trans)
        # Updated coordinates: [Num_Nodes, 3] (Strictly Equivariant!)
        x_new = x + x_update
        
        # 4. Invariant Node Feature Update
        # Aggregate messages onto node i
        m_i = torch.zeros(num_nodes, m_ij.size(-1), device=h.device)
        m_i.index_add_(0, row, m_ij)
        
        node_input = torch.cat([h, m_i], dim=-1)
        h_new = h + self.node_mlp(node_input) # Residual connection
        
        return h_new, x_new


class EquivariantMolecularGNN(nn.Module):
    """
    Full Molecular Neural Network predicting:
    1. Invariant Total Molecular Potential Energy (scalar E)
    2. Equivariant Relaxed 3D Molecular Conformation (coordinates x)
    """
    def __init__(self, node_dim=16, hidden_dim=32, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim=node_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        # Molecular Energy Readout Head (sums invariant atomic representations)
        self.energy_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, x, edge_index):
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
            
        # Invariant energy: sum pooling over all atoms in the molecule
        atomic_energies = self.energy_head(h) # [Num_Nodes, 1]
        total_energy = torch.sum(atomic_energies, dim=0) # [1]
        
        return total_energy, x


def generate_random_3d_rotation():
    """Generates a random 3D orthogonal rotation matrix R in SO(3)."""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


if __name__ == "__main__":
    print("--- 47. Equivariant Graph Neural Network (EGNN) ---")
    print("Testing Mathematical Invariance & Equivariance under 3D Rotations & Translations...\n")

    # 1. Construct a mock 5-atom molecule (e.g. Methane CH4)
    # Node features: atomic embeddings (carbon + 4 hydrogens)
    num_atoms = 5
    node_dim = 16
    h_atoms = torch.randn(num_atoms, node_dim)
    
    # 3D spatial coordinates [5, 3]
    x_coords = torch.tensor([
        [0.0, 0.0, 0.0],       # Carbon center
        [0.63, 0.63, 0.63],    # H1
        [-0.63, -0.63, 0.63],  # H2
        [-0.63, 0.63, -0.63],  # H3
        [0.63, -0.63, -0.63]   # H4
    ], dtype=torch.float32)

    # Fully connected graph for 5 atoms (all pairwise interactions)
    row = []
    col = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)

    # Instantiate EGNN
    model = EquivariantMolecularGNN(node_dim=node_dim, hidden_dim=32, num_layers=2)
    model.eval()

    # 2. Baseline Forward Pass
    energy_orig, x_out_orig = model(h_atoms, x_coords, edge_index)
    print(f"Original Molecular Energy: {energy_orig.item():.6f}")

    # 3. Apply a random 3D Rotation R and Translation T in space
    R = generate_random_3d_rotation()
    T = torch.tensor([[10.5, -3.2, 7.8]])
    
    # Transformed coordinates: x_rot = x * R^T + T
    x_transformed = torch.matmul(x_coords, R.T) + T

    # 4. Transformed Forward Pass
    energy_trans, x_out_trans = model(h_atoms, x_transformed, edge_index)
    print(f"Transformed (Rotated + Translated) Energy: {energy_trans.item():.6f}")

    # 5. Rigorous Symmetry Checks
    energy_diff = torch.abs(energy_orig - energy_trans).item()
    print(f"\n[Test 1] Invariant Energy Difference |E(X) - E(X_rot)|: {energy_diff:.8f}")
    assert energy_diff < 1e-5, "Failed: Energy must be invariant!"
    print(">>> PASS: Molecular Energy is mathematically invariant to 3D transformations.")

    # Expected transformed coordinates from original output: x_expected = x_out_orig * R^T + T
    x_out_expected = torch.matmul(x_out_orig, R.T) + T
    coord_error = torch.max(torch.abs(x_out_trans - x_out_expected)).item()
    print(f"[Test 2] Equivariant Coordinate Max Error: {coord_error:.8f}")
    assert coord_error < 1e-5, "Failed: Coordinates must be equivariant!"
    print(">>> PASS: Updated Atomic Positions rotate and translate with exact physical equivariance.")

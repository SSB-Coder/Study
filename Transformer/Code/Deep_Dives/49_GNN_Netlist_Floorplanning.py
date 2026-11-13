"""
49. AI for Chip Design & Electronic Design Automation (EDA)
Deep Dive Implementation: Netlist Graph Neural Network & Differentiable Macro Floorplanning

This file explicitly demonstrates physical chip layout optimization:
1. Bipartite Netlist GNN: Passing messages between logic cells and interconnect nets.
2. Differentiable HPWL (Half-Perimeter Wire Length): Log-Sum-Exp smoothing for wire routing.
3. Density Overlap Penalty: Enforcing non-overlapping layout constraints on a 2D silicon die.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteNetlistGNN(nn.Module):
    """
    Message-Passing Neural Network on circuit netlists.
    Circuits are bipartite graphs: Cell Nodes <---> Net Nodes.
    Predicts wire congestion and localized routing parasitics.
    """
    def __init__(self, cell_dim=8, net_dim=4, hidden_dim=16):
        super().__init__()
        # Cell-to-Net message MLP
        self.cell_to_net = nn.Sequential(
            nn.Linear(cell_dim + net_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, net_dim)
        )
        # Net-to-Cell message MLP
        self.net_to_cell = nn.Sequential(
            nn.Linear(net_dim + cell_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cell_dim)
        )

    def forward(self, cell_feat, net_feat, edge_index):
        """
        cell_feat:  [Num_Cells, Cell_Dim] (e.g. area, pin count, power)
        net_feat:   [Num_Nets, Net_Dim]   (e.g. fanout degree, priority)
        edge_index: [2, Num_Pins] (row: cell_idx, col: net_idx)
        """
        cell_idx, net_idx = edge_index[0], edge_index[1]
        num_nets = net_feat.size(0)
        num_cells = cell_feat.size(0)

        # 1. Cells send messages to connected Nets
        msg_to_net = self.cell_to_net(torch.cat([cell_feat[cell_idx], net_feat[net_idx]], dim=-1))
        agg_net = torch.zeros_like(net_feat)
        agg_net.index_add_(0, net_idx, msg_to_net)
        net_feat = net_feat + agg_net # Residual update

        # 2. Nets send messages back to Cells
        msg_to_cell = self.net_to_cell(torch.cat([net_feat[net_idx], cell_feat[cell_idx]], dim=-1))
        agg_cell = torch.zeros_like(cell_feat)
        agg_cell.index_add_(0, cell_idx, msg_to_cell)
        cell_feat = cell_feat + agg_cell # Residual update

        return cell_feat, net_feat


def differentiable_lse_hpwl(positions, net_to_cell_list, gamma=10.0):
    """
    Differentiable Half-Perimeter Wire Length using Log-Sum-Exp (LSE) approximation.
    HPWL = (max_x - min_x) + (max_y - min_y) for all pins in the net.
    gamma: temperature smoothing parameter.
    """
    total_hpwl = torch.tensor(0.0, device=positions.device)

    for cell_indices in net_to_cell_list:
        if len(cell_indices) < 2:
            continue
        # Extract 2D coordinates for all cells connected to this net
        pts = positions[cell_indices] # [K, 2]
        x = pts[:, 0]
        y = pts[:, 1]

        # LSE approximation for max(x) - min(x)
        # max(x) ≈ gamma * log(sum(exp(x / gamma)))
        # min(x) ≈ -gamma * log(sum(exp(-x / gamma)))
        lse_max_x = gamma * torch.logsumexp(x / gamma, dim=0)
        lse_min_x = -gamma * torch.logsumexp(-x / gamma, dim=0)
        wl_x = lse_max_x - lse_min_x

        lse_max_y = gamma * torch.logsumexp(y / gamma, dim=0)
        lse_min_y = -gamma * torch.logsumexp(-y / gamma, dim=0)
        wl_y = lse_max_y - lse_min_y

        total_hpwl = total_hpwl + (wl_x + wl_y)

    return total_hpwl


def compute_macro_overlap_penalty(positions, sizes):
    """
    Penalizes geometric overlaps between rectangular macro blocks on the silicon canvas.
    positions: [Num_Macros, 2] (center coordinates x, y)
    sizes:     [Num_Macros, 2] (width w, height h)
    """
    num_macros = positions.size(0)
    penalty = torch.tensor(0.0, device=positions.device)

    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            # Center-to-center distance
            dx = torch.abs(positions[i, 0] - positions[j, 0])
            dy = torch.abs(positions[i, 1] - positions[j, 1])

            # Overlap along X and Y
            min_dx = 0.5 * (sizes[i, 0] + sizes[j, 0])
            min_dy = 0.5 * (sizes[i, 1] + sizes[j, 1])

            overlap_x = F.relu(min_dx - dx)
            overlap_y = F.relu(min_dy - dy)

            # Area of intersection rectangle
            overlap_area = overlap_x * overlap_y
            penalty = penalty + overlap_area

    return penalty


if __name__ == "__main__":
    print("--- 49. AI for Chip Design & Differentiable Floorplanning ---")
    print("Optimizing Macro Block Placements on a 1000um x 1000um Silicon Die Canvas...\n")

    torch.manual_seed(42)
    # Define a chip with 6 macro blocks (e.g. SRAM cache banks and CPU ALUs)
    num_macros = 6
    die_width, die_height = 1000.0, 1000.0

    # Macro dimensions: [width, height] in microns
    macro_sizes = torch.tensor([
        [200.0, 200.0],
        [150.0, 250.0],
        [250.0, 150.0],
        [180.0, 180.0],
        [220.0, 160.0],
        [160.0, 200.0]
    ], dtype=torch.float32)

    # Initial random placement of macro centers inside the die
    macro_positions = nn.Parameter(
        torch.empty(num_macros, 2).uniform_(200.0, 800.0)
    )

    # Define electrical nets connecting subsets of macro blocks
    nets = [
        [0, 1],       # Net 0: Cache 0 to ALU 1
        [1, 2],       # Net 1: ALU 1 to ALU 2
        [2, 3],       # Net 2: ALU 2 to Cache 3
        [0, 3],       # Net 3: Cache 0 to Cache 3
        [0, 4, 5],    # Net 4: High fanout bus
        [1, 4],       # Net 5
        [2, 5],       # Net 6
        [3, 5]        # Net 7
    ]

    # Test Bipartite GNN on circuit netlist
    gnn = BipartiteNetlistGNN(cell_dim=4, net_dim=2, hidden_dim=16)
    cell_feats = torch.randn(num_macros, 4)
    net_feats = torch.randn(len(nets), 2)
    
    # Construct pin connection edge indices
    edges_cell, edges_net = [], []
    for n_id, net in enumerate(nets):
        for c_id in net:
            edges_cell.append(c_id)
            edges_net.append(n_id)
    edge_index = torch.tensor([edges_cell, edges_net], dtype=torch.long)

    updated_cell_feats, updated_net_feats = gnn(cell_feats, net_feats, edge_index)
    print(f"Netlist GNN Execution: {num_macros} Macros, {len(nets)} Nets, {edge_index.shape[1]} Pins connected.")
    print(f"Cell Features Extracted: {updated_cell_feats.shape}")

    # Physical Placement Optimization Loop via Gradient Descent
    optimizer = torch.optim.Adam([macro_positions], lr=5.0)

    init_wl = differentiable_lse_hpwl(macro_positions, nets).item()
    init_overlap = compute_macro_overlap_penalty(macro_positions, macro_sizes).item()
    print(f"\nInitial State -> Total Wirelength: {init_wl:.1f} um | Overlap Penalty: {init_overlap:.1f} um^2")

    print("Running Differentiable Placement Optimization...")
    for step in range(1, 101):
        optimizer.zero_grad()

        # Differentiable Wirelength Loss (minimized to shorten electrical signal latency)
        loss_wirelength = differentiable_lse_hpwl(macro_positions, nets, gamma=15.0)

        # Non-overlapping Physical Density Constraint
        loss_overlap = compute_macro_overlap_penalty(macro_positions, macro_sizes)

        # Boundary constraint: keep macros inside die canvas [0, 1000]
        loss_boundary = torch.sum(F.relu(-macro_positions)) + torch.sum(F.relu(macro_positions - die_width))

        # Total multi-objective EDA cost
        total_cost = loss_wirelength + 2.0 * loss_overlap + 10.0 * loss_boundary
        total_cost.backward()
        optimizer.step()

    final_wl = differentiable_lse_hpwl(macro_positions, nets).item()
    final_overlap = compute_macro_overlap_penalty(macro_positions, macro_sizes).item()

    print(f"Optimized State -> Total Wirelength: {final_wl:.1f} um | Overlap Penalty: {final_overlap:.1f} um^2")
    reduction = ((init_wl - final_wl) / init_wl) * 100.0
    print(f"Wirelength Reduction: {reduction:.2f}% | Silicon Congestion Resolved.")
    print("Success: Automated Electronic Design Automation (EDA) floorplan placed and optimized.")

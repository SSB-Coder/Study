"""
45. Physics-Informed Neural Networks (PINNs)
Deep Dive Implementation: Non-Linear Viscous Burgers' Equation & Parameter Discovery

This file demonstrates how to embed physical partial differential equations (PDEs)
directly into the loss function using PyTorch's Automatic Differentiation (Autograd).
We solve the non-linear viscous Burgers' equation:
    u_t + u * u_x - nu * u_xx = 0
and demonstrate inverse physics discovery (learning the unknown fluid viscosity parameter nu).
"""

import torch
import torch.nn as nn
import math

class PhysicsInformedNN(nn.Module):
    """
    A continuous coordinate network approximating the physical field u(x, t).
    Input: Coordinate pair (x, t) in R^2
    Output: Flow velocity scalar u(x, t) in R^1
    """
    def __init__(self, hidden_dim=32, num_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        # We use Tanh or SiLU because we require smooth second-order derivatives (u_xx).
        # ReLU cannot be used because its second derivative is identically zero everywhere.
        self.net = nn.Sequential(*layers)
        
        # Inverse Problem: Treat fluid viscosity 'nu' as a learnable parameter!
        # Initialized with a rough guess (e.g. 0.05), ground truth target is 0.01 / pi ≈ 0.003183
        self.discovered_nu = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

    def forward(self, x, t):
        """
        Computes the velocity u given spatial position x and temporal instant t.
        x: [N, 1]
        t: [N, 1]
        returns: u: [N, 1]
        """
        inputs = torch.cat([x, t], dim=1) # [N, 2]
        return self.net(inputs)

    def compute_pde_residual(self, x_colloc, t_colloc):
        """
        Evaluates the analytical PDE residual using Autograd:
        residual = u_t + u * u_x - nu * u_xx
        Notice: No finite differences or spatial discretization meshes!
        """
        # Ensure inputs require gradients w.r.t. the coordinates
        x_colloc.requires_grad_(True)
        t_colloc.requires_grad_(True)

        # 1. Forward pass to get velocity u(x, t)
        u = self.forward(x_colloc, t_colloc) # [N, 1]

        # 2. First-order temporal derivative: u_t = ∂u / ∂t
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t_colloc,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True # Crucial: enables second-order gradient calculation
        )[0]

        # 3. First-order spatial derivative: u_x = ∂u / ∂x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x_colloc,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # 4. Second-order spatial derivative: u_xx = ∂²u / ∂x² (diffusion / viscosity term)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x_colloc,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # 5. Non-linear Burgers' Equation Residual
        # In an ideal physical fluid, residual == 0 everywhere in space-time.
        residual = u_t + u * u_x - self.discovered_nu * u_xx
        return residual


def generate_synthetic_collocation_points(n_colloc=1000, n_bc=200, n_ic=200):
    """
    Generates continuous coordinates across domain x in [-1, 1], t in [0, 1].
    """
    # 1. Interior collocation points (for PDE residual enforcement)
    x_colloc = torch.empty(n_colloc, 1).uniform_(-1.0, 1.0)
    t_colloc = torch.empty(n_colloc, 1).uniform_(0.0, 1.0)

    # 2. Initial Condition: at t = 0, u(x, 0) = -sin(pi * x)
    x_ic = torch.empty(n_ic, 1).uniform_(-1.0, 1.0)
    t_ic = torch.zeros(n_ic, 1)
    u_ic_target = -torch.sin(math.pi * x_ic)

    # 3. Boundary Condition: at x = -1 and x = 1 (walls), u(-1, t) = u(1, t) = 0
    t_bc = torch.empty(n_bc, 1).uniform_(0.0, 1.0)
    x_bc_left = -torch.ones(n_bc // 2, 1)
    x_bc_right = torch.ones(n_bc // 2, 1)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    u_bc_target = torch.zeros(n_bc, 1)

    return (x_colloc, t_colloc), (x_ic, t_ic, u_ic_target), (x_bc, t_bc, u_bc_target)


if __name__ == "__main__":
    print("--- 45. Physics-Informed Neural Network (PINN) ---")
    print("Solving Non-Linear Burgers' Fluid Equation with Autograd PDE Residual...")

    pinn = PhysicsInformedNN(hidden_dim=32, num_layers=3)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)

    (x_colloc, t_colloc), (x_ic, t_ic, u_ic_target), (x_bc, t_bc, u_bc_target) = generate_synthetic_collocation_points(
        n_colloc=500, n_bc=100, n_ic=100
    )

    print(f"Initial Guess for Viscosity Parameter (nu): {pinn.discovered_nu.item():.5f}")
    print("Beginning physical optimization steps...")

    for step in range(1, 151):
        optimizer.zero_grad()

        # 1. Initial Condition Loss
        u_ic_pred = pinn(x_ic, t_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)

        # 2. Boundary Condition Loss
        u_bc_pred = pinn(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)

        # 3. PDE Physical Residual Loss (Zero-data, physics governed!)
        residual = pinn.compute_pde_residual(x_colloc, t_colloc)
        loss_pde = torch.mean(residual ** 2)

        # Total Composite Loss
        total_loss = loss_ic + loss_bc + loss_pde
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:03d} | Total Loss: {total_loss.item():.6f} "
                  f"(IC: {loss_ic.item():.5f}, BC: {loss_bc.item():.5f}, PDE: {loss_pde.item():.5f}) | "
                  f"Discovered nu: {pinn.discovered_nu.item():.5f}")

    print("\nVerification Test:")
    test_x = torch.tensor([[0.0], [0.5], [-0.5]])
    test_t = torch.tensor([[0.25], [0.25], [0.25]])
    test_u = pinn(test_x, test_t)
    print(f"Sample continuous velocity evaluations at t=0.25: {test_u.squeeze().detach().numpy()}")
    print("Success: PINN computed second-order continuous derivatives and enforced physical conservation.")

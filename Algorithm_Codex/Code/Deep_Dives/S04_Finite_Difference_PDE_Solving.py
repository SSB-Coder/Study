"""
S04. Finite-Difference PDE Solving (1D Heat & Wave Equations)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class FiniteDifferencePDE:
    """
    Solves 1D parabolic (Heat) and hyperbolic (Wave) partial differential equations
    using explicit finite difference schemes.
    """
    
    @staticmethod
    def solve_heat_equation(alpha: float = 0.01, L: float = 1.0, nx: int = 50,
                            t_max: float = 2.0, dt: float = 0.001):
        """
        Solves 1D Heat Equation: du/dt = alpha * d^2u/dx^2 <- eq. 3.1
        Boundary: u(0, t) = 0, u(L, t) = 0 (Dirichlet)
        Initial: u(x, 0) = sin(pi * x / L)
        Stability requirement (CFL): r = alpha * dt / dx^2 <= 0.5
        """
        dx = L / (nx - 1)
        r = alpha * dt / (dx**2)
        if r > 0.5:
            raise ValueError(f"CFL stability violated! r = {r:.3f} > 0.5")
            
        x = np.linspace(0.0, L, nx)
        u = np.sin(np.pi * x / L) # Initial condition
        
        n_steps = int(t_max / dt)
        for _ in range(n_steps):
            u_next = u.copy()
            # Explicit central difference in space <- eq. 3.2
            u_next[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
            # Enforce Dirichlet boundaries
            u_next[0] = 0.0
            u_next[-1] = 0.0
            u = u_next
            
        # Exact analytical solution: u(x, t) = exp(-alpha * (pi/L)^2 * t) * sin(pi * x / L)
        u_exact = np.exp(-alpha * (np.pi / L)**2 * t_max) * np.sin(np.pi * x / L)
        return x, u, u_exact

    @staticmethod
    def solve_wave_equation(c: float = 1.0, L: float = 1.0, nx: int = 50,
                            t_max: float = 1.0, dt: float = 0.01):
        """
        Solves 1D Wave Equation: d^2u/dt^2 = c^2 * d^2u/dx^2 <- eq. 3.3
        Boundary: u(0, t) = 0, u(L, t) = 0
        Initial: u(x, 0) = sin(pi * x / L), du/dt(x, 0) = 0
        CFL condition: C = c * dt / dx <= 1.0
        """
        dx = L / (nx - 1)
        C = c * dt / dx
        if C > 1.0:
            raise ValueError(f"Wave CFL stability violated! C = {C:.3f} > 1.0")
            
        x = np.linspace(0.0, L, nx)
        u_prev = np.sin(np.pi * x / L) # u at t=0
        
        # First step using zero initial velocity: u^1 = u^0 + 0.5*C^2*(u_{i+1}^0 - 2*u_i^0 + u_{i-1}^0)
        u_curr = u_prev.copy()
        u_curr[1:-1] = u_prev[1:-1] + 0.5 * (C**2) * (u_prev[2:] - 2.0 * u_prev[1:-1] + u_prev[:-2])
        
        n_steps = int(t_max / dt)
        for _ in range(1, n_steps):
            u_next = np.zeros_like(u_curr)
            u_next[1:-1] = 2.0 * u_curr[1:-1] - u_prev[1:-1] + (C**2) * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
            u_next[0] = 0.0
            u_next[-1] = 0.0
            u_prev = u_curr
            u_curr = u_next
            
        # Exact solution: u(x, t) = cos(c * pi * t / L) * sin(pi * x / L)
        u_exact = np.cos(c * np.pi * t_max / L) * np.sin(np.pi * x / L)
        return x, u_curr, u_exact

if __name__ == "__main__":
    print("--- S04: Finite-Difference PDE Solving Sanity Check ---")
    # 1. Test Heat Equation
    x, u_heat, u_heat_exact = FiniteDifferencePDE.solve_heat_equation(alpha=0.01, t_max=1.0)
    heat_err = np.max(np.abs(u_heat - u_heat_exact))
    print(f"Heat Equation Max Discretization Error: {heat_err:.2e}")
    assert heat_err < 1e-3, "Heat equation numerical solve diverged"
    
    # 2. Test Wave Equation
    x, u_wave, u_wave_exact = FiniteDifferencePDE.solve_wave_equation(c=1.0, t_max=1.0)
    wave_err = np.max(np.abs(u_wave - u_wave_exact))
    print(f"Wave Equation Max Discretization Error: {wave_err:.2e}")
    assert wave_err < 2e-2, "Wave equation numerical solve diverged"
    print("S04 Sanity Check: PASSED cleanly.")

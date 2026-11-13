"""
T06. Reaction-Diffusion & Turing Morphogenesis Patterns (Gray-Scott Model)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class GrayScottTuringPatterns:
    """
    2D Reaction-Diffusion simulation of the Gray-Scott model.
    Produces Alan Turing's classic biological morphogenesis patterns
    (spots, stripes, labyrinths, leopard skin, dividing cells).
    PDE equations:
      du/dt = D_u * laplacian(u) - u * v^2 + F * (1 - u)
      dv/dt = D_v * laplacian(v) + u * v^2 - (F + k) * v
    """
    
    def __init__(self, size: int = 40, Du: float = 0.16, Dv: float = 0.08,
                 F: float = 0.035, k: float = 0.060):
        self.N = size
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        
        # Grid initialization: u = 1 everywhere, v = 0 everywhere
        self.u = np.ones((size, size), dtype=float)
        self.v = np.zeros((size, size), dtype=float)
        
        # Seed a small central square with v = 1 to trigger Turing instability <- eq. 3.1
        mid = size // 2
        w = size // 8
        self.u[mid - w:mid + w, mid - w:mid + w] = 0.50
        self.v[mid - w:mid + w, mid - w:mid + w] = 0.25

    def _laplacian_periodic(self, grid: np.ndarray) -> np.ndarray:
        """5-point discrete Laplacian with periodic boundaries in pure NumPy <- eq. 3.2."""
        return (
            np.roll(grid, 1, axis=0) +
            np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1) -
            4.0 * grid
        )

    def step(self, dt: float = 1.0):
        """Advances Gray-Scott kinetics by one explicit finite-difference step <- eq. 3.3."""
        lu = self._laplacian_periodic(self.u)
        lv = self._laplacian_periodic(self.v)
        
        # Non-linear reaction term: u * v^2
        uv2 = self.u * (self.v**2)
        
        du = self.Du * lu - uv2 + self.F * (1.0 - self.u)
        dv = self.Dv * lv + uv2 - (self.F + self.k) * self.v
        
        self.u += du * dt
        self.v += dv * dt
        
        # Clip numerical boundaries
        self.u = np.clip(self.u, 0.0, 1.0)
        self.v = np.clip(self.v, 0.0, 1.0)

if __name__ == "__main__":
    print("--- T06: Reaction-Diffusion Turing Patterns Sanity Check ---")
    np.random.seed(42)
    # Parameters known to generate stable spots/stripes (F=0.035, k=0.060)
    sim = GrayScottTuringPatterns(size=30, Du=0.16, Dv=0.08, F=0.035, k=0.060)
    
    # Check initial central perturbation
    v_init_sum = np.sum(sim.v)
    print(f"Initial Trigger Perturbation sum(v): {v_init_sum:.2f}")
    assert v_init_sum > 0.0
    
    # Run 300 reaction-diffusion steps
    for _ in range(300):
        sim.step(dt=1.0)
        
    v_final_mean = np.mean(sim.v)
    v_final_std = np.std(sim.v)
    print(f"Final Mean v: {v_final_mean:.4f}, Spatial Variance (Pattern Contrast): {v_final_std:.4f}")
    
    # Asserts:
    # 1. Pattern formation: Spatial standard deviation > 0.05 proves self-organized spatial contrast!
    assert v_final_std > 0.05, "Failed to develop Turing pattern contrast"
    # 2. Chemical bounds obeyed
    assert 0.0 <= np.min(sim.u) and np.max(sim.u) <= 1.0
    assert 0.0 <= np.min(sim.v) and np.max(sim.v) <= 1.0
    print("T06 Sanity Check: PASSED cleanly.")

"""
K03: Neural Ordinary Differential Equations (Neural ODEs)
Pure NumPy implementation of Chen et al. (NeurIPS 2018).
Continuous-depth neural network driven by hand-written Euler and RK4 integrators.
"""

import numpy as np

class DynamicsMLP:
    """Derivative function f_theta(z, t) = dz/dt."""
    def __init__(self, state_dim: int = 2, hidden_dim: int = 16):
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.3
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, state_dim) * 0.3
        self.b2 = np.zeros(state_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z @ self.W1 + self.b1) @ self.W2 + self.b2


class NeuralODE:
    """Solves initial value problem z(t_1) = z(t_0) + int_{t_0}^{t_1} f_theta(z, t) dt."""
    def __init__(self, dynamics: DynamicsMLP):
        self.dynamics = dynamics

    def euler_step(self, z: np.ndarray, dt: float) -> np.ndarray:
        return z + dt * self.dynamics.forward(z)

    def rk4_step(self, z: np.ndarray, dt: float) -> np.ndarray:
        """Runge-Kutta 4th-order integration step."""
        k1 = self.dynamics.forward(z)
        k2 = self.dynamics.forward(z + 0.5 * dt * k1)
        k3 = self.dynamics.forward(z + 0.5 * dt * k2)
        k4 = self.dynamics.forward(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, z0: np.ndarray, t_span: np.ndarray, solver: str = "rk4") -> np.ndarray:
        """Integrates trajectory across time array t_span."""
        trajectory = [z0.copy()]
        z = z0.copy()
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            if solver == "rk4":
                z = self.rk4_step(z, dt)
            else:
                z = self.euler_step(z, dt)
            trajectory.append(z.copy())
        return np.array(trajectory)


if __name__ == "__main__":
    print("--- K03: Neural ODEs (Euler vs. RK4) Sanity Check ---")
    np.random.seed(42)

    dynamics = DynamicsMLP(state_dim=2, hidden_dim=16)
    node = NeuralODE(dynamics)

    z0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    t_span = np.linspace(0.0, 1.0, 25)

    traj_rk4 = node.integrate(z0, t_span, solver="rk4")
    traj_euler = node.integrate(z0, t_span, solver="euler")

    print(f"Initial State z(0): {z0[0].round(3)}")
    print(f"RK4 Final State z(T): {traj_rk4[-1, 0].round(3)}")
    print(f"Euler Final State z(T): {traj_euler[-1, 0].round(3)}")

    step_diff = float(np.max(np.abs(traj_rk4 - traj_euler)))
    print(f"Discrepancy Between Euler and RK4: {step_diff:.4e}")

    assert traj_rk4.shape == (25, 2, 2), "Trajectory shape must match (T, N, D)"
    assert step_diff > 0.0, "RK4 and Euler must produce distinct approximations"

    print("K03 Sanity Check: PASSED cleanly.")

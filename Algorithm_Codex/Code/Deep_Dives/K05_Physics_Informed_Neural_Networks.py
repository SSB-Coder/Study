"""
K05: Physics-Informed Neural Networks (PINNs)
Pure NumPy implementation of Raissi, Perdikaris, & Karniadakis (JCP 2019).
Solves 1D Harmonic Oscillator u_tt + omega_0^2 u = 0 via collocation least-squares residual minimization.
"""

import numpy as np

class HarmonicOscillatorPINN:
    """
    PINN solving u_tt + omega_0^2 u = 0 with u(0) = 1, u_t(0) = 0.
    Analytical ground truth: u(t) = cos(omega_0 * t).
    """
    def __init__(self, hidden_dim: int = 24, omega0: float = 1.0):
        self.hidden_dim = hidden_dim
        self.omega0 = omega0
        self.W1 = np.random.randn(1, hidden_dim) * 1.5
        self.b1 = np.random.randn(hidden_dim) * 1.5
        self.W2 = None

    def fit(self, t_colloc: np.ndarray, w_bc: float = 50.0):
        # 1. Collocation residual operator features
        z = t_colloc @ self.W1 + self.b1
        a = np.tanh(z)
        da = (1.0 - a ** 2) * self.W1
        d2a = -2.0 * a * (1.0 - a ** 2) * (self.W1 ** 2)

        # Residual operator: d2u/dt2 + omega0^2 * u = 0 -> (d2a + omega0^2 * a) @ W2 = 0
        A_res = d2a + (self.omega0 ** 2) * a
        y_res = np.zeros(len(t_colloc))

        # 2. Boundary conditions at t=0: u(0) = 1, u_t(0) = 0
        t0 = np.array([[0.0]])
        z0 = t0 @ self.W1 + self.b1
        a0 = np.tanh(z0)
        da0 = (1.0 - a0 ** 2) * self.W1

        A_bc1 = a0
        y_bc1 = np.array([1.0])
        A_bc2 = da0
        y_bc2 = np.array([0.0])

        # Composite linear system
        A_total = np.vstack([A_res, w_bc * A_bc1, w_bc * A_bc2])
        y_total = np.concatenate([y_res, w_bc * y_bc1, w_bc * y_bc2])

        self.W2, _, _, _ = np.linalg.lstsq(A_total, y_total, rcond=None)

    def predict(self, t: np.ndarray) -> np.ndarray:
        z = t @ self.W1 + self.b1
        a = np.tanh(z)
        return (a @ self.W2).squeeze()


if __name__ == "__main__":
    print("--- K05: Physics-Informed Neural Networks (PINNs) Sanity Check ---")
    np.random.seed(42)

    pinn = HarmonicOscillatorPINN(hidden_dim=24, omega0=1.0)
    t_colloc = np.linspace(0.0, 3.0, 40).reshape(-1, 1)

    pinn.fit(t_colloc, w_bc=50.0)

    # Evaluate against analytical solution u(t) = cos(t)
    t_test = np.linspace(0.0, 3.0, 100).reshape(-1, 1)
    u_pred = pinn.predict(t_test)
    u_true = np.cos(t_test.squeeze())

    mse = float(np.mean((u_pred - u_true) ** 2))
    print(f"PINN Mean Squared Error vs. Analytical u(t)=cos(t): {mse:.6f}")

    assert mse < 0.05, f"Expected PINN MSE < 0.05, got {mse:.6f}"
    print("K05 Sanity Check: PASSED cleanly.")

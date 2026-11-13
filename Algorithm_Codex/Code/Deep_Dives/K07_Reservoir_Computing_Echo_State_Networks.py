"""
K07: Reservoir Computing & Echo State Networks (ESN)
Pure NumPy implementation of an Echo State Network for sequential time series forecasting.
"""

import numpy as np

class EchoStateNetwork:
    """Echo State Network with spectral radius normalization."""
    def __init__(self, n_reservoir: int = 50, spectral_radius: float = 0.95, sparsity: float = 0.2, seed: int = 42):
        self.n_res = n_reservoir
        self.rho = spectral_radius
        self.sparsity = sparsity
        self.seed = seed
        self.W_in = None
        self.W_res = None
        self.W_out = None

    def _init_weights(self, in_dim: int, rng: np.random.Generator):
        self.W_in = rng.uniform(-0.5, 0.5, size=(self.n_res, in_dim))

        # Sparse reservoir matrix
        W = rng.normal(0, 1.0, size=(self.n_res, self.n_res))
        mask = rng.random((self.n_res, self.n_res)) > self.sparsity
        W[mask] = 0.0

        # Rescale to exact spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eig = float(np.max(np.abs(eigenvalues)))
        if max_eig > 1e-8:
            self.W_res = (self.rho / max_eig) * W
        else:
            self.W_res = W

    def fit(self, u_seq: np.ndarray, y_target: np.ndarray, washout: int = 15, reg: float = 1e-4):
        # u_seq: (T, in_dim), y_target: (T, out_dim)
        T, in_dim = u_seq.shape
        rng = np.random.default_rng(self.seed)
        self._init_weights(in_dim, rng)

        # Collect states
        X = np.zeros((T, self.n_res))
        x = np.zeros(self.n_res)
        for t in range(T):
            u = u_seq[t]
            x = np.tanh(self.W_in @ u + self.W_res @ x)
            X[t] = x

        # Discard washout period
        X_train = X[washout:]
        Y_train = y_target[washout:]

        # Ridge regression: W_out = Y^T X (X^T X + reg I)^(-1)
        I = np.eye(self.n_res)
        self.W_out = Y_train.T @ X_train @ np.linalg.inv(X_train.T @ X_train + reg * I)

    def predict(self, u_seq: np.ndarray) -> np.ndarray:
        T, _ = u_seq.shape
        X = np.zeros((T, self.n_res))
        x = np.zeros(self.n_res)
        for t in range(T):
            u = u_seq[t]
            x = np.tanh(self.W_in @ u + self.W_res @ x)
            X[t] = x
        return X @ self.W_out.T


if __name__ == "__main__":
    print("--- K07: Echo State Network (ESN) Sanity Check ---")
    # Non-linear 1D sequential task: predict next step of a sine wave
    np.random.seed(42)
    t_vals = np.linspace(0, 20 * np.pi, 300)
    signal = np.sin(t_vals)

    # Input: signal[t], Target: signal[t+1]
    u = signal[:-1].reshape(-1, 1)
    y = signal[1:].reshape(-1, 1)

    esn = EchoStateNetwork(n_reservoir=40, spectral_radius=0.95, seed=42)
    esn.fit(u, y, washout=20)

    preds = esn.predict(u)
    test_mse = float(np.mean((preds[20:] - y[20:]) ** 2))
    print(f"Echo State Network Forecasting MSE: {test_mse:.5f}")

    assert test_mse < 0.05, f"ESN should track sequential oscillation accurately, got MSE {test_mse}"
    print("K07 Sanity Check: PASSED cleanly.")

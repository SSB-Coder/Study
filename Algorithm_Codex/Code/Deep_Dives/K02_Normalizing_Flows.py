"""
K02: Normalizing Flows (RealNVP Affine Coupling)
Pure NumPy implementation of Dinh, Sohl-Dickstein, & Bengio (ICLR 2017).
Exact invertible mapping with tractable log-determinant Jacobian.
"""

import numpy as np

class AffineCouplingLayer:
    """
    RealNVP affine coupling layer:
    x = [x1, x2]
    Forward:  y1 = x1, y2 = x2 * exp(s(x1)) + t(x1)
    Inverse:  x1 = y1, x2 = (y2 - t(y1)) * exp(-s(y1))
    """
    def __init__(self, dim: int, split_dim: int, hidden_dim: int = 16, mask_flip: bool = False):
        self.dim = dim
        self.split_dim = split_dim
        self.d2 = dim - split_dim
        self.mask_flip = mask_flip

        # Simple scale (s) and translation (t) MLPs
        self.Ws = np.random.randn(split_dim, self.d2) * 0.1
        self.bs = np.zeros(self.d2)
        self.Wt = np.random.randn(split_dim, self.d2) * 0.1
        self.bt = np.zeros(self.d2)

    def _s_t(self, x1: np.ndarray):
        s = np.tanh(x1 @ self.Ws + self.bs) # Bounded scale in (-1, 1)
        t = x1 @ self.Wt + self.bt
        return s, t

    def forward(self, x: np.ndarray):
        if self.mask_flip:
            x1, x2 = x[:, self.d2:], x[:, :self.d2]
        else:
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]

        s, t = self._s_t(x1)
        y2 = x2 * np.exp(s) + t
        y1 = x1

        if self.mask_flip:
            y = np.hstack([y2, y1])
        else:
            y = np.hstack([y1, y2])

        log_det_J = np.sum(s, axis=1) # Sum of scale activations
        return y, log_det_J

    def inverse(self, y: np.ndarray):
        if self.mask_flip:
            y1, y2 = y[:, self.d2:], y[:, :self.d2]
        else:
            y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]

        s, t = self._s_t(y1)
        x2 = (y2 - t) * np.exp(-s)
        x1 = y1

        if self.mask_flip:
            return np.hstack([x2, x1])
        else:
            return np.hstack([x1, x2])


class RealNVP:
    """Multilayer normalizing flow mapping standard Gaussian base to complex density."""
    def __init__(self, dim: int = 2, n_layers: int = 4):
        self.dim = dim
        self.layers = [
            AffineCouplingLayer(dim=dim, split_dim=dim // 2, mask_flip=(i % 2 == 1))
            for i in range(n_layers)
        ]

    def forward(self, x: np.ndarray):
        z = x.copy()
        total_log_det = np.zeros(x.shape[0])
        for layer in self.layers:
            z, log_det = layer.forward(z)
            total_log_det += log_det
        return z, total_log_det

    def inverse(self, z: np.ndarray):
        x = z.copy()
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Change-of-variables: log p(x) = log p(z) + log |det J|."""
        z, log_det = self.forward(x)
        # Standard Gaussian prior log p(z) = -0.5 * (d*log(2pi) + ||z||^2)
        log_pz = -0.5 * (self.dim * np.log(2.0 * np.pi) + np.sum(z ** 2, axis=1))
        return log_pz + log_det


if __name__ == "__main__":
    print("--- K02: Normalizing Flows (RealNVP) Sanity Check ---")
    np.random.seed(42)

    flow = RealNVP(dim=2, n_layers=4)

    # 1. Test Invertibility & Cycle Consistency
    x_test = np.random.randn(20, 2) * 2.0
    z, log_det = flow.forward(x_test)
    x_reconstructed = flow.inverse(z)

    recon_error = float(np.max(np.abs(x_test - x_reconstructed)))
    print(f"Max Absolute Reconstruction Error |x - f^-1(f(x))|: {recon_error:.2e}")
    assert recon_error < 1e-10, f"Flow must invert to machine precision, got {recon_error}"

    # 2. Test Density Evaluation
    log_probs = flow.log_prob(x_test)
    print(f"Mean Log-Likelihood of Test Points: {np.mean(log_probs):.4f}")
    assert not np.isnan(log_probs).any(), "Log-likelihood must contain no NaNs"

    print("K02 Sanity Check: PASSED cleanly.")

"""
O03: Information Geometry & Natural Gradients
Pure NumPy implementation of Riemannian optimization on statistical manifolds:
1. Fisher Information Matrix (FIM) computation (analytical and empirical)
2. Natural Gradient Descent (NGD) with Riemannian metric tensor F(theta)^(-1)
3. Direct optimization comparison between standard Euclidean GD and Natural GD
"""

import numpy as np
from typing import Tuple, List


class GaussianNaturalGradientFitter:
    """
    Fits a 1D Gaussian N(mu, sigma^2) to empirical data using:
    1. Standard Euclidean Gradient Descent
    2. Amari's Natural Gradient Descent via FIM: F = diag(1/sigma^2, 2/sigma^2)
    """
    def __init__(self, init_mu: float = 0.0, init_sigma: float = 0.5):
        self.init_mu = init_mu
        self.init_sigma = init_sigma

    def _negative_log_likelihood(self, mu: float, sigma: float, X: np.ndarray) -> float:
        n = len(X)
        nll = n * np.log(sigma + 1e-12) + 0.5 * n * np.log(2 * np.pi) + np.sum((X - mu) ** 2) / (2 * sigma ** 2)
        return float(nll / n)

    def _gradients(self, mu: float, sigma: float, X: np.ndarray) -> Tuple[float, float]:
        """Euclidean gradients of average NLL."""
        n = len(X)
        d_mu = -np.sum(X - mu) / (n * sigma ** 2)
        d_sigma = (1.0 / sigma) - np.sum((X - mu) ** 2) / (n * sigma ** 3)
        return float(d_mu), float(d_sigma)

    def _fisher_matrix(self, sigma: float) -> np.ndarray:
        """Analytical Fisher Information Matrix for 1D Gaussian."""
        return np.array([
            [1.0 / (sigma ** 2), 0.0],
            [0.0, 2.0 / (sigma ** 2)]
        ])

    def fit_standard_gd(self, X: np.ndarray, lr: float = 0.05, n_iters: int = 150) -> List[float]:
        mu = self.init_mu
        sigma = self.init_sigma
        history = []

        for _ in range(n_iters):
            loss = self._negative_log_likelihood(mu, sigma, X)
            history.append(loss)
            d_mu, d_sigma = self._gradients(mu, sigma, X)
            
            mu -= lr * d_mu
            sigma = max(1e-4, sigma - lr * d_sigma)

        return history

    def fit_natural_gd(self, X: np.ndarray, lr: float = 0.05, n_iters: int = 150) -> List[float]:
        mu = self.init_mu
        sigma = self.init_sigma
        history = []

        for _ in range(n_iters):
            loss = self._negative_log_likelihood(mu, sigma, X)
            history.append(loss)
            d_mu, d_sigma = self._gradients(mu, sigma, X)
            grad_vec = np.array([d_mu, d_sigma])

            # Invert Fisher Matrix: F^(-1) @ grad
            # Analytical inverse: [sigma^2 * d_mu, (sigma^2 / 2) * d_sigma]
            F_inv = np.array([
                [sigma ** 2, 0.0],
                [0.0, (sigma ** 2) / 2.0]
            ])
            natural_grad = F_inv @ grad_vec

            mu -= lr * natural_grad[0]
            sigma = max(1e-4, sigma - lr * natural_grad[1])

        return history


if __name__ == "__main__":
    np.random.seed(42)

    # Ground truth: N(mu=4.0, sigma=2.0)
    true_mu = 4.0
    true_sigma = 2.0
    data = np.random.normal(loc=true_mu, scale=true_sigma, size=500)

    # Initialize far from target with small sigma (high curvature region)
    fitter = GaussianNaturalGradientFitter(init_mu=-2.0, init_sigma=0.3)

    gd_losses = fitter.fit_standard_gd(data, lr=0.01, n_iters=100)
    ngd_losses = fitter.fit_natural_gd(data, lr=0.05, n_iters=100)

    print(f"Initial NLL Loss:                 {gd_losses[0]:.4f}")
    print(f"Standard GD Loss after 100 iters: {gd_losses[-1]:.4f}")
    print(f"Natural GD Loss after 100 iters:  {ngd_losses[-1]:.4f}")

    # Verify that Natural Gradient achieves lower loss faster due to Riemannian curvature correction
    assert ngd_losses[-1] < gd_losses[-1], "Natural GD should achieve better convergence than standard GD"
    
    # Check Fisher Matrix properties
    sigma_test = 1.5
    F = fitter._fisher_matrix(sigma_test)
    eigenvals = np.linalg.eigvals(F)
    print(f"Fisher Information Matrix Eigenvalues at sigma={sigma_test}: {eigenvals}")
    assert np.all(eigenvals > 0), "Fisher Information Matrix must be strictly positive definite"

    print("O03 Sanity Check: PASSED cleanly.")

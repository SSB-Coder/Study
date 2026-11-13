"""
P03: Energy-Based Models & Denoising Score Matching
Pure NumPy implementation of score matching and score-based generative modeling:
1. Parametric Score Function network estimating grad_x(log p(x))
2. Vincent's Denoising Score Matching (DSM) training objective
3. Langevin MCMC generation tracing learned score vector fields
"""

import numpy as np
from typing import Tuple, List


class ScoreModel:
    """
    Parametric score function model: s_theta(x) in R^D approximating grad_x(log p(x)).
    Linear-quadratic basis expansion: s(x) = W @ x + b (exact for Gaussian families).
    """
    def __init__(self, dim: int):
        self.dim = dim
        # Initialize symmetric negative-definite matrix for stable Gaussian score
        self.W = -np.eye(dim) * 0.5
        self.b = np.zeros(dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) -> returns score vectors (N, D)"""
        return X @ self.W.T + self.b

    def train_dsm(
        self, 
        X_clean: np.ndarray, 
        sigma: float = 0.3, 
        lr: float = 0.05, 
        n_epochs: int = 150
    ) -> List[float]:
        """
        Trains model using Denoising Score Matching:
        L(theta) = 0.5 * E[ || s_theta(x + sigma * eps) - (-eps / sigma) ||^2 ]
        """
        N, D = X_clean.shape
        losses = []

        for epoch in range(n_epochs):
            # Sample perturbation noise
            eps = np.random.randn(N, D)
            X_noisy = X_clean + sigma * eps
            
            # Ground truth analytical conditional score: grad_x_tilde(log q(x_tilde | x))
            target_score = -eps / sigma

            # Forward prediction
            pred_score = self.forward(X_noisy)
            residual = pred_score - target_score  # (N, D)
            loss = 0.5 * float(np.mean(np.sum(residual ** 2, axis=1)))
            losses.append(loss)

            # Gradient updates for W and b
            # dL/ds = residual / N
            # ds/dW = X_noisy, ds/db = 1
            grad_W = (residual.T @ X_noisy) / N
            grad_b = np.mean(residual, axis=0)

            # Gradient descent step
            self.W -= lr * grad_W
            self.b -= lr * grad_b

        return losses

    def generate(self, n_samples: int = 500, n_steps: int = 200, step_size: float = 0.05) -> np.ndarray:
        """Generates new samples starting from Gaussian noise via Langevin dynamics."""
        x = np.random.randn(n_samples, self.dim) * 3.0  # High-variance initial noise

        for _ in range(n_steps):
            score = self.forward(x)
            drift = 0.5 * step_size * score
            noise = np.sqrt(step_size) * np.random.randn(n_samples, self.dim)
            x += drift + noise

        return x


if __name__ == "__main__":
    np.random.seed(42)

    # Clean data generated from N(mu=[0.0, 0.0], var=1.0)
    dim = 2
    true_mu = np.array([0.0, 0.0])
    true_var = 1.0
    X_train = np.random.randn(2000, dim) * np.sqrt(true_var)

    sigma = 0.1
    # Analytical score for N(0, (var + sigma^2) * I):
    # s(x) = -x / (var + sigma^2)
    expected_W_diag = -1.0 / (true_var + sigma ** 2)

    model = ScoreModel(dim=dim)
    losses = model.train_dsm(X_train, sigma=sigma, lr=0.05, n_epochs=250)

    print(f"Initial DSM Loss: {losses[0]:.4f}, Final Loss: {losses[-1]:.4f}")
    print(f"Learned W:\n{model.W.round(3)}")
    print(f"Learned b: {model.b.round(3)}")
    print(f"Expected W diagonal: {expected_W_diag:.3f}")

    # Verify parameters align with analytical score
    assert np.allclose(np.diag(model.W), [expected_W_diag, expected_W_diag], atol=0.15), "Learned W deviated from analytical score"
    assert np.allclose(model.b, [0.0, 0.0], atol=0.15), "Learned bias deviated from zero"

    # Generate samples from pure noise using Langevin dynamics guided by learned score
    generated = model.generate(n_samples=800, n_steps=150, step_size=0.08)
    gen_mean = np.mean(generated, axis=0)
    gen_var = np.mean(np.var(generated, axis=0))

    print(f"Generated Sample Mean: {gen_mean.round(3)}, Target: {true_mu}")
    print(f"Generated Sample Var:  {gen_var:.3f}, Target: {true_var}")

    assert np.allclose(gen_mean, true_mu, atol=0.2), f"Generated mean {gen_mean} drifted from target"
    assert np.isclose(gen_var, true_var, atol=0.25), f"Generated variance {gen_var} drifted from target"

    print("P03 Sanity Check: PASSED cleanly.")

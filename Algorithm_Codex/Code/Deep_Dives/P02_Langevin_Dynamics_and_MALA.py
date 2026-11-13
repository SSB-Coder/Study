"""
P02: Langevin Dynamics & Metropolis-Adjusted Langevin Algorithm (MALA)
Pure NumPy implementation of continuous-time SDE sampling and diffusion:
1. Unadjusted Langevin Algorithm (ULA) via Euler-Maruyama discretization
2. Metropolis-Adjusted Langevin Algorithm (MALA) with exact transition density ratios
"""

import numpy as np
from typing import Callable, Tuple


class LangevinSampler:
    """
    Simulates Langevin Diffusion for posterior sampling:
    dx_t = 0.5 * grad(log p(x_t)) dt + dW_t
    """
    def __init__(
        self, 
        log_prob: Callable[[np.ndarray], float],
        grad_log_prob: Callable[[np.ndarray], np.ndarray],
        dim: int,
        step_size: float = 0.05
    ):
        self.log_p = log_prob
        self.grad_log_p = grad_log_prob
        self.dim = dim
        self.eps = step_size
        self.eps_sq = step_size ** 2

    def sample_ula(self, init_x: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """Unadjusted Langevin Algorithm (ULA)."""
        samples = np.zeros((n_steps, self.dim))
        x = init_x.copy().astype(float)

        for s in range(n_steps):
            drift = 0.5 * self.eps_sq * self.grad_log_p(x)
            noise = self.eps * np.random.randn(self.dim)
            x += drift + noise
            samples[s] = x

        return samples

    def _log_q(self, x_to: np.ndarray, x_from: np.ndarray) -> float:
        """Evaluates Gaussian proposal log-density: q(x_to | x_from)."""
        mu = x_from + 0.5 * self.eps_sq * self.grad_log_p(x_from)
        diff = x_to - mu
        # Gaussian log density with variance eps_sq
        log_norm = -0.5 * self.dim * np.log(2.0 * np.pi * self.eps_sq)
        quad = -0.5 * float(np.dot(diff, diff)) / self.eps_sq
        return log_norm + quad

    def sample_mala(self, init_x: np.ndarray, n_steps: int = 1000) -> Tuple[np.ndarray, float]:
        """Metropolis-Adjusted Langevin Algorithm (MALA)."""
        samples = np.zeros((n_steps, self.dim))
        x = init_x.copy().astype(float)
        accepted = 0

        for s in range(n_steps):
            # Propose next state
            mu = x + 0.5 * self.eps_sq * self.grad_log_p(x)
            x_prop = mu + self.eps * np.random.randn(self.dim)

            # Evaluate forward and backward proposal probabilities
            log_q_forward = self._log_q(x_prop, x)
            log_q_reverse = self._log_q(x, x_prop)

            # Metropolis Hastings ratio
            log_accept = self.log_p(x_prop) - self.log_p(x) + log_q_reverse - log_q_forward

            if np.log(np.random.uniform(1e-12, 1.0)) < log_accept:
                x = x_prop
                accepted += 1

            samples[s] = x

        accept_rate = accepted / n_steps
        return samples, accept_rate


if __name__ == "__main__":
    np.random.seed(42)

    # 2D Correlated Gaussian Distribution for precise statistical validation
    true_mu = np.array([1.5, -0.5])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    inv_cov = np.linalg.inv(cov)

    def log_prob_gauss(x: np.ndarray) -> float:
        diff = x - true_mu
        return -0.5 * float(diff @ inv_cov @ diff)

    def grad_log_prob_gauss(x: np.ndarray) -> np.ndarray:
        return -inv_cov @ (x - true_mu)

    sampler = LangevinSampler(
        log_prob=log_prob_gauss,
        grad_log_prob=grad_log_prob_gauss,
        dim=2,
        step_size=0.4
    )

    init = np.array([1.0, 0.0])
    n_samples = 6000

    # 1. Run ULA
    ula_samples = sampler.sample_ula(init, n_steps=1000)

    # 2. Run MALA
    mala_samples, accept_rate = sampler.sample_mala(init, n_steps=n_samples)
    print(f"MALA Acceptance Rate: {accept_rate * 100:.1f}%")

    post_burnin = mala_samples[1000:]
    sample_mean = np.mean(post_burnin, axis=0)
    sample_cov = np.cov(post_burnin, rowvar=False)

    print(f"True Mean:     {true_mu}, Estimated: {sample_mean.round(3)}")
    print(f"Sample Covariance:\n{sample_cov.round(3)}")

    assert accept_rate >= 0.70, f"Expected MALA acceptance >= 70%, got {accept_rate * 100:.1f}%"
    assert np.allclose(sample_mean, true_mu, atol=0.25), f"Mean mismatch: {sample_mean} vs {true_mu}"
    assert np.allclose(sample_cov, cov, atol=0.25), f"Covariance mismatch:\n{sample_cov}"

    print("P02 Sanity Check: PASSED cleanly.")

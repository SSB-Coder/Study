"""
P01: Hamiltonian Monte Carlo (HMC) & Leapfrog Numerical Integration
Pure NumPy implementation of physics-based MCMC sampling in phase space.
Powers modern Bayesian inference engines (Stan / PyMC).
"""

import numpy as np
from typing import Callable, Tuple, List


class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo sampler with symplectic Leapfrog integrator.
    H(q, p) = U(q) + 0.5 * p^T M^(-1) p
    """
    def __init__(
        self, 
        neg_log_prob: Callable[[np.ndarray], float], 
        grad_neg_log_prob: Callable[[np.ndarray], np.ndarray],
        dim: int,
        step_size: float = 0.1,
        n_leapfrog: int = 15
    ):
        self.U = neg_log_prob
        self.grad_U = grad_neg_log_prob
        self.dim = dim
        self.eps = step_size
        self.L = n_leapfrog

    def _leapfrog(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symplectic, reversible leapfrog numerical integration."""
        q_curr = q.copy()
        p_curr = p.copy()

        # Half-step momentum kick
        p_curr -= 0.5 * self.eps * self.grad_U(q_curr)

        # Alternating full steps
        for step in range(self.L):
            q_curr += self.eps * p_curr
            if step < self.L - 1:
                p_curr -= self.eps * self.grad_U(q_curr)

        # Final half-step momentum kick
        p_curr -= 0.5 * self.eps * self.grad_U(q_curr)
        # Negate momentum for time-reversibility
        return q_curr, -p_curr

    def sample(self, init_q: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, float]:
        """Draws MCMC samples via Hamiltonian dynamics and Metropolis filtering."""
        samples = np.zeros((n_samples, self.dim))
        q = init_q.astype(float)
        accepted = 0

        for s in range(n_samples):
            # Sample random Gaussian momentum: p ~ N(0, I)
            p = np.random.randn(self.dim)
            
            # Initial total energy
            current_U = self.U(q)
            current_K = 0.5 * float(np.dot(p, p))
            current_H = current_U + current_K

            # Simulate Hamiltonian trajectory
            q_prop, p_prop = self._leapfrog(q, p)

            # Proposed total energy
            prop_U = self.U(q_prop)
            prop_K = 0.5 * float(np.dot(p_prop, p_prop))
            prop_H = prop_U + prop_K

            # Metropolis acceptance
            log_accept_ratio = current_H - prop_H
            if np.log(np.random.uniform(1e-12, 1.0)) < log_accept_ratio:
                q = q_prop
                accepted += 1

            samples[s] = q

        accept_rate = accepted / n_samples
        return samples, accept_rate


if __name__ == "__main__":
    np.random.seed(42)

    # Correlated Bivariate Gaussian target:
    # Mean = [2.0, -1.0], Covariance with correlation rho = 0.9
    true_mean = np.array([2.0, -1.0])
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    inv_cov = np.linalg.inv(cov)

    def potential_energy(q: np.ndarray) -> float:
        diff = q - true_mean
        return 0.5 * float(diff @ inv_cov @ diff)

    def potential_gradient(q: np.ndarray) -> np.ndarray:
        return inv_cov @ (q - true_mean)

    hmc = HamiltonianMonteCarlo(
        neg_log_prob=potential_energy,
        grad_neg_log_prob=potential_gradient,
        dim=2,
        step_size=0.15,
        n_leapfrog=10
    )

    init_pos = np.array([0.0, 0.0])
    samples, acc_rate = hmc.sample(init_pos, n_samples=1500)

    # Discard burn-in
    post_burnin = samples[300:]
    sample_mean = np.mean(post_burnin, axis=0)
    sample_cov = np.cov(post_burnin, rowvar=False)

    print(f"HMC Acceptance Rate: {acc_rate * 100:.1f}%")
    print(f"True Mean:   {true_mean}, Estimated: {sample_mean.round(3)}")
    print(f"Sample Covariance:\n{sample_cov.round(3)}")

    assert acc_rate >= 0.70, f"Expected acceptance rate >= 70%, got {acc_rate * 100:.1f}%"
    assert np.allclose(sample_mean, true_mean, atol=0.25), "Sample mean diverged from ground truth"
    assert np.allclose(sample_cov, cov, atol=0.25), "Sample covariance diverged from ground truth"

    print("P01 Sanity Check: PASSED cleanly.")

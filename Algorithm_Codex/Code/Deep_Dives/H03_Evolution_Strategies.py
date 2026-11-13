"""
H03: Evolution Strategies (ES & CMA-ES)
Pure NumPy implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
"""

import numpy as np

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy for continuous optimization."""
    def __init__(self, dim: int, m_init: np.ndarray, sigma: float = 0.5, pop_size: int = 20):
        self.dim = dim
        self.m = np.array(m_init, dtype=np.float64)
        self.sigma = sigma
        self.pop_size = pop_size
        self.mu = pop_size // 2

        # Recombination weights: sum(w) = 1
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)

        # Covariance matrix and learning rates
        self.C = np.eye(dim, dtype=np.float64)
        self.c_mu = min(1.0 - 1e-4, 2.0 / (dim ** 2 + self.mu))

    def optimize(self, cost_fn, generations: int = 50, seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        best_x = self.m.copy()
        best_cost = cost_fn(best_x)

        for _ in range(generations):
            # Sample population: x_k = m + sigma * N(0, C)
            # Factorize C = L L^T via Cholesky
            try:
                L = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                # Regularize if ill-conditioned
                self.C += 1e-6 * np.eye(self.dim)
                L = np.linalg.cholesky(self.C)

            z = rng.normal(0, 1, size=(self.pop_size, self.dim))
            samples = self.m + self.sigma * (z @ L.T)

            # Evaluate
            costs = np.array([cost_fn(ind) for ind in samples])
            sorted_idx = np.argsort(costs)

            if costs[sorted_idx[0]] < best_cost:
                best_cost = float(costs[sorted_idx[0]])
                best_x = samples[sorted_idx[0]].copy()

            # Select top mu
            top_samples = samples[sorted_idx[:self.mu]]
            
            # Update mean: m = sum w_i x_i
            old_m = self.m.copy()
            self.m = np.sum(self.weights[:, None] * top_samples, axis=0)

            # Rank-mu covariance update
            y = (top_samples - old_m) / self.sigma
            C_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                C_mu += self.weights[i] * np.outer(y[i], y[i])

            self.C = (1.0 - self.c_mu) * self.C + self.c_mu * C_mu
            # Symmetrize
            self.C = 0.5 * (self.C + self.C.T)

        return best_x, best_cost


if __name__ == "__main__":
    print("--- H03: CMA-ES Sanity Check ---")
    # Optimize 3D Sphere function: f(x) = sum(x_i^2), optimum at [0, 0, 0] with f(x)=0
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    m_start = np.array([2.5, -3.0, 4.0])
    cma = CMAES(dim=3, m_init=m_start, sigma=0.8, pop_size=24)
    best_sol, best_val = cma.optimize(sphere, generations=60, seed=42)

    print(f"Initial Cost at {m_start}: {sphere(m_start):.4f}")
    print(f"CMA-ES Discovered Minimum: {best_sol} | Final Cost: {best_val:.6f}")

    assert best_val < 0.05, f"CMA-ES should drive Sphere function near zero, got {best_val}"
    print("H03 Sanity Check: PASSED cleanly.")

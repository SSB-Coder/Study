"""
E09. Bayesian Optimization (Expected Improvement & UCB on Gaussian Process)
From-scratch implementation in pure NumPy.
"""

import numpy as np
import math

class GaussianProcessMini:
    """Minimal self-contained GP for Bayesian Optimization."""
    def __init__(self, length_scale: float = 0.5, sigma_f: float = 1.0, noise: float = 1e-4):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise = noise
        self.X = None
        self.y = None
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        diff = self.X[:, None, :] - self.X[None, :, :]
        K = (self.sigma_f ** 2) * np.exp(-np.sum(diff ** 2, axis=-1) / (2.0 * self.length_scale ** 2))
        K += (self.noise ** 2) * np.eye(len(self.X))
        self.K_inv = np.linalg.pinv(K)
        return self

    def predict(self, X_cand: np.ndarray):
        diff_star = self.X[:, None, :] - X_cand[None, :, :]
        K_star = (self.sigma_f ** 2) * np.exp(-np.sum(diff_star ** 2, axis=-1) / (2.0 * self.length_scale ** 2))
        
        diff_cand = X_cand[:, None, :] - X_cand[None, :, :]
        K_cand = (self.sigma_f ** 2) * np.exp(-np.sum(diff_cand ** 2, axis=-1) / (2.0 * self.length_scale ** 2))

        mu = K_star.T @ (self.K_inv @ self.y)
        sigma2 = np.diag(K_cand - K_star.T @ (self.K_inv @ K_star))
        std = np.sqrt(np.maximum(1e-12, sigma2))
        return mu, std

class BayesianOptimizationScratch:
    """Bayesian Optimization using Expected Improvement and UCB acquisition functions."""
    def __init__(self, acquisition: str = "ei", kappa: float = 2.0, length_scale: float = 0.5):
        self.acquisition = acquisition
        self.kappa = kappa
        self.gp = GaussianProcessMini(length_scale=length_scale)

    @staticmethod
    def _std_norm_cdf(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

    @staticmethod
    def _std_norm_pdf(x):
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)

    def acquisition_score(self, mu: np.ndarray, std: np.ndarray, current_max: float) -> np.ndarray:
        if self.acquisition == "ucb":
            # UCB: mu + kappa * std <- eq. 3.2
            return mu + self.kappa * std
        elif self.acquisition == "ei":
            # Expected Improvement <- eq. 3.1
            diff = mu - current_max
            z = diff / (std + 1e-12)
            ei = diff * self._std_norm_cdf(z) + std * self._std_norm_pdf(z)
            ei[std <= 1e-6] = 0.0
            return ei
        raise ValueError(f"Unknown acquisition {self.acquisition}")

    def optimize(self, objective_fn, bounds: tuple, n_initial: int = 4, n_iters: int = 15):
        low, high = bounds
        # Initial evaluations
        X = np.linspace(low, high, n_initial).reshape(-1, 1)
        y = np.array([objective_fn(x) for x in X])

        candidate_grid = np.linspace(low, high, 300).reshape(-1, 1)

        for iteration in range(n_iters):
            self.gp.fit(X, y)
            mu, std = self.gp.predict(candidate_grid)
            acq = self.acquisition_score(mu, std, current_max=float(np.max(y)))

            # Pick next point maximizing acquisition function
            best_idx = int(np.argmax(acq))
            x_next = candidate_grid[best_idx:best_idx + 1]

            # Evaluate black box
            y_next = objective_fn(x_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

        best_idx = np.argmax(y)
        return float(X[best_idx, 0]), float(y[best_idx])

if __name__ == "__main__":
    print("--- E09: Bayesian Optimization Sanity Check ---")
    np.random.seed(42)
    # Maximize multi-modal 1D function: f(x) = - (x - 2)^2 + 3 * cos(3 * x) on [0, 5]
    # Global maximum is near x = 2.05 where f(x) approx 2.9
    def test_func(x):
        x_val = float(np.ravel(x)[0])
        return - (x_val - 2.0) ** 2 + 3.0 * np.cos(3.0 * x_val)

    bo = BayesianOptimizationScratch(acquisition="ei", length_scale=0.4)
    best_x, best_y = bo.optimize(test_func, bounds=(0.0, 5.0), n_initial=4, n_iters=15)
    
    print(f"Bayesian Optimization discovered maximum at x = {best_x:.4f}, f(x) = {best_y:.4f}")
    assert best_y > 2.0 # Must find the high peak
    print("E09 Sanity Check: PASSED cleanly.")

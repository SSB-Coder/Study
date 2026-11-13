"""
W14. Surrogate-Based Design Optimization (Gaussian Process & Expected Improvement)
From-scratch implementation in pure NumPy.
"""

import numpy as np
import math

class SurrogateOptimizer:
    """
    Optimizes expensive black-box engineering simulators (e.g. engine calibration,
    aerodynamic CFD, antenna tuning) using a Gaussian Process surrogate and
    Expected Improvement (EI) Bayesian Optimization.
    """
    
    def __init__(self, bounds: list, length_scale: float = 0.2, noise: float = 1e-4):
        """
        bounds: list of (min, max) tuples for each parameter.
        """
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)
        self.l = length_scale
        self.noise = noise
        self.X_sample = []
        self.y_sample = []

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared Exponential / RBF Kernel <- eq. 3.1."""
        dist_sq = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=-1)
        return np.exp(-0.5 * dist_sq / (self.l**2))

    def add_observation(self, x: np.ndarray, y: float):
        self.X_sample.append(np.array(x, dtype=float))
        self.y_sample.append(float(y))

    def predict(self, X_cand: np.ndarray):
        """
        Computes GP posterior mean mu and variance sigma^2 <- eq. 3.2.
        """
        X_cand = np.atleast_2d(X_cand)
        X_train = np.array(self.X_sample)
        y_train = np.array(self.y_sample)
        
        K = self._rbf_kernel(X_train, X_train) + (self.noise**2) * np.eye(len(X_train))
        K_s = self._rbf_kernel(X_train, X_cand)
        K_ss = self._rbf_kernel(X_cand, X_cand)
        
        # Cholesky solve for stability
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # Mean
        mu = K_s.T @ alpha
        
        # Variance
        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-10))
        return mu, sigma

    def expected_improvement(self, X_cand: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement acquisition function for MINIMIZATION <- eq. 3.3:
        EI(x) = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
        where Z = (y_best - mu - xi) / sigma.
        """
        mu, sigma = self.predict(X_cand)
        y_best = np.min(self.y_sample)
        
        diff = y_best - mu - xi
        Z = diff / sigma
        
        # Standard normal CDF and PDF in pure NumPy
        phi = np.exp(-0.5 * Z**2) / np.sqrt(2.0 * np.pi)
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(Z / np.sqrt(2.0)))
        
        ei = diff * Phi + sigma * phi
        ei[sigma < 1e-8] = 0.0
        return ei

    def suggest_next(self, n_candidates: int = 1000) -> np.ndarray:
        """Draws uniform candidates across bounds and picks point with maximum EI."""
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        candidates = np.random.uniform(lows, highs, size=(n_candidates, self.dim))
        ei = self.expected_improvement(candidates)
        best_idx = np.argmax(ei)
        return candidates[best_idx]

if __name__ == "__main__":
    print("--- W14: Surrogate-Based Design Optimization Sanity Check ---")
    np.random.seed(42)
    # Expensive simulator ground truth: Branin-like 1D function f(x) = (x - 2)^2 + sin(5*x)
    # Global minimum near x = 2.0
    expensive_sim = lambda x: (x[0] - 2.0)**2 + np.sin(5.0 * x[0])
    
    optimizer = SurrogateOptimizer(bounds=[(0.0, 4.0)], length_scale=0.3)
    
    # 3 initial sample evaluations
    init_x = [0.5, 2.5, 3.8]
    for x_val in init_x:
        optimizer.add_observation([x_val], expensive_sim([x_val]))
        
    print(f"Initial Best Value: {min(optimizer.y_sample):.4f}")
    
    # Run 10 Bayesian Optimization surrogate design loops
    for step in range(10):
        next_x = optimizer.suggest_next(n_candidates=1000)
        y_val = expensive_sim(next_x)
        optimizer.add_observation(next_x, y_val)
        
    best_idx = np.argmin(optimizer.y_sample)
    best_design = optimizer.X_sample[best_idx]
    best_score = optimizer.y_sample[best_idx]
    print(f"Optimized Design Point: x = {best_design[0]:.4f}")
    print(f"Optimized Simulator Score: {best_score:.4f}")
    
    assert best_score < min([expensive_sim([x]) for x in init_x]), "BO must improve upon initial design"
    assert abs(best_design[0] - 2.0) < 0.5, "Optimizer should converge toward true basin"
    print("W14 Sanity Check: PASSED cleanly.")

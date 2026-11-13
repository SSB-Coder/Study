"""
E05. Gaussian Processes (RBF Kernel Regression & Uncertainty Bands)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class GaussianProcessRegressorScratch:
    """Gaussian Process Regressor with Squared Exponential (RBF) kernel."""
    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0, noise_level: float = 1e-2):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # RBF Kernel: sigma_f^2 * exp(-||x1 - x2||^2 / (2 * l^2)) <- eq. 3.3
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return (self.sigma_f ** 2) * np.exp(-dist_sq / (2.0 * (self.length_scale ** 2)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.X_train = X
        self.y_train = y
        m = len(X)

        # K = K(X, X) + noise * I <- eq. 3.1
        K = self._kernel(X, X) + (self.noise_level ** 2) * np.eye(m)
        self.K_inv = np.linalg.pinv(K)
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = True):
        X_test = np.asarray(X_test, dtype=np.float64)
        K_star = self._kernel(self.X_train, X_test) # Shape (m_train, m_test)
        K_star_star = self._kernel(X_test, X_test) # Shape (m_test, m_test)

        # Posterior Mean: mu* = K_*^T @ K^(-1) @ y <- eq. 3.2
        mu_star = K_star.T @ (self.K_inv @ self.y_train)

        if not return_std:
            return mu_star

        # Posterior Covariance: Sigma* = K_** - K_*^T @ K^(-1) @ K_* <- eq. 3.2
        sigma_star = K_star_star - K_star.T @ (self.K_inv @ K_star)
        uncertainty = np.sqrt(np.maximum(1e-12, np.diag(sigma_star)))
        return mu_star, uncertainty

if __name__ == "__main__":
    print("--- E05: Gaussian Processes Sanity Check ---")
    np.random.seed(42)
    # Train on 6 noisy observations from a sine curve
    X_gp = np.array([[-3.0], [-1.5], [0.0], [1.0], [2.5], [3.5]])
    y_gp = np.sin(X_gp[:, 0]) + np.random.normal(0, 0.05, len(X_gp))

    gp = GaussianProcessRegressorScratch(length_scale=1.2, sigma_f=1.0, noise_level=0.05)
    gp.fit(X_gp, y_gp)

    # Predict on points right at training data vs far away
    X_query = np.array([[0.0], [10.0]]) # [0.0] has training point; [10.0] is far away
    mu, std = gp.predict(X_query, return_std=True)

    print(f"Prediction at x=0.0:  mean={mu[0]:.4f}, std={std[0]:.4f} (Close to data -> low uncertainty)")
    print(f"Prediction at x=10.0: mean={mu[1]:.4f}, std={std[1]:.4f} (Far from data -> high uncertainty)")
    assert std[0] < 0.15 # Low uncertainty near observation
    assert std[1] > 0.80 # High epistemic uncertainty far away
    print("E05 Sanity Check: PASSED cleanly.")

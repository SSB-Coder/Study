"""
C04. Gaussian Mixture Models & the EM Algorithm
From-scratch implementation in pure NumPy.
"""

import numpy as np

class GaussianMixtureModelScratch:
    """GMM clustering with full covariance matrices trained via Expectation-Maximization."""
    def __init__(self, k: int = 2, max_iter: int = 100, tol: float = 1e-4, reg_covar: float = 1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.pi = None
        self.mu = None
        self.cov = None
        self.log_likelihood_history = []

    def _multivariate_gaussian(self, X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Evaluates multivariate normal density for all samples <- eq. 3.2"""
        d = X.shape[1]
        cov_reg = cov + self.reg_covar * np.eye(d)
        
        diff = X - mu
        # Solve cov * x = diff.T for numerical stability without direct inverse
        try:
            sol = np.linalg.solve(cov_reg, diff.T)
            quad = np.sum(diff * sol.T, axis=1)
            sign, logdet = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                logdet = np.sum(np.log(np.linalg.svd(cov_reg, compute_uv=False) + 1e-12))
        except np.linalg.LinAlgError:
            return np.full(len(X), 1e-12)

        log_prob = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
        return np.exp(np.clip(log_prob, -50, 50))

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        m, d = X.shape

        # 1. Initialization
        self.pi = np.full(self.k, 1.0 / self.k)
        # Random sample centroids
        rand_idx = np.random.choice(m, self.k, replace=False)
        self.mu = X[rand_idx].copy()
        # Shared empirical covariance as starting point
        sample_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(d)
        self.cov = np.array([sample_cov.copy() for _ in range(self.k)])

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # 2. E-Step: Compute Responsibilities gamma_ik <- eq. 3.2
            densities = np.zeros((m, self.k))
            for k in range(self.k):
                densities[:, k] = self.pi[k] * self._multivariate_gaussian(X, self.mu[k], self.cov[k])
            
            density_sum = np.sum(densities, axis=1, keepdims=True) + 1e-15
            gamma = densities / density_sum

            # Evaluate Log-Likelihood <- eq. 3.1
            current_ll = np.sum(np.log(density_sum))
            self.log_likelihood_history.append(current_ll)

            if abs(current_ll - prev_ll) < self.tol:
                break
            prev_ll = current_ll

            # 3. M-Step: Update Parameters <- eq. 3.3
            N_k = np.sum(gamma, axis=0) + 1e-15
            self.pi = N_k / m
            
            for k in range(self.k):
                self.mu[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k[k]
                diff = X - self.mu[k]
                weighted_diff = gamma[:, k:k+1] * diff
                self.cov[k] = (weighted_diff.T @ diff) / N_k[k] + self.reg_covar * np.eye(d)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        densities = np.zeros((m, self.k))
        for k in range(self.k):
            densities[:, k] = self.pi[k] * self._multivariate_gaussian(X, self.mu[k], self.cov[k])
        return densities / (np.sum(densities, axis=1, keepdims=True) + 1e-15)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

if __name__ == "__main__":
    print("--- C04: GMM & EM Algorithm Sanity Check ---")
    np.random.seed(42)
    # Generate 2 elliptical clusters with distinct covariances
    cov1 = np.array([[2.0, 1.2], [1.2, 1.0]])
    cov2 = np.array([[1.0, -0.7], [-0.7, 1.5]])
    X_gmm = np.vstack([
        np.random.multivariate_normal([3, 3], cov1, size=80),
        np.random.multivariate_normal([-3, -3], cov2, size=80)
    ])
    
    gmm = GaussianMixtureModelScratch(k=2).fit(X_gmm)
    # Log-likelihood must monotonically increase over iterations
    ll = gmm.log_likelihood_history
    print(f"Initial LL: {ll[0]:.2f} -> Final LL: {ll[-1]:.2f}")
    assert ll[-1] >= ll[0]
    preds = gmm.predict(X_gmm)
    acc = max(np.mean(preds == np.array([0]*80 + [1]*80)), np.mean(preds == np.array([1]*80 + [0]*80)))
    print(f"GMM Cluster Separation Accuracy: {acc * 100:.2f}%")
    assert acc > 0.95
    print("C04 Sanity Check: PASSED cleanly.")

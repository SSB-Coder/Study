"""
E04. Variational Inference (Mean-Field CAVI for Gaussian Mixtures)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class CAVIGaussianMixture:
    """Coordinate Ascent Variational Inference for a 1D Gaussian Mixture with known unit variance."""
    def __init__(self, k: int = 2, sigma_0_sq: float = 10.0, max_iter: int = 50, tol: float = 1e-4):
        self.k = k
        self.sigma_0_sq = sigma_0_sq
        self.max_iter = max_iter
        self.tol = tol
        self.m = None # Variational means m_k
        self.s2 = None # Variational variances s_k^2
        self.gamma = None # Variational responsibilities gamma_ik

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        N = len(X)

        # 1. Initialize variational parameters
        self.m = np.random.choice(X, size=self.k, replace=False)
        self.s2 = np.ones(self.k)
        self.gamma = np.full((N, self.k), 1.0 / self.k)

        elbo_history = []

        for iteration in range(self.max_iter):
            # 2. Update q(c_i): Update Responsibilities gamma_ik <- eq. 3.2
            # E[mu_k^2] = m_k^2 + s_k^2
            e_mu = self.m
            e_mu2 = self.m ** 2 + self.s2
            
            # log gamma_ik = e_mu_k * x_i - 0.5 * e_mu2_k
            log_gamma = X[:, np.newaxis] * e_mu[np.newaxis, :] - 0.5 * e_mu2[np.newaxis, :]
            # Softmax normalize across components
            log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
            self.gamma = np.exp(log_gamma) / np.sum(np.exp(log_gamma), axis=1, keepdims=True)

            # 3. Update q(mu_k): Update m_k and s_k^2 <- eq. 3.2
            sum_gamma = np.sum(self.gamma, axis=0) # Shape (k,)
            self.s2 = 1.0 / (1.0 / self.sigma_0_sq + sum_gamma)
            self.m = self.s2 * np.sum(self.gamma * X[:, np.newaxis], axis=0)

            # 4. Compute ELBO <- eq. 3.1
            elbo = 0.0
            for k in range(self.k):
                # Prior E[log p(mu_k)]
                elbo += -0.5 * np.log(2.0 * np.pi * self.sigma_0_sq) - 0.5 * (self.m[k]**2 + self.s2[k]) / self.sigma_0_sq
                # Entropy E[-log q(mu_k)]
                elbo += 0.5 * np.log(2.0 * np.pi * np.e * self.s2[k])
            
            # Likelihood and categorical entropy
            elbo += np.sum(self.gamma * (X[:, np.newaxis] * self.m - 0.5 * (self.m**2 + self.s2)))
            elbo -= np.sum(self.gamma * np.log(np.maximum(1e-12, self.gamma)))

            elbo_history.append(elbo)
            if len(elbo_history) > 1 and abs(elbo - elbo_history[-2]) < self.tol:
                break

        return self

if __name__ == "__main__":
    print("--- E04: Variational Inference (CAVI) Sanity Check ---")
    np.random.seed(42)
    # Generate 1D mixture of two well-separated Gaussians: N(-4, 1) and N(4, 1)
    X_vi = np.concatenate([np.random.normal(-4, 1, 60), np.random.normal(4, 1, 60)])

    cavi = CAVIGaussianMixture(k=2, max_iter=30).fit(X_vi)
    sorted_means = np.sort(cavi.m)
    print(f"CAVI Estimated Cluster Means: {sorted_means} (True: [-4.0, 4.0])")
    assert np.isclose(sorted_means[0], -4.0, atol=0.5)
    assert np.isclose(sorted_means[1], 4.0, atol=0.5)
    print("E04 Sanity Check: PASSED cleanly.")

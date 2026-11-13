"""
B08. Generalized Linear Models (Poisson & Gamma via IRLS)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class GeneralizedLinearModelScratch:
    """GLM solver implementing Poisson and Gamma regressions via Fisher Scoring (IRLS)."""
    def __init__(self, family: str = "poisson", max_iter: int = 50, tol: float = 1e-6):
        self.family = family # 'poisson' (log link), 'gamma' (inverse link)
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m, d = X.shape

        # Add intercept column
        X_b = np.c_[np.ones((m, 1)), X]
        p = d + 1

        # Initialize mu
        if self.family == "poisson":
            mu = np.maximum(y, 0.1)
            eta = np.log(mu) # g(mu) = ln(mu)
        elif self.family == "gamma":
            mu = np.maximum(y, 0.1)
            eta = 1.0 / mu # g(mu) = 1 / mu
        else:
            raise ValueError(f"Unsupported family {self.family}")

        beta = np.zeros(p)

        for iteration in range(self.max_iter):
            if self.family == "poisson":
                # Poisson: link g(mu) = ln(mu), g'(mu) = 1/mu, V(mu) = mu
                # w_i = 1 / (V(mu) * [g'(mu)]^2) = 1 / (mu * 1/mu^2) = mu <- eq. 3.2
                mu = np.exp(eta)
                mu = np.clip(mu, 1e-10, 1e10)
                w = mu
                g_prime = 1.0 / mu
                z = eta + (y - mu) * g_prime
            elif self.family == "gamma":
                # Gamma: link g(mu) = 1/mu, g'(mu) = -1/mu^2, V(mu) = mu^2
                # w_i = 1 / (mu^2 * 1/mu^4) = mu^2 <- eq. 3.2
                mu = 1.0 / np.clip(eta, 1e-10, 1e10)
                w = mu ** 2
                g_prime = -1.0 / (mu ** 2)
                z = eta + (y - mu) * g_prime

            # Weighted Least Squares step: (X^T W X)^(-1) X^T W z <- eq. 3.2
            W_sqrt = np.sqrt(w)
            X_w = X_b * W_sqrt[:, np.newaxis]
            z_w = z * W_sqrt

            beta_new = np.linalg.pinv(X_w.T @ X_w) @ X_w.T @ z_w

            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break
            beta = beta_new
            eta = X_b @ beta

        self.beta = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        X_b = np.c_[np.ones((m, 1)), X]
        eta = X_b @ self.beta
        if self.family == "poisson":
            return np.exp(eta)
        elif self.family == "gamma":
            return 1.0 / np.clip(eta, 1e-10, 1e10)

if __name__ == "__main__":
    print("--- B08: Generalized Linear Models (IRLS) Sanity Check ---")
    np.random.seed(42)
    m = 200
    X_poi = np.random.uniform(0.5, 2.5, size=(m, 1))
    true_beta = np.array([0.5, 1.2]) # intercept=0.5, slope=1.2
    # Poisson count target: lambda = exp(0.5 + 1.2 * x)
    expected_lambda = np.exp(0.5 + 1.2 * X_poi[:, 0])
    y_poi = np.random.poisson(lam=expected_lambda)

    glm_poi = GeneralizedLinearModelScratch(family="poisson").fit(X_poi, y_poi)
    print(f"True Poisson Beta: {true_beta}")
    print(f"IRLS Estimated Beta: {glm_poi.beta}")
    assert np.allclose(glm_poi.beta, true_beta, atol=0.2)
    print("B08 Sanity Check: PASSED cleanly.")

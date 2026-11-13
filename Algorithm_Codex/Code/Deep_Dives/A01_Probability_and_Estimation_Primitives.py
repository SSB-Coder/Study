"""
A01. Probability & Estimation Primitives (MLE vs. MAP)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class BernoulliEstimator:
    """Estimates the parameter theta of a Bernoulli distribution via MLE and MAP."""
    def __init__(self, alpha_prior: float = 2.0, beta_prior: float = 2.0):
        # Beta(alpha, beta) prior. alpha=beta=2 represents a weak prior centered at 0.5
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.theta_mle = None
        self.theta_map = None

    def fit(self, X: np.ndarray):
        """X is an array of binary outcomes (0 or 1)."""
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        k = np.sum(X) # Number of successes
        
        # MLE: k / m  <- eq. 3.1 in report
        self.theta_mle = k / m
        
        # MAP with Beta(alpha, beta) conjugate prior <- eq. 3.2 in report
        self.theta_map = (k + self.alpha_prior - 1.0) / (m + self.alpha_prior + self.beta_prior - 2.0)
        return self

class GaussianEstimator:
    """Estimates mean and variance of a 1D Gaussian via MLE and MAP."""
    def __init__(self, mu_0: float = 0.0, sigma_0_sq: float = 1.0, known_sigma_sq: float = 1.0):
        self.mu_0 = mu_0
        self.sigma_0_sq = sigma_0_sq
        self.known_sigma_sq = known_sigma_sq
        self.mu_mle = None
        self.var_mle = None
        self.mu_map = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        sample_mean = np.mean(X)
        
        # MLE Mean and Variance <- eq. 3.3 in report
        self.mu_mle = sample_mean
        self.var_mle = np.var(X, ddof=0) # Biased MLE variance
        
        # MAP Mean under known variance and Gaussian prior N(mu_0, sigma_0_sq) <- eq. 3.4 in report
        prec_data = m / self.known_sigma_sq
        prec_prior = 1.0 / self.sigma_0_sq
        self.mu_map = (prec_data * sample_mean + prec_prior * self.mu_0) / (prec_data + prec_prior)
        return self

if __name__ == "__main__":
    print("--- A01: Probability & Estimation Primitives Sanity Check ---")
    np.random.seed(42)
    
    # 1. Test Bernoulli Estimator on biased coin (true theta = 0.7) with small sample
    true_theta = 0.7
    small_sample = np.random.binomial(n=1, p=true_theta, size=10)
    b_est = BernoulliEstimator(alpha_prior=5.0, beta_prior=5.0) # Prior believes coin is 0.5
    b_est.fit(small_sample)
    
    print(f"Sample size: {len(small_sample)}, Heads: {int(np.sum(small_sample))}")
    print(f"Bernoulli MLE theta: {b_est.theta_mle:.4f}")
    print(f"Bernoulli MAP theta: {b_est.theta_map:.4f} (regularized toward 0.5 prior)")
    assert 0.0 <= b_est.theta_mle <= 1.0
    assert 0.0 <= b_est.theta_map <= 1.0
    
    # 2. Test Gaussian Estimator with sample size scaling
    true_mu = 3.5
    true_var = 2.0
    X_samples = np.random.normal(loc=true_mu, scale=np.sqrt(true_var), size=50)
    g_est = GaussianEstimator(mu_0=0.0, sigma_0_sq=2.0, known_sigma_sq=true_var)
    g_est.fit(X_samples)
    
    print(f"Gaussian Sample Mean: {np.mean(X_samples):.4f}")
    print(f"Gaussian MLE mu: {g_est.mu_mle:.4f} | MLE var: {g_est.var_mle:.4f}")
    print(f"Gaussian MAP mu: {g_est.mu_map:.4f}")
    assert np.isclose(g_est.mu_mle, np.mean(X_samples))
    print("A01 Sanity Check: PASSED cleanly.")

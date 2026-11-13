"""
E03. Markov Chain Monte Carlo (Metropolis-Hastings & Gibbs Sampling)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class MCMCSampler:
    """Implementations of Metropolis-Hastings and Gibbs Sampling."""

    @staticmethod
    def metropolis_hastings(target_log_prob_fn, x0: np.ndarray, n_samples: int = 3000, 
                            proposal_std: float = 2.0, burn_in: int = 1000):
        """Random-walk Metropolis-Hastings for arbitrary unnormalized log-probability functions <- eq. 3.1"""
        x = np.asarray(x0, dtype=np.float64).copy()
        samples = []
        accepted = 0

        for step in range(n_samples + burn_in):
            # Symmetric Gaussian proposal: q(x' | x) = N(x, proposal_std^2 * I)
            x_prop = x + np.random.normal(0, proposal_std, size=x.shape)

            # Log-acceptance ratio: log(p(x')) - log(p(x))
            log_alpha = target_log_prob_fn(x_prop) - target_log_prob_fn(x)

            # Accept / Reject
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                x = x_prop
                if step >= burn_in:
                    accepted += 1

            if step >= burn_in:
                samples.append(x.copy())

        accept_rate = accepted / n_samples
        return np.array(samples), accept_rate

    @staticmethod
    def gibbs_bivariate_gaussian(mu: np.ndarray, rho: float, n_samples: int = 2000, burn_in: int = 500):
        """Exact Gibbs sampler for 2D Gaussian with correlation rho <- eq. 3.2"""
        # Full conditionals:
        # x1 | x2 ~ N(mu1 + rho * (x2 - mu2), 1 - rho^2)
        # x2 | x1 ~ N(mu2 + rho * (x1 - mu1), 1 - rho^2)
        cond_std = np.sqrt(1.0 - rho ** 2)
        x1, x2 = mu[0], mu[1]
        samples = []

        for step in range(n_samples + burn_in):
            # Update x1 given x2
            mean1 = mu[0] + rho * (x2 - mu[1])
            x1 = np.random.normal(mean1, cond_std)

            # Update x2 given x1
            mean2 = mu[1] + rho * (x1 - mu[0])
            x2 = np.random.normal(mean2, cond_std)

            if step >= burn_in:
                samples.append([x1, x2])

        return np.array(samples)

if __name__ == "__main__":
    print("--- E03: MCMC (Metropolis-Hastings & Gibbs) Sanity Check ---")
    np.random.seed(42)
    # 1. Test Metropolis-Hastings on 1D Mixture of Gaussians: 0.3 * N(-3, 1) + 0.7 * N(3, 1)
    def target_bimodal(x):
        d1 = np.exp(-0.5 * (x - 3.0)**2)
        d2 = np.exp(-0.5 * (x + 3.0)**2)
        return np.log(0.7 * d1 + 0.3 * d2 + 1e-15)

    samples_mh, acc_rate = MCMCSampler.metropolis_hastings(target_bimodal, x0=np.array([1.0]), n_samples=4000, proposal_std=2.2)
    sample_mean_mh = np.mean(samples_mh)
    # Theoretical mean = 0.7 * (3.0) + 0.3 * (-3.0) = 2.1 - 0.9 = 1.2
    print(f"MH Acceptance Rate: {acc_rate * 100:.1f}%")
    print(f"MH Sample Mean: {sample_mean_mh:.4f} (Theoretical: 1.2000)")
    assert 0.20 < acc_rate < 0.75
    assert np.isclose(sample_mean_mh, 1.2, atol=0.35)

    # 2. Test Gibbs Sampler on Bivariate Normal with correlation rho = 0.8
    samples_gibbs = MCMCSampler.gibbs_bivariate_gaussian(mu=np.array([0.0, 0.0]), rho=0.8, n_samples=2000)
    emp_corr = np.corrcoef(samples_gibbs[:, 0], samples_gibbs[:, 1])[0, 1]
    print(f"Gibbs Empirical Correlation: {emp_corr:.4f} (Theoretical: 0.8000)")
    assert np.isclose(emp_corr, 0.8, atol=0.05)
    print("E03 Sanity Check: PASSED cleanly.")

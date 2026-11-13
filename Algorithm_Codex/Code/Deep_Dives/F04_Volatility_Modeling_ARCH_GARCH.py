"""
F04: Volatility Modeling (ARCH and GARCH)
Pure NumPy implementation of GARCH(1,1) recursive variance computation and simulation.
"""

import numpy as np

class GARCH11:
    """GARCH(1,1) model for conditional heteroskedasticity."""
    def __init__(self, omega: float = 0.05, alpha: float = 0.1, beta: float = 0.8):
        assert alpha + beta < 1.0, "Stationarity requires alpha + beta < 1"
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.unconditional_var = omega / (1.0 - (alpha + beta))

    def compute_conditional_variances(self, returns: np.ndarray) -> np.ndarray:
        """Computes rolling sigma_t^2 given empirical shocks."""
        T = len(returns)
        sigma2 = np.zeros(T, dtype=np.float64)
        sigma2[0] = self.unconditional_var

        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * (returns[t-1] ** 2) + self.beta * sigma2[t-1]

        return sigma2

    def simulate(self, T: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Simulates returns with volatility clustering."""
        rng = np.random.default_rng(seed)
        returns = np.zeros(T)
        sigma2 = np.zeros(T)
        sigma2[0] = self.unconditional_var

        for t in range(T):
            if t > 0:
                sigma2[t] = self.omega + self.alpha * (returns[t-1] ** 2) + self.beta * sigma2[t-1]
            z = rng.normal(0, 1)
            returns[t] = np.sqrt(sigma2[t]) * z

        return returns, sigma2


if __name__ == "__main__":
    print("--- F04: GARCH(1,1) Volatility Modeling Sanity Check ---")
    garch = GARCH11(omega=0.04, alpha=0.15, beta=0.80)
    expected_unconditional = 0.04 / (1.0 - 0.95)  # 0.80
    print(f"GARCH(1,1) Long-run Unconditional Variance: {garch.unconditional_var:.4f}")
    assert np.isclose(garch.unconditional_var, 0.80)

    # Simulate 2000 steps
    returns_sim, sigma2_sim = garch.simulate(T=2000, seed=123)
    empirical_var = np.var(returns_sim)
    print(f"Empirical Sample Variance: {empirical_var:.4f} (Expected: ~0.80)")
    assert np.isclose(empirical_var, 0.80, atol=0.15)

    # Reconstruct conditional variances from simulated returns
    reconstructed_sigma2 = garch.compute_conditional_variances(returns_sim)
    np.testing.assert_allclose(sigma2_sim[10:], reconstructed_sigma2[10:], atol=1e-5)
    print("Conditional variance sequence perfectly reconstructed.")
    print("F04 Sanity Check: PASSED cleanly.")

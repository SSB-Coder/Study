"""
F02: Autoregressive & Moving Average Models (ARMA / ARIMA)
Pure NumPy implementation of AR(p) via Yule-Walker and Least Squares.
"""

import numpy as np

class AutoregressiveModel:
    """Autoregressive AR(p) model with exact least squares fitting and multi-step forecasting."""
    def __init__(self, p: int = 2):
        self.p = p
        self.phi = None
        self.c = 0.0
        self.sigma2 = 1.0

    def fit(self, y: np.ndarray):
        series = np.array(y, dtype=np.float64)
        n = len(series)
        assert n > self.p + 2, "Series too short for chosen AR lag"

        # Build design matrix X and target Y
        Y = series[self.p:]
        X = np.zeros((len(Y), self.p + 1))
        X[:, 0] = 1.0  # Intercept
        for lag in range(1, self.p + 1):
            X[:, lag] = series[self.p - lag : n - lag]

        # OLS: beta = (X^T X)^(-1) X^T Y
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        self.c = float(beta[0])
        self.phi = beta[1:]

        residuals = Y - X @ beta
        self.sigma2 = float(np.var(residuals))

    def predict(self, history: np.ndarray, steps: int = 3) -> np.ndarray:
        curr_hist = list(history[-self.p:])
        forecasts = []
        for _ in range(steps):
            pred = self.c
            for lag in range(self.p):
                pred += self.phi[lag] * curr_hist[-1 - lag]
            forecasts.append(pred)
            curr_hist.append(pred)
        return np.array(forecasts)


if __name__ == "__main__":
    print("--- F02: Autoregressive AR(p) Sanity Check ---")
    # Generate synthetic AR(1) process: y_t = 2.0 + 0.7 * y_{t-1} + e_t
    np.random.seed(42)
    n = 1000
    true_phi = 0.7
    true_c = 2.0
    y_sim = np.zeros(n)
    for t in range(1, n):
        y_sim[t] = true_c + true_phi * y_sim[t-1] + np.random.normal(0, 0.5)

    ar = AutoregressiveModel(p=1)
    ar.fit(y_sim)

    print(f"True params: c={true_c}, phi={true_phi}")
    print(f"Fitted params: c={ar.c:.4f}, phi={ar.phi[0]:.4f}, sigma2={ar.sigma2:.4f}")

    assert np.isclose(ar.phi[0], true_phi, atol=0.08), f"Fitted AR(1) coefficient should match {true_phi}"
    assert np.isclose(ar.c, true_c, atol=0.3), f"Fitted intercept should match {true_c}"

    # Forecast
    fut = ar.predict(y_sim, steps=3)
    assert len(fut) == 3
    print("F02 Sanity Check: PASSED cleanly.")

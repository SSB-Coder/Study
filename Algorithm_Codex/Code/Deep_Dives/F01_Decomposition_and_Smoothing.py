"""
F01: Classical Time Series Decomposition & Exponential Smoothing (Holt-Winters)
Pure NumPy implementation of Holt-Winters Triple Exponential Smoothing.
"""

import numpy as np

class HoltWintersAdditive:
    """Holt-Winters Additive Triple Exponential Smoothing."""
    def __init__(self, season_length: int = 4, alpha: float = 0.2, beta: float = 0.1, gamma: float = 0.3):
        self.m = season_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit_predict(self, series: np.ndarray, horizon: int = 4) -> np.ndarray:
        y = np.array(series, dtype=np.float64)
        n = len(y)
        assert n >= 2 * self.m, "Series must contain at least 2 full seasonal cycles"

        # Initialize seasonals: average of first cycle differences
        seasonals = np.zeros(self.m)
        cycle1_mean = np.mean(y[:self.m])
        for i in range(self.m):
            seasonals[i] = y[i] - cycle1_mean

        # Initialize level and trend
        level = y[0] - seasonals[0]
        trend = (y[self.m] - y[0]) / self.m

        # Filter forward
        fitted = np.zeros(n)
        for t in range(n):
            s_idx = t % self.m
            fitted[t] = level + trend + seasonals[s_idx]

            # Update
            new_level = self.alpha * (y[t] - seasonals[s_idx]) + (1.0 - self.alpha) * (level + trend)
            new_trend = self.beta * (new_level - level) + (1.0 - self.beta) * trend
            seasonals[s_idx] = self.gamma * (y[t] - new_level) + (1.0 - self.gamma) * seasonals[s_idx]

            level = new_level
            trend = new_trend

        # Forecast future steps
        forecast = np.zeros(horizon)
        for h in range(1, horizon + 1):
            s_idx = (n + h - 1) % self.m
            forecast[h - 1] = level + h * trend + seasonals[s_idx]

        return np.concatenate([fitted, forecast])


if __name__ == "__main__":
    print("--- F01: Holt-Winters Smoothing Sanity Check ---")
    # Synthetic quarterly series with linear trend + quarter seasonality:
    # Trend: 2.0 * t, Seasonality: [+5, -2, -5, +2]
    np.random.seed(42)
    t_steps = 16  # 4 full cycles
    seasonal_true = np.tile([5.0, -2.0, -5.0, 2.0], 4)
    trend_true = 2.0 * np.arange(t_steps)
    y_synth = trend_true + seasonal_true

    hw = HoltWintersAdditive(season_length=4, alpha=0.3, beta=0.1, gamma=0.3)
    preds = hw.fit_predict(y_synth, horizon=4)

    # Sanity check: verify fitted series captures trend and seasonality
    fitted_vals = preds[:t_steps]
    rmse = np.sqrt(np.mean((fitted_vals[4:] - y_synth[4:]) ** 2))
    print(f"Fitted In-Sample RMSE (excluding cycle 1 warmup): {rmse:.4f}")
    assert rmse < 2.5, f"Holt-Winters should fit seasonal trend cleanly, got RMSE {rmse}"

    # Verify future forecast preserves seasonality
    forecast_4 = preds[t_steps:]
    print(f"4-Quarter Out-of-Sample Forecast: {forecast_4.round(2)}")
    # Q1 of cycle 5 should have high seasonal bump (+5) compared to Q3 (-5)
    assert forecast_4[0] > forecast_4[2], "Forecast must preserve seasonal phase (Q1 > Q3)"
    print("F01 Sanity Check: PASSED cleanly.")

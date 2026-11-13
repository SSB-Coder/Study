"""
J07: Causal Inference (SCMs, Propensity Score Matching, Difference-in-Differences)
Pure NumPy implementation of causal effect estimation on observational data.
Addresses confounding bias via Propensity Score Matching and panel DiD.
"""

import numpy as np

class PropensityScoreMatcher:
    """Estimates Average Treatment Effect (ATE) via 1-NN Propensity Score Matching."""
    def __init__(self, lr: float = 0.05, max_iter: int = 300):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0

    def fit_propensity(self, X: np.ndarray, T: np.ndarray):
        """Fits logistic regression model P(T=1 | X)."""
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.max_iter):
            logits = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            grad_w = (X.T @ (p - T)) / n
            grad_b = np.mean(p - T)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        """Computes ATE by matching each treated unit to closest control unit on propensity score."""
        scores = self.predict_propensity(X)
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        treatment_effects = []
        for i in treated_idx:
            dist = np.abs(scores[control_idx] - scores[i])
            match_idx = control_idx[np.argmin(dist)]
            # Counterfactual: Y(1) - Y(0)
            treatment_effects.append(Y[i] - Y[match_idx])

        return float(np.mean(treatment_effects))


def difference_in_differences(Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated) -> float:
    """
    Computes canonical 2x2 Difference-in-Differences treatment effect:
    delta_DiD = (E[Y_post | Treated] - E[Y_pre | Treated]) - (E[Y_post | Control] - E[Y_pre | Control])
    """
    delta_treated = np.mean(Y_post_treated) - np.mean(Y_pre_treated)
    delta_control = np.mean(Y_post_control) - np.mean(Y_pre_control)
    return float(delta_treated - delta_control)


if __name__ == "__main__":
    print("--- J07: Causal Inference Sanity Check ---")
    np.random.seed(42)

    N = 500
    # Confounder X: e.g. healthiness / age
    X = np.random.randn(N, 1)

    # Treatment assignment T is confounded by X: healthier people take treatment more often
    p_true = 1.0 / (1.0 + np.exp(-(1.5 * X.squeeze())))
    T = (np.random.uniform(0, 1, size=N) < p_true).astype(int)

    # True Causal Model: Y = 2.0 * T + 1.2 * X + noise (True ATE = 2.0)
    Y = 2.0 * T + 1.2 * X.squeeze() + np.random.randn(N) * 0.5

    # Naive observational difference (severely biased upward by positive confounder!)
    naive_ate = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))
    print(f"Naive Observational ATE (Confounded): {naive_ate:.3f} (True = 2.000)")

    # 1. Propensity Score Matching
    psm = PropensityScoreMatcher(lr=0.1, max_iter=400)
    psm.fit_propensity(X, T)
    psm_ate = psm.estimate_ate(X, T, Y)
    print(f"Propensity Score Matching ATE:        {psm_ate:.3f}")
    assert abs(psm_ate - 2.0) < abs(naive_ate - 2.0), "PSM must debias observational estimate toward truth"

    # 2. Difference-in-Differences
    # Panel setup: common secular time trend +0.8, treatment effect +3.0 in post period
    N_group = 200
    pre_control = np.random.randn(N_group) + 5.0
    post_control = pre_control + 0.8 + np.random.randn(N_group) * 0.2

    pre_treated = np.random.randn(N_group) + 3.0 # Different baseline
    post_treated = pre_treated + 0.8 + 3.0 + np.random.randn(N_group) * 0.2 # Trend + Treatment

    did_ate = difference_in_differences(pre_control, post_control, pre_treated, post_treated)
    print(f"Difference-in-Differences ATE:        {did_ate:.3f} (True = 3.000)")
    assert np.isclose(did_ate, 3.0, atol=0.2), "DiD must recover true panel treatment effect"

    print("J07 Sanity Check: PASSED cleanly.")

"""
F03: Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)
Pure NumPy implementation of non-parametric Kaplan-Meier estimator
and semi-parametric Cox Proportional Hazards model with partial likelihood.
"""

import numpy as np

class KaplanMeierEstimator:
    """
    Kaplan-Meier product-limit estimator for right-censored time-to-event data.
    S(t) = prod_{t_i <= t} (1 - d_i / n_i)
    """
    def __init__(self):
        self.timeline = None
        self.survival_probs = None
        self.events_at_time = None
        self.at_risk_at_time = None

    def fit(self, durations: np.ndarray, events: np.ndarray):
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=int)

        unique_times = np.unique(durations[events == 1])
        unique_times.sort()

        self.timeline = [0.0]
        self.survival_probs = [1.0]
        self.events_at_time = [0]
        self.at_risk_at_time = [len(durations)]

        curr_prob = 1.0
        for t in unique_times:
            n_at_risk = np.sum(durations >= t)
            d_events = np.sum((durations == t) & (events == 1))
            if n_at_risk > 0:
                step = 1.0 - (d_events / n_at_risk)
                curr_prob *= step

            self.timeline.append(float(t))
            self.survival_probs.append(float(curr_prob))
            self.events_at_time.append(int(d_events))
            self.at_risk_at_time.append(int(n_at_risk))

        self.timeline = np.array(self.timeline)
        self.survival_probs = np.array(self.survival_probs)

    def predict_survival(self, t: float) -> float:
        idx = np.searchsorted(self.timeline, t, side='right') - 1
        return float(self.survival_probs[idx])


class CoxProportionalHazards:
    """
    Cox Proportional Hazards model: h(t | x) = h_0(t) * exp(x^T beta).
    Fits coefficients beta by maximizing Cox's partial log-likelihood via gradient ascent.
    """
    def __init__(self, lr: float = 0.01, max_iter: int = 200, l2_reg: float = 1e-4):
        self.lr = lr
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.beta = None

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=int)

        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)

        # Sort descending by duration for risk set calculations
        order = np.argsort(durations)
        X_sorted = X[order]
        dur_sorted = durations[order]
        ev_sorted = events[order]

        for it in range(self.max_iter):
            # Compute hazard ratios: theta_i = exp(X_i @ beta)
            theta = np.exp(np.clip(X_sorted @ self.beta, -20.0, 20.0))

            grad = np.zeros(n_features)
            for i in range(n_samples):
                if ev_sorted[i] == 1:
                    risk_mask = dur_sorted >= dur_sorted[i]
                    denom = np.sum(theta[risk_mask])
                    numer = np.sum(theta[risk_mask, None] * X_sorted[risk_mask], axis=0)
                    grad += X_sorted[i] - (numer / denom)

            # L2 penalty
            grad -= self.l2_reg * self.beta
            self.beta += self.lr * grad

            if np.linalg.norm(grad) < 1e-5:
                break

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(X @ self.beta, -20.0, 20.0))


if __name__ == "__main__":
    print("--- F03: Survival Analysis (Kaplan-Meier & Cox PH) Sanity Check ---")
    np.random.seed(42)

    # 1. Test Kaplan-Meier
    # 5 patients: 2 died at t=2,3; 1 censored at t=3; 1 died at t=4; 1 censored at t=5
    durations = np.array([2.0, 3.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 0, 1, 0])

    km = KaplanMeierEstimator()
    km.fit(durations, events)

    s0 = km.predict_survival(0.0)
    s2 = km.predict_survival(2.5) # after t=2: 1 - 1/5 = 0.80
    s3 = km.predict_survival(3.5) # after t=3: 0.8 * (1 - 1/4) = 0.60
    s4 = km.predict_survival(4.5) # after t=4: 0.6 * (1 - 1/2) = 0.30

    print(f"KM S(0) = {s0:.3f} | S(2.5) = {s2:.3f} | S(3.5) = {s3:.3f} | S(4.5) = {s4:.3f}")
    assert np.isclose(s0, 1.0), "KM S(0) must be 1.0"
    assert np.isclose(s2, 0.80), f"Expected S(2)=0.80, got {s2}"
    assert np.isclose(s3, 0.60), f"Expected S(3)=0.60, got {s3}"
    assert np.isclose(s4, 0.30), f"Expected S(4)=0.30, got {s4}"

    # 2. Test Cox Proportional Hazards
    n = 100
    X = np.random.randn(n, 2)
    true_beta = np.array([1.2, -0.8])
    hazard = np.exp(X @ true_beta)
    dur_sim = -np.log(np.random.uniform(0.01, 1.0, size=n)) / hazard
    censor_time = np.percentile(dur_sim, 75)
    ev_sim = (dur_sim <= censor_time).astype(int)
    dur_sim = np.minimum(dur_sim, censor_time)

    cox = CoxProportionalHazards(lr=0.005, max_iter=300)
    cox.fit(X, dur_sim, ev_sim)
    print(f"Fitted Cox beta: {cox.beta.round(3)} (True signs: [+, -])")
    assert cox.beta[0] > 0 and cox.beta[1] < 0, "Cox coefficients must match hazard effect signs"

    print("F03 Sanity Check: PASSED cleanly.")

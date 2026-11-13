"""
B09. Probability Calibration (Platt Scaling & Isotonic PAVA)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PlattScalingScratch:
    """Platt Scaling: Sigmoid calibration via regularized logistic regression."""
    def __init__(self, lr: float = 0.05, max_iter: int = 1500):
        self.lr = lr
        self.max_iter = max_iter
        self.w = 1.0
        self.b = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = len(y)

        # Target probability smoothing <- eq. 3.1
        n_pos = np.sum(y == 1)
        n_neg = m - n_pos
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(y == 1, t_pos, t_neg)

        # Standard logistic regression: p = sigma(w * scores + b)
        w, b = 0.1, 0.0

        for _ in range(self.max_iter):
            z = np.clip(w * scores + b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - targets
            dw = np.mean(err * scores)
            db = np.mean(err)
            w -= self.lr * dw
            b -= self.lr * db

        self.w = w
        self.b = b
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        z = np.clip(self.w * scores + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

class IsotonicRegressionScratch:
    """Isotonic Calibration via Pool Adjacent Violators Algorithm (PAVA)."""
    def __init__(self):
        self.x_knots = None
        self.y_knots = None

    def fit(self, scores: np.ndarray, y: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Sort by scores
        order = np.argsort(scores)
        x_sorted = scores[order]
        y_sorted = y[order]

        # PAVA implementation <- eq. 3.2
        blocks = [] # (weight, sum_y, start_idx, end_idx)
        for i in range(len(y_sorted)):
            blocks.append([1.0, float(y_sorted[i]), i, i])
            # Backtrack and pool if adjacent violators exist
            while len(blocks) >= 2:
                prev_val = blocks[-2][1] / blocks[-2][0]
                curr_val = blocks[-1][1] / blocks[-1][0]
                if prev_val > curr_val:
                    # Pool
                    pooled = [
                        blocks[-2][0] + blocks[-1][0],
                        blocks[-2][1] + blocks[-1][1],
                        blocks[-2][2],
                        blocks[-1][3]
                    ]
                    blocks.pop()
                    blocks.pop()
                    blocks.append(pooled)
                else:
                    break

        # Reconstruct monotonic step curve
        calibrated_y = np.zeros(len(y_sorted))
        for w, sum_y, start, end in blocks:
            calibrated_y[start:end + 1] = sum_y / w

        self.x_knots = x_sorted
        self.y_knots = calibrated_y
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        # Linear interpolation over isotonic knots
        return np.interp(scores, self.x_knots, self.y_knots, left=0.0, right=1.0)

if __name__ == "__main__":
    print("--- B09: Probability Calibration Sanity Check ---")
    np.random.seed(42)
    m = 100
    # True probabilities: sigmoid(2 * x)
    x_synth = np.linspace(-3, 3, m)
    true_probs = 1.0 / (1.0 + np.exp(-2.0 * x_synth))
    y_synth = (np.random.rand(m) < true_probs).astype(int)
    
    # Simulate uncalibrated margin scores (distorted by cubic non-linearity)
    raw_scores = x_synth ** 3
    # Normalize uncalibrated to [0, 1] for baseline brier score
    raw_norm = (raw_scores - np.min(raw_scores)) / np.ptp(raw_scores)
    brier_before = np.mean((raw_norm - y_synth) ** 2)

    # Fit Platt Scaling
    platt = PlattScalingScratch().fit(raw_scores, y_synth)
    platt_cal = platt.predict_proba(raw_scores)
    brier_platt = np.mean((platt_cal - y_synth) ** 2)

    # Fit Isotonic Regression (PAVA)
    isotonic = IsotonicRegressionScratch().fit(raw_scores, y_synth)
    pava_cal = isotonic.predict_proba(raw_scores)
    brier_pava = np.mean((pava_cal - y_synth) ** 2)

    print(f"Uncalibrated Brier Score:   {brier_before:.4f}")
    print(f"Platt Calibrated Brier:     {brier_platt:.4f}")
    print(f"Isotonic PAVA Brier:        {brier_pava:.4f}")
    assert brier_platt < brier_before
    assert brier_pava < brier_before
    # Monotonicity check on isotonic
    assert np.all(np.diff(isotonic.y_knots) >= -1e-12)
    print("B09 Sanity Check: PASSED cleanly.")

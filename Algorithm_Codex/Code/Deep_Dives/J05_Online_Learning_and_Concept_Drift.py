"""
J05: Online Learning and Concept Drift
Pure NumPy implementation of streaming online Perceptron classification
paired with the ADWIN (Adaptive Windowing) statistical concept drift detector.
"""

import numpy as np

class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) concept drift detector.
    Maintains a sliding window of recent values and detects significant mean shifts.
    """
    def __init__(self, delta: float = 0.05, max_window: int = 200, min_subwindow: int = 20):
        self.delta = delta
        self.max_window = max_window
        self.min_subwindow = min_subwindow
        self.window = []

    def update(self, val: float) -> bool:
        """Adds a new stream observation and tests for concept drift."""
        self.window.append(float(val))
        if len(self.window) > self.max_window:
            self.window.pop(0)

        n = len(self.window)
        if n < 2 * self.min_subwindow:
            return False

        # Test cut points
        for i in range(self.min_subwindow, n - self.min_subwindow + 1):
            w0 = np.array(self.window[:i])
            w1 = np.array(self.window[i:])

            n0, n1 = len(w0), len(w1)
            mu0, mu1 = np.mean(w0), np.mean(w1)

            # Harmonic mean of sample sizes
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            # Hoeffding bound threshold
            eps_cut = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 * n / self.delta))

            if abs(mu0 - mu1) > eps_cut:
                # Drift detected! Truncate older window
                self.window = self.window[i:]
                return True
        return False


class StreamingClassifier:
    """Online linear classifier using Passive-Aggressive updates."""
    def __init__(self, n_features: int, C: float = 1.0):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.C = C

    def predict(self, x: np.ndarray) -> int:
        margin = np.dot(self.w, x) + self.b
        return 1 if margin >= 0.0 else -1

    def update(self, x: np.ndarray, y: int):
        """
        y in {-1, +1}. Passive-aggressive update on hinge loss:
        tau = min(C, loss / (||x||^2 + 1))
        """
        margin = y * (np.dot(self.w, x) + self.b)
        loss = max(0.0, 1.0 - margin)

        if loss > 0.0:
            norm_sq = np.sum(x ** 2) + 1.0
            tau = min(self.C, loss / norm_sq)
            self.w += tau * y * x
            self.b += tau * y

    def reset(self):
        self.w.fill(0.0)
        self.b = 0.0


if __name__ == "__main__":
    print("--- J05: Online Learning & Concept Drift Sanity Check ---")
    np.random.seed(42)

    n_features = 2
    n_stream = 600

    clf = StreamingClassifier(n_features=n_features, C=0.5)
    adwin = ADWINDetector(delta=0.01, max_window=150, min_subwindow=15)

    drift_points = []
    errors = []

    # Stream Concept 1: x0 + x1 > 0 -> +1 (steps 0 to 299)
    # Stream Concept 2: x0 + x1 < 0 -> +1 (steps 300 to 599, abrupt inversion!)
    for t in range(n_stream):
        x = np.random.randn(n_features)
        if t < 300:
            y = 1 if (x[0] + x[1]) >= 0 else -1
        else:
            y = 1 if (x[0] + x[1]) < 0 else -1

        pred = clf.predict(x)
        err = 1 if pred != y else 0
        errors.append(err)

        # Update drift detector on error stream
        drift = adwin.update(err)
        if drift:
            drift_points.append(t)
            # Re-adapt or reset
            clf.reset()

        clf.update(x, y)

    print(f"Detected Concept Drift Points: {drift_points}")
    # Drift occurred at t=300; ADWIN should detect it shortly after (300-360)
    detected_around_300 = any(290 <= dp <= 360 for dp in drift_points)
    assert detected_around_300, f"ADWIN must detect concept drift near t=300, got {drift_points}"

    # Accuracy in final 100 steps after adaptation should be high (> 80%)
    final_acc = 1.0 - np.mean(errors[-100:])
    print(f"Post-Drift Re-adaptation Accuracy: {final_acc * 100:.1f}%")
    assert final_acc >= 0.80, f"Online model must recover accuracy after drift, got {final_acc * 100:.1f}%"

    print("J05 Sanity Check: PASSED cleanly.")

"""
D02. AdaBoost (Adaptive Boosting with Decision Stumps)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class DecisionStump:
    """A 1-split weak decision tree classifier."""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1 # 1 if x <= thresh is -1, -1 otherwise

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray):
        m, d = X.shape
        min_err = np.inf

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                for polarity in [1, -1]:
                    preds = np.ones(m)
                    if polarity == 1:
                        preds[X[:, j] <= t] = -1
                    else:
                        preds[X[:, j] > t] = -1

                    # Weighted error <- eq. 3.2
                    err = np.sum(sample_weights[y != preds])
                    if err < min_err:
                        min_err = err
                        self.feature_idx = j
                        self.threshold = t
                        self.polarity = polarity

        return min_err

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = len(X)
        preds = np.ones(m)
        if self.polarity == 1:
            preds[X[:, self.feature_idx] <= self.threshold] = -1
        else:
            preds[X[:, self.feature_idx] > self.threshold] = -1
        return preds

class AdaBoostScratch:
    """AdaBoost.M1 classifier built from mathematical first principles."""
    def __init__(self, n_estimators: int = 40):
        self.n_estimators = n_estimators
        self.stumps_ = []
        self.alphas_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Ensure labels are -1 and +1
        y = np.where(y <= 0, -1.0, 1.0)
        m = len(X)

        # Initialize uniform sample weights <- eq. 3.1
        w = np.full(m, 1.0 / m)

        self.stumps_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            err = stump.fit(X, y, sample_weights=w)
            err = np.clip(err, 1e-10, 1.0 - 1e-10)

            # Learner weight alpha_m = 0.5 * ln((1 - err) / err) <- eq. 3.2
            alpha = 0.5 * np.log((1.0 - err) / err)

            preds = stump.predict(X)
            # Update sample weights: w * exp(-alpha * y * preds) <- eq. 3.3
            w *= np.exp(-alpha * y * preds)
            w /= np.sum(w) # Normalize

            self.stumps_.append(stump)
            self.alphas_.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        stump_preds = np.array([alpha * stump.predict(X) for stump, alpha in zip(self.stumps_, self.alphas_)])
        # H(x) = sign(sum(alpha_m * h_m(x)))
        strong_pred = np.sum(stump_preds, axis=0)
        return np.where(strong_pred >= 0, 1, 0)

if __name__ == "__main__":
    print("--- D02: AdaBoost Sanity Check ---")
    np.random.seed(42)
    # Linearly non-separable concentric ring dataset
    m = 100
    r = np.random.uniform(0.5, 4.0, m)
    theta = np.random.uniform(0, 2*np.pi, m)
    X_ada = np.c_[r * np.cos(theta), r * np.sin(theta)]
    y_ada = (r > 2.2).astype(int)

    ada = AdaBoostScratch(n_estimators=40).fit(X_ada, y_ada)
    preds = ada.predict(X_ada)
    acc = np.mean(preds == y_ada)
    print(f"AdaBoost Accuracy on Concentric Rings: {acc * 100:.2f}%")
    assert acc > 0.85 # Stumps combine to carve bounding box around inner ring
    print("D02 Sanity Check: PASSED cleanly.")

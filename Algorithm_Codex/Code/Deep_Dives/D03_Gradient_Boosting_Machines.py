"""
D03. Gradient Boosting Machines (Residual Fitting & Shrinkage)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class RegressionTreeStump:
    """Minimal regression tree fitting pseudo-residuals."""
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.tree = None

    def _build(self, X: np.ndarray, r: np.ndarray, depth: int):
        m, d = X.shape
        if depth >= self.max_depth or m <= 4:
            return float(np.mean(r))

        best_mse = np.inf
        best_split = None

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                mask = X[:, j] <= t
                if np.sum(mask) == 0 or np.sum(mask) == m:
                    continue
                mse = np.sum((r[mask] - np.mean(r[mask])) ** 2) + np.sum((r[~mask] - np.mean(r[~mask])) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_split = (j, t)

        if best_split is None:
            return float(np.mean(r))

        j, t = best_split
        mask = X[:, j] <= t
        left = self._build(X[mask], r[mask], depth + 1)
        right = self._build(X[~mask], r[~mask], depth + 1)
        return (j, t, left, right)

    def fit(self, X: np.ndarray, r: np.ndarray):
        self.tree = self._build(X, r, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node):
        if not isinstance(node, tuple):
            return node
        j, t, left, right = node
        if x[j] <= t:
            return self._predict_one(x, left)
        return self._predict_one(x, right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.tree) for row in X])

class GradientBoostingRegressorScratch:
    """Gradient Boosting for regression with shrinkage learning rate."""
    def __init__(self, n_estimators: int = 25, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees_ = []
        self.f0_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = len(y)

        # 1. Base prediction is target mean <- eq. 3.2
        self.f0_ = float(np.mean(y))
        F = np.full(m, self.f0_)
        self.trees_ = []

        for _ in range(self.n_estimators):
            # 2. Compute negative gradient pseudo-residuals r = y - F <- eq. 3.2
            residuals = y - F

            # 3. Fit regression tree to residuals
            tree = RegressionTreeStump(max_depth=self.max_depth).fit(X, residuals)
            self.trees_.append(tree)

            # 4. Update ensemble predictions with shrinkage lr
            F += self.lr * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.f0_)
        for tree in self.trees_:
            F += self.lr * tree.predict(X)
        return F

if __name__ == "__main__":
    print("--- D03: Gradient Boosting Machines Sanity Check ---")
    np.random.seed(42)
    m = 100
    # Non-linear quadratic function
    X_gb = np.linspace(-3, 3, m).reshape(-1, 1)
    y_gb = X_gb[:, 0] ** 2 + np.random.normal(0, 0.2, m)

    gbm = GradientBoostingRegressorScratch(n_estimators=30, lr=0.1, max_depth=3).fit(X_gb, y_gb)
    preds = gbm.predict(X_gb)
    mse = np.mean((preds - y_gb) ** 2)
    print(f"Gradient Boosting MSE on Quadratic Curve: {mse:.4f}")
    assert mse < 0.20 # Tree ensemble should fit quadratic curve closely
    print("D03 Sanity Check: PASSED cleanly.")

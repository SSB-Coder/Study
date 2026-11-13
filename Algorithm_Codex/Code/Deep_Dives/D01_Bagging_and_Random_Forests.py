"""
D01. Bagging & Random Forests (Bootstrap Aggregation over CART trees & OOB Error)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class DecisionTreeStumpOrDeep:
    """Lightweight CART tree supporting random feature subsampling."""
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_features: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        m, d = X.shape
        # Feature subsampling <- eq. 3.1
        feat_indices = np.random.choice(d, size=self.max_features, replace=False) if self.max_features else np.arange(d)
        
        best_gini = 1.0
        best_split = None

        def gini(sub_y):
            if len(sub_y) == 0: return 0.0
            _, counts = np.unique(sub_y, return_counts=True)
            p = counts / len(sub_y)
            return 1.0 - np.sum(p ** 2)

        for feat in feat_indices:
            threshs = np.unique(X[:, feat])
            for t in threshs:
                mask = X[:, feat] <= t
                g = (np.sum(mask) * gini(y[mask]) + np.sum(~mask) * gini(y[~mask])) / m
                if g < best_gini:
                    best_gini = g
                    best_split = (feat, t)

        return best_split

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        m = len(y)
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or m < self.min_samples_split:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]

        split = self._best_split(X, y)
        if split is None:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]

        feat, thresh = split
        left_mask = X[:, feat] <= thresh
        if np.sum(left_mask) == 0 or np.sum(left_mask) == m:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return (feat, thresh, left, right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build(X, y, depth=0)
        return self

    def _pred_one(self, x: np.ndarray, node):
        if not isinstance(node, tuple):
            return node
        feat, thresh, left, right = node
        if x[feat] <= thresh:
            return self._pred_one(x, left)
        return self._pred_one(x, right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._pred_one(row, self.tree) for row in X])

class RandomForestScratch:
    """Random Forest classifier with Out-Of-Bag (OOB) error estimation."""
    def __init__(self, n_estimators: int = 15, max_depth: int = 5, max_features: str = "sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees_ = []
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        m, d = X.shape

        # Set max_features
        m_try = int(np.sqrt(d)) if self.max_features == "sqrt" else d
        m_try = max(1, m_try)

        self.trees_ = []
        oob_votes = [[] for _ in range(m)]

        for _ in range(self.n_estimators):
            # Bootstrap sample <- eq. 3.1
            boot_idx = np.random.choice(m, size=m, replace=True)
            oob_idx = np.setdiff1d(np.arange(m), np.unique(boot_idx))

            tree = DecisionTreeStumpOrDeep(max_depth=self.max_depth, max_features=m_try)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees_.append(tree)

            # Record OOB predictions <- eq. 3.2
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for idx, p in zip(oob_idx, preds):
                    oob_votes[idx].append(p)

        # Compute OOB Accuracy Score
        correct = 0
        total_eval = 0
        for i in range(m):
            if len(oob_votes[i]) > 0:
                vals, counts = np.unique(oob_votes[i], return_counts=True)
                maj_vote = vals[np.argmax(counts)]
                if maj_vote == y[i]:
                    correct += 1
                total_eval += 1

        self.oob_score_ = correct / total_eval if total_eval > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(X) for tree in self.trees_]) # Shape (M, N)
        # Majority vote
        final_preds = []
        for i in range(len(X)):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds)

if __name__ == "__main__":
    print("--- D01: Random Forests Sanity Check ---")
    np.random.seed(42)
    # Synthetic classification with 6 features
    m = 120
    X_rf = np.random.randn(m, 6)
    y_rf = (X_rf[:, 0] + X_rf[:, 1] * 1.5 - X_rf[:, 2] > 0).astype(int)

    rf = RandomForestScratch(n_estimators=15, max_depth=4).fit(X_rf, y_rf)
    train_acc = np.mean(rf.predict(X_rf) == y_rf)
    print(f"Random Forest Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Out-of-Bag (OOB) Score:          {rf.oob_score_ * 100:.2f}%")
    assert train_acc > 0.85
    assert rf.oob_score_ > 0.70
    print("D01 Sanity Check: PASSED cleanly.")

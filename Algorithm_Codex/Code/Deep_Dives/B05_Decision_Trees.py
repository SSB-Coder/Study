"""
B05. Decision Trees (CART - Gini/Entropy Splitting & Pruning)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class TreeNode:
    """Node in a CART decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # Index of split feature
        self.threshold = threshold # Threshold value for split
        self.left = left # Left child
        self.right = right # Right child
        self.value = value # Prediction value if leaf node

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTreeScratch:
    """CART classification tree supporting Gini impurity and Information Gain."""
    def __init__(self, criterion: str = "gini", max_depth: int = 10, min_samples_split: int = 2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == "gini":
            # Gini: 1 - sum(p_k^2) <- eq. 3.2
            return 1.0 - np.sum(probs ** 2)
        else:
            # Entropy: -sum(p_k * log(p_k)) <- eq. 3.2
            return -np.sum(probs * np.log2(probs + 1e-15))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1.0
        split_feat, split_thresh = None, None
        current_impurity = self._impurity(y)
        m, d = X.shape

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Impurity reduction <- eq. 3.1
                w_left = np.sum(left_mask) / m
                w_right = np.sum(right_mask) / m
                gain = current_impurity - (w_left * self._impurity(y[left_mask]) + 
                                           w_right * self._impurity(y[right_mask]))

                if gain > best_gain:
                    best_gain = gain
                    split_feat = j
                    split_thresh = t

        return split_feat, split_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        m, d = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or m < self.min_samples_split:
            vals, counts = np.unique(y, return_counts=True)
            leaf_val = vals[np.argmax(counts)]
            return TreeNode(value=leaf_val)

        feat, thresh, gain = self._best_split(X, y)
        if gain <= 0.0 or feat is None:
            vals, counts = np.unique(y, return_counts=True)
            return TreeNode(value=vals[np.argmax(counts)])

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_row(self, x: np.ndarray, node: TreeNode):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(row, self.root) for row in X])

if __name__ == "__main__":
    print("--- B05: Decision Trees (CART) Sanity Check ---")
    np.random.seed(42)
    # 4-quadrant non-linear dataset
    X_quad = np.vstack([
        np.random.randn(30, 2) + np.array([-2, -2]),
        np.random.randn(30, 2) + np.array([2, -2]),
        np.random.randn(30, 2) + np.array([-2, 2]),
        np.random.randn(30, 2) + np.array([2, 2])
    ])
    y_quad = np.array([0]*30 + [1]*30 + [1]*30 + [0]*30)

    tree = DecisionTreeScratch(criterion="gini", max_depth=5).fit(X_quad, y_quad)
    preds = tree.predict(X_quad)
    acc = np.mean(preds == y_quad)
    print(f"CART on 4-Quadrant Dataset Accuracy: {acc * 100:.2f}%")
    assert acc > 0.90 # Tree solves 4-quadrant non-linear problem cleanly
    print("B05 Sanity Check: PASSED cleanly.")

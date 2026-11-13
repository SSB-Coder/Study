"""
J08: Anomaly Detection (Isolation Forest)
Pure NumPy implementation of Liu, Ting, & Zhou (ICDM 2008).
Ensemble of random isolation trees isolating anomalies near root nodes.
"""

import numpy as np

class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feature=None, split_value=None, size=0):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size
        self.is_leaf = (left is None and right is None)


class IsolationTree:
    """Single isolation tree recursively partitioning feature space with random cuts."""
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, depth: int = 0) -> IsolationTreeNode:
        n, d = X.shape
        if depth >= self.max_depth or n <= 1:
            return IsolationTreeNode(size=n)

        # Select random feature
        feat = np.random.randint(0, d)
        f_min, f_max = np.min(X[:, feat]), np.max(X[:, feat])

        if np.isclose(f_min, f_max):
            return IsolationTreeNode(size=n)

        # Select random split value uniformly between min and max
        split_val = np.random.uniform(f_min, f_max)
        left_mask = X[:, feat] < split_val
        right_mask = ~left_mask

        left_child = self.fit(X[left_mask], depth + 1)
        right_child = self.fit(X[right_mask], depth + 1)

        return IsolationTreeNode(
            left=left_child, right=right_child,
            split_feature=feat, split_value=split_val, size=n
        )

    def path_length(self, x: np.ndarray, node: IsolationTreeNode, current_depth: int = 0) -> float:
        if node.is_leaf:
            # Average path length correction for unbuilt subtrees
            return current_depth + c_factor(node.size)
        if x[node.split_feature] < node.split_value:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)


def c_factor(n: int) -> float:
    """Euler-Mascheroni average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_gamma = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)


class IsolationForest:
    """Ensemble of isolation trees computing normalized anomaly scores."""
    def __init__(self, n_estimators: int = 50, max_samples: int = 128):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
        self.subsample_size = 0

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        self.subsample_size = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(max(self.subsample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=self.subsample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.root = tree.fit(X[idx])
            self.trees.append(tree)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score s(x, n) = 2^(- E[h(x)] / c(n)).
        Scores close to 1 indicate definitive anomalies; scores < 0.5 indicate normal inliers.
        """
        n_samples = X.shape[0]
        paths = np.zeros((n_samples, self.n_estimators))

        for t_idx, tree in enumerate(self.trees):
            for i in range(n_samples):
                paths[i, t_idx] = tree.path_length(X[i], tree.root)

        mean_paths = np.mean(paths, axis=1)
        c_n = c_factor(self.subsample_size)
        scores = 2.0 ** (-mean_paths / c_n)
        return scores


if __name__ == "__main__":
    print("--- J08: Anomaly Detection (Isolation Forest) Sanity Check ---")
    np.random.seed(42)

    # 1. Generate normal clustered inliers
    N_inliers = 200
    inliers = np.random.randn(N_inliers, 2) * 0.8 + np.array([2.0, 2.0])

    # 2. Plant extreme outliers
    outliers = np.array([
        [10.0, 10.0],
        [-8.0, -7.0],
        [12.0, -10.0],
        [-9.0, 11.0]
    ])

    X_train = np.vstack([inliers, outliers])

    forest = IsolationForest(n_estimators=60, max_samples=128)
    forest.fit(X_train)

    inlier_scores = forest.anomaly_score(inliers)
    outlier_scores = forest.anomaly_score(outliers)

    print(f"Mean Inlier Anomaly Score:  {np.mean(inlier_scores):.4f} (Expected < 0.50)")
    print(f"Mean Outlier Anomaly Score: {np.mean(outlier_scores):.4f} (Expected > 0.60)")

    assert np.mean(outlier_scores) > np.mean(inlier_scores) + 0.15, "Outlier anomaly scores must be markedly higher than inliers"
    assert np.all(outlier_scores > 0.55), "All planted outliers must score above anomaly threshold"

    print("J08 Sanity Check: PASSED cleanly.")

"""
B03. k-Nearest Neighbors & Hand-built KD-Tree
From-scratch implementation in pure NumPy.
"""

import numpy as np

class KDNode:
    """A single node in a k-dimensional tree."""
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    """Space-partitioning tree for exact O(log N) nearest neighbor search."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        if len(X) == 0:
            return None
        
        d = X.shape[1]
        axis = depth % d
        
        # Sort along axis and choose median <- eq. 3.2
        sorted_indices = np.argsort(X[:, axis])
        median_idx = len(sorted_indices) // 2
        
        node_idx = sorted_indices[median_idx]
        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx + 1:]
        
        return KDNode(
            point=X[node_idx],
            label=y[node_idx],
            axis=axis,
            left=self._build_tree(X[left_indices], y[left_indices], depth + 1),
            right=self._build_tree(X[right_indices], y[right_indices], depth + 1)
        )

    def search_knn(self, query: np.ndarray, k: int = 3):
        """Finds k nearest neighbors using recursive branch-and-bound."""
        best_neighbors = [] # List of tuples: (-distance, label)

        def _search(node):
            if node is None:
                return
            
            dist = np.linalg.norm(query - node.point)
            
            # Maintain max-heap of size k
            if len(best_neighbors) < k:
                best_neighbors.append((dist, node.label))
                best_neighbors.sort(key=lambda x: x[0])
            elif dist < best_neighbors[-1][0]:
                best_neighbors[-1] = (dist, node.label)
                best_neighbors.sort(key=lambda x: x[0])

            axis = node.axis
            diff = query[axis] - node.point[axis]
            
            first_branch = node.left if diff <= 0 else node.right
            second_branch = node.right if diff <= 0 else node.left
            
            _search(first_branch)
            
            # Prune opposite branch if distance to hyperplane >= k-th distance
            if len(best_neighbors) < k or abs(diff) < best_neighbors[-1][0]:
                _search(second_branch)

        _search(self.root)
        return best_neighbors

class KNNClassifierScratch:
    """k-NN Classifier using our hand-built KDTree."""
    def __init__(self, k: int = 3):
        self.k = k
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.tree = KDTree(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x_q in X:
            neighbors = self.tree.search_knn(x_q, k=self.k)
            labels = [n[1] for n in neighbors]
            # Majority vote
            vals, counts = np.unique(labels, return_counts=True)
            predictions.append(vals[np.argmax(counts)])
        return np.array(predictions)

if __name__ == "__main__":
    print("--- B03: k-NN & KD-Tree Sanity Check ---")
    np.random.seed(42)
    m = 120
    X_train = np.random.uniform(-5, 5, size=(m, 2))
    # Concentric circles dataset
    y_train = (np.sum(X_train**2, axis=1) > 9.0).astype(int)

    knn = KNNClassifierScratch(k=5).fit(X_train, y_train)
    
    # Test point at center (should be class 0)
    test_pt = np.array([[0.0, 0.0], [4.0, 4.0]])
    preds = knn.predict(test_pt)
    print(f"Prediction for (0,0): {preds[0]} (Expected: 0)")
    print(f"Prediction for (4,4): {preds[1]} (Expected: 1)")
    assert preds[0] == 0
    assert preds[1] == 1
    print("B03 Sanity Check: PASSED cleanly.")

"""
M03: Space Partitioning Trees (k-d Tree & Ball Tree)
Pure NumPy implementation of geometric spatial trees for exact nearest neighbor search:
1. KD-Tree with orthogonal axis-aligned splitting and hyperplane pruning
2. Ball Tree with metric hyperspherical bounds and triangle inequality pruning
"""

import numpy as np
from typing import Optional, Tuple, List


class KDNode:
    def __init__(self, point: np.ndarray, point_idx: int, axis: int, left=None, right=None):
        self.point = point
        self.point_idx = point_idx
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """k-d Tree with branch-and-bound nearest neighbor search."""
    def __init__(self, X: np.ndarray):
        self.X = X.astype(float)
        self.dim = X.shape[1]
        self.root = self._build_tree(np.arange(len(X)), depth=0)

    def _build_tree(self, indices: np.ndarray, depth: int) -> Optional[KDNode]:
        if len(indices) == 0:
            return None
        
        axis = depth % self.dim
        # Sort indices along axis
        sorted_indices = indices[np.argsort(self.X[indices, axis])]
        median_idx = len(sorted_indices) // 2
        
        node_idx = sorted_indices[median_idx]
        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx + 1:]

        return KDNode(
            point=self.X[node_idx],
            point_idx=node_idx,
            axis=axis,
            left=self._build_tree(left_indices, depth + 1),
            right=self._build_tree(right_indices, depth + 1)
        )

    def nearest_neighbor(self, query: np.ndarray) -> Tuple[float, int]:
        """Finds the single exact nearest neighbor."""
        best_dist = float("inf")
        best_idx = -1

        def _search(node: Optional[KDNode]):
            nonlocal best_dist, best_idx
            if node is None:
                return

            d = np.linalg.norm(query - node.point)
            if d < best_dist:
                best_dist = d
                best_idx = node.point_idx

            axis = node.axis
            diff = query[axis] - node.point[axis]
            
            near_child = node.left if diff <= 0 else node.right
            far_child = node.right if diff <= 0 else node.left

            # Explore near child first
            _search(near_child)

            # Pruning check: distance to splitting plane must be smaller than best_dist
            if abs(diff) < best_dist:
                _search(far_child)

        _search(self.root)
        return best_dist, best_idx


class BallNode:
    def __init__(self, centroid: np.ndarray, radius: float, indices: np.ndarray, left=None, right=None):
        self.centroid = centroid
        self.radius = radius
        self.indices = indices
        self.left = left
        self.right = right


class BallTree:
    """Ball Tree utilizing reverse triangle inequality for spatial metric pruning."""
    def __init__(self, X: np.ndarray, leaf_size: int = 4):
        self.X = X.astype(float)
        self.dim = X.shape[1]
        self.leaf_size = leaf_size
        self.root = self._build_tree(np.arange(len(X)))

    def _build_tree(self, indices: np.ndarray) -> BallNode:
        centroid = np.mean(self.X[indices], axis=0)
        dists_to_centroid = np.linalg.norm(self.X[indices] - centroid, axis=1)
        radius = float(np.max(dists_to_centroid))

        if len(indices) <= self.leaf_size:
            return BallNode(centroid, radius, indices)

        # Dimension of highest variance
        variances = np.var(self.X[indices], axis=0)
        split_axis = int(np.argmax(variances))
        
        median_val = np.median(self.X[indices, split_axis])
        left_mask = self.X[indices, split_axis] <= median_val
        
        # Guard against zero splits
        if np.all(left_mask) or not np.any(left_mask):
            left_mask = np.arange(len(indices)) < (len(indices) // 2)

        left_node = self._build_tree(indices[left_mask])
        right_node = self._build_tree(indices[~left_mask])

        return BallNode(centroid, radius, indices, left=left_node, right=right_node)

    def nearest_neighbor(self, query: np.ndarray) -> Tuple[float, int]:
        best_dist = float("inf")
        best_idx = -1

        def _search(node: BallNode):
            nonlocal best_dist, best_idx
            # Distance from query to ball center
            d_center = np.linalg.norm(query - node.centroid)

            # Triangle inequality pruning: min possible distance to ball
            if d_center - node.radius >= best_dist:
                return

            if node.left is None and node.right is None:
                # Leaf evaluation
                leaf_dists = np.linalg.norm(self.X[node.indices] - query, axis=1)
                min_i = int(np.argmin(leaf_dists))
                if leaf_dists[min_i] < best_dist:
                    best_dist = leaf_dists[min_i]
                    best_idx = node.indices[min_i]
                return

            # Prioritize child whose center is closer to query
            d_left = np.linalg.norm(query - node.left.centroid)
            d_right = np.linalg.norm(query - node.right.centroid)

            if d_left < d_right:
                _search(node.left)
                _search(node.right)
            else:
                _search(node.right)
                _search(node.left)

        _search(self.root)
        return best_dist, best_idx


if __name__ == "__main__":
    np.random.seed(42)
    n_points = 200
    dim = 8
    X = np.random.randn(n_points, dim)

    kd_tree = KDTree(X)
    ball_tree = BallTree(X, leaf_size=8)

    # Test exactness against brute force on 20 queries
    n_queries = 20
    for _ in range(n_queries):
        q = np.random.randn(dim)
        
        # Ground truth
        exact_dists = np.linalg.norm(X - q, axis=1)
        expected_idx = int(np.argmin(exact_dists))
        expected_dist = exact_dists[expected_idx]

        # KD Tree
        kd_dist, kd_idx = kd_tree.nearest_neighbor(q)
        assert np.isclose(kd_dist, expected_dist), f"KD-tree mismatch: {kd_dist} vs {expected_dist}"
        assert kd_idx == expected_idx, f"KD-tree index mismatch"

        # Ball Tree
        ball_dist, ball_idx = ball_tree.nearest_neighbor(q)
        assert np.isclose(ball_dist, expected_dist), f"Ball-tree mismatch: {ball_dist} vs {expected_dist}"
        assert ball_idx == expected_idx, f"Ball-tree index mismatch"

    print(f"Verified KDTree and BallTree exactness across {n_queries} queries.")
    print("M03 Sanity Check: PASSED cleanly.")

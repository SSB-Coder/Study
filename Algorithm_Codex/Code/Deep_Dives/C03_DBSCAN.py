"""
C03. DBSCAN (Core, Border, Noise Points & Density Expansion)
From-scratch implementation in pure NumPy.
"""

import numpy as np
from collections import deque

class DBSCANScratch:
    """DBSCAN clustering algorithm built from first principles."""
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        self.labels_ = np.full(m, -1, dtype=int) # -1 indicates Noise
        visited = np.zeros(m, dtype=bool)
        core_indices = []

        # Precompute pairwise distance matrix <- eq. 3.1
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        cluster_id = 0

        for i in range(m):
            if visited[i]:
                continue
            visited[i] = True

            # Region query
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # Mark as Noise (could become Border later)
            else:
                # Core point found! Start expanding cluster
                core_indices.append(i)
                self.labels_[i] = cluster_id
                
                # Expansion queue
                queue = deque(neighbors)
                while queue:
                    q = queue.popleft()
                    
                    if not visited[q]:
                        visited[q] = True
                        q_neighbors = np.where(dist_matrix[q] <= self.eps)[0]
                        if len(q_neighbors) >= self.min_samples:
                            core_indices.append(q)
                            queue.extend(q_neighbors)

                    # If q is not assigned to any cluster yet (or was noise), adopt it
                    if self.labels_[q] == -1:
                        self.labels_[q] = cluster_id

                cluster_id += 1

        self.core_sample_indices_ = np.unique(core_indices)
        return self

if __name__ == "__main__":
    print("--- C03: DBSCAN Sanity Check ---")
    np.random.seed(42)
    # Generate concentric rings (classic non-convex problem where k-means fails)
    n_pts = 60
    theta1 = np.linspace(0, 2*np.pi, n_pts)
    inner_ring = np.c_[np.cos(theta1), np.sin(theta1)] * 1.5 + np.random.normal(0, 0.1, (n_pts, 2))
    
    theta2 = np.linspace(0, 2*np.pi, n_pts * 2)
    outer_ring = np.c_[np.cos(theta2), np.sin(theta2)] * 4.0 + np.random.normal(0, 0.1, (n_pts * 2, 2))
    
    # Add a few outliers (noise)
    noise = np.array([[0.0, 0.0], [7.0, 7.0], [-7.0, -7.0]])
    X_db = np.vstack([inner_ring, outer_ring, noise])

    db = DBSCANScratch(eps=0.8, min_samples=4).fit(X_db)
    unique_clusters = set(db.labels_)
    print(f"Discovered Clusters: {unique_clusters} (-1 is noise)")
    # Should discover cluster 0, cluster 1, and noise -1
    assert 0 in unique_clusters and 1 in unique_clusters
    assert -1 in unique_clusters # Outliers should be detected as noise
    print(f"Number of core points: {len(db.core_sample_indices_)}")
    print("C03 Sanity Check: PASSED cleanly.")

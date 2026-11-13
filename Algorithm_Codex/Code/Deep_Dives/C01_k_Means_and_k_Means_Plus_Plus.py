"""
C01. k-Means & k-Means++ (Inertia, Elbow Method, Silhouette Score)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class KMeansScratch:
    """k-Means clustering with k-Means++ initialization and silhouette scoring."""
    def __init__(self, k: int = 3, init: str = "k-means++", max_iter: int = 300, tol: float = 1e-4):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        m, d = X.shape
        if self.init == "random":
            idx = np.random.choice(m, size=self.k, replace=False)
            return X[idx].copy()
        
        # k-means++ seeding <- eq. 3.2
        centroids = [X[np.random.choice(m)]]
        for _ in range(1, self.k):
            # Compute distance to closest center
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            next_c = X[np.random.choice(m, p=probs)]
            centroids.append(next_c)
        return np.array(centroids)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            # Assignment Step: dist matrix (m, k) <- eq. 3.1
            dists = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=-1)
            new_labels = np.argmin(dists, axis=1)

            # Update Step: centroids are cluster means <- eq. 3.1
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.k):
                pts = X[new_labels == k]
                new_centroids[k] = np.mean(pts, axis=0) if len(pts) > 0 else self.centroids[k]

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels = new_labels

            if shift < self.tol:
                break

        # Compute Inertia (WCSS) <- eq. 3.1
        final_dists = np.min(np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=-1), axis=1)
        self.inertia_ = float(np.sum(final_dists))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=-1)
        return np.argmin(dists, axis=1)

    def silhouette_score(self, X: np.ndarray) -> float:
        """Computes mean silhouette coefficient across all points <- eq. 3.3"""
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        scores = np.zeros(m)
        for i in range(m):
            own_cluster = self.labels[i]
            own_pts = X[self.labels == own_cluster]
            
            if len(own_pts) <= 1:
                scores[i] = 0.0
                continue
            
            # a(i): intra-cluster mean distance
            a_i = np.mean(np.linalg.norm(own_pts - X[i], axis=1)) * (len(own_pts) / (len(own_pts) - 1))
            
            # b(i): nearest other cluster mean distance
            b_i = np.inf
            for other_c in range(self.k):
                if other_c == own_cluster:
                    continue
                other_pts = X[self.labels == other_c]
                if len(other_pts) > 0:
                    dist_to_other = np.mean(np.linalg.norm(other_pts - X[i], axis=1))
                    b_i = min(b_i, dist_to_other)
                    
            scores[i] = (b_i - a_i) / max(a_i, b_i)
        return float(np.mean(scores))

if __name__ == "__main__":
    print("--- C01: k-Means & k-Means++ Sanity Check ---")
    np.random.seed(42)
    # Generate 3 well-separated 2D blobs
    X_blobs = np.vstack([
        np.random.randn(40, 2) + np.array([-4, -4]),
        np.random.randn(40, 2) + np.array([4, 4]),
        np.random.randn(40, 2) + np.array([0, 4])
    ])
    
    km = KMeansScratch(k=3, init="k-means++").fit(X_blobs)
    sil = km.silhouette_score(X_blobs)
    print(f"k-Means Inertia (WCSS): {km.inertia_:.2f}")
    print(f"Silhouette Score: {sil:.4f} (Expected > 0.50 for well-separated clusters)")
    assert sil > 0.50
    # Inertia must be positive
    assert km.inertia_ > 0.0
    print("C01 Sanity Check: PASSED cleanly.")

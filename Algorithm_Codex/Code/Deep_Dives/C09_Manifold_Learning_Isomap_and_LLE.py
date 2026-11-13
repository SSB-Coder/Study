"""
C09. Manifold Learning (Isomap & Locally Linear Embedding - LLE)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class IsomapScratch:
    """Isometric Feature Mapping via graph geodesics and Classical MDS."""
    def __init__(self, n_components: int = 2, n_neighbors: int = 5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)

        # 1. Pairwise Euclidean distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        # 2. Build k-NN adjacency graph
        adj_matrix = np.full((m, m), np.inf)
        for i in range(m):
            knn_idx = np.argsort(dists[i])[1:self.n_neighbors + 1]
            adj_matrix[i, knn_idx] = dists[i, knn_idx]
            adj_matrix[knn_idx, i] = dists[i, knn_idx]
        np.fill_diagonal(adj_matrix, 0.0)

        # 3. Floyd-Warshall all-pairs shortest paths
        D_geo = adj_matrix.copy()
        for k in range(m):
            D_geo = np.minimum(D_geo, D_geo[:, k:k+1] + D_geo[k:k+1, :])

        # Handle any disconnected components by replacing inf with max path
        max_val = np.max(D_geo[np.isfinite(D_geo)])
        D_geo[np.isinf(D_geo)] = max_val * 2.0

        # 4. Classical MDS on geodesic distance matrix <- eq. 3.1
        H = np.eye(m) - (1.0 / m) * np.ones((m, m)) # Double centering matrix
        B = -0.5 * (H @ (D_geo ** 2) @ H)

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top positive eigenvalues
        top_evals = np.maximum(0.0, eigenvalues[:self.n_components])
        self.embedding_ = eigenvectors[:, :self.n_components] * np.sqrt(top_evals)
        return self.embedding_

class LLEScratch:
    """Locally Linear Embedding from first principles."""
    def __init__(self, n_components: int = 2, n_neighbors: int = 6, reg: float = 1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m, d = X.shape
        
        # 1. Find k-nearest neighbors
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        W = np.zeros((m, m))
        for i in range(m):
            knn_idx = np.argsort(dists[i])[1:self.n_neighbors + 1]
            z = X[knn_idx] - X[i] # Neighborhood diffs
            C = z @ z.T # Local Gram matrix <- eq. 3.2
            C += self.reg * np.trace(C) * np.eye(self.n_neighbors) # Regularize
            
            # Solve C * w = 1
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w = w / np.sum(w) # Sum to 1 constraint
            W[i, knn_idx] = w

        # 2. Compute embedding coordinates from M = (I - W)^T (I - W) <- eq. 3.2
        M = (np.eye(m) - W).T @ (np.eye(m) - W)
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort ascending (smallest non-zero eigenvalues)
        idx = np.argsort(eigenvalues)
        # Skip the 0-th eigenvector (trivial constant vector)
        self.embedding_ = eigenvectors[:, idx[1:self.n_components + 1]]
        return self.embedding_

if __name__ == "__main__":
    print("--- C09: Manifold Learning (Isomap & LLE) Sanity Check ---")
    np.random.seed(42)
    # Generate an unrolled S-curve manifold
    n = 60
    t = np.linspace(0, 3*np.pi, n)
    x_s = np.sin(t)
    y_s = np.sign(t - 1.5*np.pi) * (np.cos(t) - 1.0)
    z_s = np.random.uniform(-1, 1, n)
    X_manifold = np.c_[x_s, y_s, z_s]

    isomap = IsomapScratch(n_components=2, n_neighbors=6).fit_transform(X_manifold)
    lle = LLEScratch(n_components=2, n_neighbors=6).fit_transform(X_manifold)

    print(f"Isomap Output Shape: {isomap.shape}")
    print(f"LLE Output Shape:    {lle.shape}")
    assert isomap.shape == (n, 2)
    assert lle.shape == (n, 2)
    assert np.all(np.isfinite(isomap))
    assert np.all(np.isfinite(lle))
    print("C09 Sanity Check: PASSED cleanly.")

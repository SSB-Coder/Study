"""
C10. UMAP (Fuzzy Simplicial Sets & Cross-Entropy Layout Optimization)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class UMAPScratch:
    """Simplified UMAP implementation built from mathematical first principles."""
    def __init__(self, n_components: int = 2, n_neighbors: int = 8, 
                 min_dist: float = 0.1, lr: float = 1.0, n_epochs: int = 200, seed: int = 42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.embedding_ = None

    def _fuzzy_simplicial_set(self, X: np.ndarray):
        m = len(X)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        # rho_i: distance to nearest neighbor <- eq. 3.1
        rho = np.zeros(m)
        for i in range(m):
            nonzero_dists = dists[i, dists[i] > 0]
            rho[i] = np.min(nonzero_dists) if len(nonzero_dists) > 0 else 0.0

        # Binary search for sigma_i matching target sum log2(k)
        target = np.log2(self.n_neighbors)
        P = np.zeros((m, m))

        for i in range(m):
            knn_idx = np.argsort(dists[i])[1:self.n_neighbors + 1]
            d_knn = dists[i, knn_idx]
            
            sigma = 1.0
            for _ in range(30):
                val = np.sum(np.exp(-np.maximum(0.0, d_knn - rho[i]) / sigma))
                diff_val = val - target
                if abs(diff_val) < 1e-4:
                    break
                sigma *= 1.1 if diff_val < 0 else 0.9

            # Compute directed probabilities
            p_directed = np.exp(-np.maximum(0.0, d_knn - rho[i]) / sigma)
            P[i, knn_idx] = p_directed

        # Symmetrize via algebraic t-conorm: P + P^T - P * P^T <- eq. 3.1
        P_sym = P + P.T - (P * P.T)
        return np.maximum(P_sym, 1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        rng = np.random.default_rng(self.seed)

        # 1. High-dimensional fuzzy simplicial set
        P = self._fuzzy_simplicial_set(X)

        # 2. Random initialization
        Y = rng.normal(0.0, 1.0, size=(m, self.n_components))

        # 3. Optimize fuzzy cross-entropy layout via gradient descent <- eq. 3.2
        a, b = 1.0, 1.0
        for epoch in range(self.n_epochs):
            alpha = self.lr * (1.0 - epoch / self.n_epochs)
            
            y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
            y_dists_sq = np.sum(y_diff ** 2, axis=-1)
            
            # Low dim affinity: q = 1 / (1 + a * dist^(2b))
            Q = 1.0 / (1.0 + a * (y_dists_sq ** b))
            np.fill_diagonal(Q, 0.0)

            # Attractive gradient: P * (1 / (1 + dist^2))
            grad_attr = (P * Q)[:, :, np.newaxis] * y_diff
            
            # Repulsive gradient: (1 - P) * Q * (1 - Q)
            grad_rep = ((1.0 - P) * (Q ** 2) / np.maximum(1e-4, 1.0 - Q))[:, :, np.newaxis] * y_diff

            # Total gradient
            dY = 2.0 * np.sum(grad_attr - grad_rep, axis=1)
            Y -= alpha * np.clip(dY, -4.0, 4.0)

        self.embedding_ = Y
        return self.embedding_

if __name__ == "__main__":
    print("--- C10: UMAP Sanity Check ---")
    np.random.seed(42)
    # 3 distinct clusters in 5D space
    c1 = np.random.randn(20, 5) + np.array([4, 4, 0, 0, 0])
    c2 = np.random.randn(20, 5) + np.array([-4, -4, 0, 0, 0])
    c3 = np.random.randn(20, 5) + np.array([0, 4, 4, 0, 0])
    X_umap = np.vstack([c1, c2, c3])

    umap = UMAPScratch(n_components=2, n_neighbors=6, n_epochs=100)
    emb = umap.fit_transform(X_umap)
    print(f"UMAP Output Shape: {emb.shape}")
    assert emb.shape == (60, 2)
    assert np.all(np.isfinite(emb))
    print("C10 Sanity Check: PASSED cleanly.")

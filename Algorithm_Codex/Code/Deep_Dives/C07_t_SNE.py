"""
C07. t-SNE (Pairwise Affinities, Student-t Kernels & KL Divergence Optimization)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class TSNEScratch:
    """t-Distributed Stochastic Neighbor Embedding built from scratch."""
    def __init__(self, n_components: int = 2, perplexity: float = 10.0, 
                 lr: float = 100.0, n_iters: int = 300, seed: int = 42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iters = n_iters
        self.seed = seed
        self.embedding_ = None

    def _h_beta(self, dist_row: np.ndarray, beta: float):
        """Computes entropy and affinities for a single row given precision beta <- eq. 3.1"""
        aff = np.exp(-dist_row * beta)
        sum_aff = np.sum(aff) + 1e-15
        h = np.log(sum_aff) + beta * np.sum(dist_row * aff) / sum_aff
        return h, aff / sum_aff

    def _binary_search_sigma(self, dist_matrix: np.ndarray):
        m = len(dist_matrix)
        P = np.zeros((m, m))
        target_h = np.log(self.perplexity)

        for i in range(m):
            dist_i = dist_matrix[i, np.arange(m) != i]
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0

            for _ in range(50):
                h, p_row = self._h_beta(dist_i, beta)
                h_diff = h - target_h
                if abs(h_diff) < 1e-5:
                    break
                if h_diff > 0:
                    beta_min = beta
                    beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0

            P[i, np.arange(m) != i] = p_row

        # Symmetrize P <- eq. 3.1
        P_sym = (P + P.T) / (2.0 * m)
        return np.maximum(P_sym, 1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        rng = np.random.default_rng(self.seed)

        # 1. High-dimensional pairwise squared distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # 2. Find high-dim joint probabilities P
        P = self._binary_search_sigma(dist_sq)

        # 3. Initialize low-dim coordinates Y randomly
        Y = rng.normal(0.0, 1e-4, size=(m, self.n_components))
        Y_prev = Y.copy()

        # 4. Optimization Loop
        for step in range(self.n_iters):
            # Compute pairwise low-dim distances and Student-t affinities <- eq. 3.2
            y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
            y_dist_sq = np.sum(y_diff ** 2, axis=-1)
            
            # Cauchy kernel: (1 + ||y_i - y_j||^2)^(-1)
            num = 1.0 / (1.0 + y_dist_sq)
            np.fill_diagonal(num, 0.0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Evaluate exact gradient dL/dy_i <- eq. 3.3
            PQ_diff = (P - Q) * num # Shape (m, m)
            dY = np.zeros_like(Y)
            for i in range(m):
                dY[i] = 4.0 * np.sum((Y[i] - Y) * PQ_diff[i, :, np.newaxis], axis=0)

            # Update with momentum
            momentum = 0.5 if step < 50 else 0.8
            Y_new = Y - self.lr * dY + momentum * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new

            # Re-center Y
            Y -= np.mean(Y, axis=0)

        self.embedding_ = Y
        return self.embedding_

if __name__ == "__main__":
    print("--- C07: t-SNE Sanity Check ---")
    np.random.seed(42)
    # 2 distinct clusters in 4-dimensional space
    c1 = np.random.randn(25, 4) + np.array([5, 5, 0, 0])
    c2 = np.random.randn(25, 4) + np.array([-5, -5, 0, 0])
    X_tsne = np.vstack([c1, c2])

    tsne = TSNEScratch(n_components=2, perplexity=8.0, n_iters=150, lr=50.0)
    embedding = tsne.fit_transform(X_tsne)
    
    # In 2D embedding space, distance between cluster centers should be significantly larger than intra-cluster spread
    center1 = np.mean(embedding[:25], axis=0)
    center2 = np.mean(embedding[25:], axis=0)
    inter_cluster_dist = np.linalg.norm(center1 - center2)
    intra_spread = np.mean(np.linalg.norm(embedding[:25] - center1, axis=1))
    
    print(f"Low-dim Inter-cluster Distance: {inter_cluster_dist:.2f}")
    print(f"Low-dim Intra-cluster Spread:   {intra_spread:.2f}")
    assert inter_cluster_dist > intra_spread * 2.0
    print("C07 Sanity Check: PASSED cleanly.")

"""
Y03: Gromov-Wasserstein & Sliced Optimal Transport
Pure NumPy implementation of:
1. Sliced Wasserstein Distance (SWD) via 1D Radon projections and analytical sorting.
2. Entropic Gromov-Wasserstein (GW) discrepancy for cross-domain metric alignment
   using projected gradient descent and inner Sinkhorn-Knopp iterations.
"""

import numpy as np

class SlicedWasserstein:
    """
    Computes Sliced Wasserstein Distance between two point clouds in R^d.
    Projects distributions onto K random 1D directions where OT is exact via sorting.
    """
    @staticmethod
    def distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 50, seed: int = 42) -> float:
        """
        X: (N, d), Y: (M, d) point matrices.
        Returns average 1D Wasserstein-1 distance across random unit projections.
        """
        rng = np.random.default_rng(seed)
        d = X.shape[1]
        
        # Sample K random unit vectors uniformly on S^{d-1}
        projections = rng.normal(size=(n_projections, d))
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        
        # Project data: (N, K) and (M, K)
        proj_X = X @ projections.T
        proj_Y = Y @ projections.T
        
        # Sort along 1D slices
        sorted_X = np.sort(proj_X, axis=0)
        sorted_Y = np.sort(proj_Y, axis=0)
        
        # If N != M, interpolate quantiles
        if len(X) != len(Y):
            quantiles = np.linspace(0, 1, max(len(X), len(Y)))
            q_X = np.quantile(sorted_X, quantiles, axis=0)
            q_Y = np.quantile(sorted_Y, quantiles, axis=0)
            w1_slices = np.mean(np.abs(q_X - q_Y), axis=0)
        else:
            w1_slices = np.mean(np.abs(sorted_X - sorted_Y), axis=0)
            
        return float(np.mean(w1_slices))

class GromovWasserstein:
    """
    Computes Entropic Gromov-Wasserstein alignment between two metric spaces
    (X, D_X) and (Y, D_Y) across incomparable coordinate representations.
    """
    @staticmethod
    def sinkhorn(cost_matrix: np.ndarray, p: np.ndarray, q: np.ndarray, eps: float = 0.05, max_iter: int = 100) -> np.ndarray:
        """Standard Sinkhorn-Knopp matrix scaling with cost-shift stabilization."""
        cost_shifted = cost_matrix - np.min(cost_matrix)
        K = np.exp(-cost_shifted / eps)
        u = np.ones_like(p) / len(p)
        for _ in range(max_iter):
            v = q / (K.T @ u + 1e-15)
            u = p / (K @ v + 1e-15)
        return np.diag(u) @ K @ np.diag(v)

    @staticmethod
    def align(D_X: np.ndarray, D_Y: np.ndarray, eps: float = 0.01, max_iter: int = 50) -> tuple[np.ndarray, float]:
        """
        Entropic Gromov-Wasserstein via projected gradient descent / entropic proximal point.
        Minimizes E(T) = sum_{i,j,k,l} |D_X[i,k] - D_Y[j,l]|^2 T[i,j] T[k,l].
        Returns optimal coupling matrix T and quadratic distortion cost.
        """
        N = len(D_X)
        M = len(D_Y)
        p = np.ones(N) / N
        q = np.ones(M) / M
        
        # Initialize coupling as uniform product: T = p (x) q
        T = np.outer(p, q)
        
        for _ in range(max_iter):
            # Cost matrix C(T) = (D_X^2 p) 1^T + 1 (D_Y^2 q)^T - 2 D_X T D_Y^T
            cost = (D_X**2 @ p)[:, None] + (D_Y**2 @ q)[None, :] - 2.0 * (D_X @ T @ D_Y.T)
            # Inner Sinkhorn step
            T = GromovWasserstein.sinkhorn(cost, p, q, eps=eps, max_iter=100)
            
        # Compute exact quadratic distortion E(T)
        diff = D_X[:, None, :, None] - D_Y[None, :, None, :]
        distortion = float(np.sum((diff**2) * T[:, :, None, None] * T[None, None, :, :]))
        return T, distortion

if __name__ == "__main__":
    print("--- Y03: Gromov-Wasserstein & Sliced Optimal Transport Sanity Check ---")
    
    # 1. Sliced Wasserstein Distance Test
    rng = np.random.default_rng(42)
    pts1 = rng.normal(loc=0.0, scale=1.0, size=(100, 3))
    # Identical distribution -> SWD ~ 0
    swd_self = SlicedWasserstein.distance(pts1, pts1)
    assert np.isclose(swd_self, 0.0), "Self Sliced Wasserstein distance must be 0"
    
    # Translated distribution by shift = (2, 0, 0)
    pts2 = pts1 + np.array([2.0, 0.0, 0.0])
    swd_shifted = SlicedWasserstein.distance(pts1, pts2, n_projections=100)
    print(f"Sliced Wasserstein Distance (Shift=2.0 along X): {swd_shifted:.4f}")
    assert 0.5 < swd_shifted < 2.0, "SWD must capture spatial distribution shift"
    
    # 2. Gromov-Wasserstein Cross-Domain Graph Alignment
    # Graph X: 3 nodes in 2D forming an asymmetric triangle (sides 3, 4, 5) to ensure unique isomorphism
    pts_X = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
    D_X = np.linalg.norm(pts_X[:, None, :] - pts_X[None, :, :], axis=-1)
    
    # Graph Y: Isometric copy embedded in 5D space with permuted node order: [2, 0, 1]
    perm = [2, 0, 1]
    # Rotate and embed into 5D via orthogonal 2x5 projection
    H = rng.normal(size=(5, 5))
    Q, _ = np.linalg.qr(H)
    R5 = Q[:2]  # 2x5 projection
    pts_Y_orig = pts_X @ R5  # (3, 5)
    pts_Y = pts_Y_orig[perm]  # permuted
    D_Y = np.linalg.norm(pts_Y[:, None, :] - pts_Y[None, :, :], axis=-1)
    
    # Expected node mapping: for each node i in X, which node j in Y does it correspond to?
    expected_mapping = [int(np.where(np.array(perm) == i)[0][0]) for i in range(len(pts_X))]
    
    # Run Gromov-Wasserstein
    coupling, gw_loss = GromovWasserstein.align(D_X, D_Y, eps=0.01, max_iter=50)
    print(f"Gromov-Wasserstein Discrepancy (Isometric 2D vs 5D): {gw_loss:.6e}")
    assert gw_loss < 1e-4, "Isometric structures across different dimensions must have near-zero GW distortion"
    
    # Check recovered permutation
    inferred_mapping = np.argmax(coupling, axis=1).tolist()
    print(f"Ground Truth Mapping: {expected_mapping}")
    print(f"GW Inferred Mapping : {inferred_mapping}")
    assert inferred_mapping == expected_mapping, "Gromov-Wasserstein must recover exact node correspondence"
    
    print("Y03 Sanity Check: PASSED cleanly.")

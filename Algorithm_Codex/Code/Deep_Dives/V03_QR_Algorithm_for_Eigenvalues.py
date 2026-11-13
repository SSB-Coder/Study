"""
V03. The QR Algorithm for Eigenvalue Extraction
From-scratch implementation in pure NumPy.
"""

import numpy as np

class QREigenvalues:
    """
    Computes all eigenvalues (and eigenvectors) of a real symmetric or general matrix
    using the unshifted or Rayleigh-quotient shifted QR iteration.
    """
    
    @staticmethod
    def _householder_qr(A: np.ndarray):
        """Internal Householder QR routine."""
        m, n = A.shape
        R = A.copy()
        Q = np.eye(m)
        for k in range(min(m - 1, n)):
            x = R[k:m, k]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-15:
                continue
            alpha = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x
            v = x.copy()
            v[0] -= alpha
            v /= np.linalg.norm(v)
            v_col = v[:, np.newaxis]
            R[k:m, k:] -= 2.0 * v_col @ (v_col.T @ R[k:m, k:])
            Q_k = np.eye(m)
            Q_k[k:m, k:m] -= 2.0 * np.outer(v, v)
            Q = Q @ Q_k
        return Q, R

    @classmethod
    def compute_eigenvalues_symmetric(cls, A: np.ndarray, max_iter: int = 300, tol: int = 1e-10):
        """
        Iterative QR algorithm for real symmetric matrices.
        Returns sorted eigenvalues and eigenvector matrix V.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if not np.allclose(A, A.T, atol=1e-7):
            # Enforce symmetry
            A = 0.5 * (A + A.T)
            
        T = A.copy()
        V = np.eye(n, dtype=float)
        
        for it in range(max_iter):
            # Off-diagonal norm check <- eq. 3.1
            off_diag = np.sqrt(np.sum(T**2) - np.sum(np.diag(T)**2))
            if off_diag < tol:
                break
                
            # Wilkinson shift on 2x2 trailing submatrix for cubic convergence <- eq. 3.2
            d = (T[-2, -2] - T[-1, -1]) / 2.0
            sign_d = np.sign(d) if d != 0 else 1.0
            shift = T[-1, -1] - sign_d * (T[-1, -2]**2) / (np.abs(d) + np.sqrt(d**2 + T[-1, -2]**2 + 1e-15))
            
            # QR step with shift
            Q, R = cls._householder_qr(T - shift * np.eye(n))
            T = R @ Q + shift * np.eye(n)
            V = V @ Q
            
        evals = np.diag(T)
        sort_idx = np.argsort(evals)[::-1]
        return evals[sort_idx], V[:, sort_idx]

if __name__ == "__main__":
    print("--- V03: QR Algorithm for Eigenvalues Sanity Check ---")
    np.random.seed(42)
    # Generate random symmetric matrix
    M = np.random.randn(4, 4)
    A = M + M.T
    
    evals, evecs = QREigenvalues.compute_eigenvalues_symmetric(A)
    print(f"Computed Eigenvalues: {evals}")
    
    # Verify A @ v = lambda * v for each pair
    for i in range(len(evals)):
        lam = evals[i]
        v = evecs[:, i]
        res = np.linalg.norm(A @ v - lam * v)
        assert res < 1e-7, f"Eigenvector {i} residual too high: {res}"
    
    # Section 3.4 Verification Ground-Truth Check
    # ground truth for verification only — not used by the algorithm above
    true_evals, _ = np.linalg.eigh(A)
    true_evals_sorted = np.sort(true_evals)[::-1]
    np.testing.assert_allclose(evals, true_evals_sorted, atol=1e-8)
    print("V03 Sanity Check: PASSED cleanly.")

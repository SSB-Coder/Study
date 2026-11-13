"""
V01. QR Decomposition (Modified Gram-Schmidt & Householder Reflections)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class QRDecomposition:
    """
    Computes the QR decomposition of matrix A in R^{m x n} (m >= n)
    such that A = Q @ R, where Q in R^{m x m} (or R^{m x n}) has orthonormal
    columns (Q.T @ Q = I) and R in R^{n x n} is upper triangular.
    """
    
    @staticmethod
    def modified_gram_schmidt(A: np.ndarray):
        """
        Modified Gram-Schmidt (MGS) orthogonalization.
        Numerically stable compared to Classical Gram-Schmidt.
        Returns Q (m x n), R (n x n).
        """
        A = np.array(A, dtype=float)
        m, n = A.shape
        Q = np.zeros((m, n), dtype=float)
        R = np.zeros((n, n), dtype=float)
        V = A.copy()
        
        for i in range(n):
            # Length of vector determines R[i, i] <- eq. 3.1
            R[i, i] = np.linalg.norm(V[:, i])
            if R[i, i] > 1e-15:
                Q[:, i] = V[:, i] / R[i, i]
            else:
                Q[:, i] = 0.0
            
            # Project out Q[:, i] from subsequent columns <- eq. 3.2
            for j in range(i + 1, n):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] = V[:, j] - R[i, j] * Q[:, i]
                
        return Q, R

    @staticmethod
    def householder(A: np.ndarray):
        """
        Householder reflection QR decomposition.
        Produces an orthogonal Q (m x m) and upper-triangular R (m x n).
        """
        A = np.array(A, dtype=float)
        m, n = A.shape
        R = A.copy()
        Q = np.eye(m, dtype=float)
        
        for k in range(min(m - 1, n)):
            # Extract vector to reflect
            x = R[k:m, k]
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-15:
                continue
            
            # Householder vector v <- eq. 3.3
            alpha = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x
            v = x.copy()
            v[0] -= alpha
            v = v / np.linalg.norm(v)
            
            # Apply reflection H_k = I - 2 * v * v.T to R[k:m, k:n]
            # R[k:m, k:n] = R[k:m, k:n] - 2 * v * (v.T @ R[k:m, k:n])
            v_col = v[:, np.newaxis]
            R[k:m, k:] -= 2.0 * v_col @ (v_col.T @ R[k:m, k:])
            
            # Accumulate into Q: Q = Q @ H_k
            Q_k = np.eye(m)
            Q_k[k:m, k:m] -= 2.0 * np.outer(v, v)
            Q = Q @ Q_k
            
        return Q, R

if __name__ == "__main__":
    print("--- V01: QR Decomposition Sanity Check ---")
    np.random.seed(42)
    A = np.random.randn(6, 4)
    
    # 1. Test Modified Gram-Schmidt
    Q_mgs, R_mgs = QRDecomposition.modified_gram_schmidt(A)
    recon_err_mgs = np.linalg.norm(A - Q_mgs @ R_mgs)
    ortho_err_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(4))
    print(f"MGS Reconstruction Error: {recon_err_mgs:.2e}, Orthogonality Error: {ortho_err_mgs:.2e}")
    assert recon_err_mgs < 1e-12, "MGS reconstruction failed"
    assert ortho_err_mgs < 1e-12, "MGS orthogonality failed"
    
    # 2. Test Householder
    Q_hh, R_hh = QRDecomposition.householder(A)
    recon_err_hh = np.linalg.norm(A - Q_hh @ R_hh)
    ortho_err_hh = np.linalg.norm(Q_hh.T @ Q_hh - np.eye(6))
    print(f"Householder Reconstruction Error: {recon_err_hh:.2e}, Orthogonality Error: {ortho_err_hh:.2e}")
    assert recon_err_hh < 1e-12, "Householder reconstruction failed"
    assert ortho_err_hh < 1e-12, "Householder orthogonality failed"
    
    # Section 3.4 Verification Ground-Truth Check
    # ground truth for verification only — not used by the algorithm above
    Q_np, R_np = np.linalg.qr(A)
    # Signs of diagonal elements of R can differ, but |R| matches
    np.testing.assert_allclose(np.abs(np.diag(R_mgs)), np.abs(np.diag(R_np)), atol=1e-10)
    print("V01 Sanity Check: PASSED cleanly.")

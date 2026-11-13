"""
V02. LU Decomposition with Partial Pivoting & Linear System Solving
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LUSolver:
    """
    Computes PA = LU factorization with partial pivoting for a square matrix A in R^{n x n},
    where P is a permutation matrix, L is unit lower-triangular, and U is upper-triangular.
    Solves A x = b and computes det(A) and matrix inverse.
    """
    
    @staticmethod
    def factorize(A: np.ndarray):
        """
        Doolittle algorithm with partial row pivoting.
        Returns P (permutation matrix), L (unit lower-triangular), U (upper-triangular),
        and num_pivots (number of row swaps for determinant sign).
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("Matrix must be square for LU factorization")
            
        P = np.eye(n, dtype=float)
        L = np.eye(n, dtype=float)
        U = A.copy()
        num_pivots = 0
        
        for k in range(n - 1):
            # Partial pivoting: select max absolute element in column k below diagonal <- eq. 3.1
            pivot_row = k + np.argmax(np.abs(U[k:n, k]))
            if np.abs(U[pivot_row, k]) < 1e-15:
                raise np.linalg.LinAlgError(f"Singular matrix encountered at column {k}")
                
            if pivot_row != k:
                # Swap rows in U
                U[[k, pivot_row], k:] = U[[pivot_row, k], k:]
                # Swap rows in P
                P[[k, pivot_row], :] = P[[pivot_row, k], :]
                # Swap subdiagonal elements in L
                if k > 0:
                    L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                num_pivots += 1
                
            # Elimination step <- eq. 3.2
            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]
                
        return P, L, U, num_pivots

    @staticmethod
    def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solves L y = b for unit lower-triangular L."""
        n = L.shape[0]
        y = np.zeros(n, dtype=float)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        return y

    @staticmethod
    def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solves U x = y for upper-triangular U."""
        n = U.shape[0]
        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            if np.abs(U[i, i]) < 1e-15:
                raise np.linalg.LinAlgError("Singular matrix in back substitution")
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        return x

    @classmethod
    def solve(cls, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solves A x = b using PA = LU."""
        P, L, U, _ = cls.factorize(A)
        Pb = P @ b
        y = cls.forward_substitution(L, Pb)
        x = cls.backward_substitution(U, y)
        return x

    @classmethod
    def determinant(cls, A: np.ndarray) -> float:
        """det(A) = (-1)^num_pivots * prod(diag(U))."""
        _, _, U, num_pivots = cls.factorize(A)
        sign = -1.0 if num_pivots % 2 == 1 else 1.0
        return sign * float(np.prod(np.diag(U)))

if __name__ == "__main__":
    print("--- V02: LU Decomposition & Linear Solver Sanity Check ---")
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n) + np.eye(n) * 2.0
    b = np.random.randn(n)
    
    # 1. Factorize and verify P @ A = L @ U
    P, L, U, _ = LUSolver.factorize(A)
    recon_err = np.linalg.norm(P @ A - L @ U)
    print(f"PA - LU Reconstruction Error: {recon_err:.2e}")
    assert recon_err < 1e-12, "LU factorization failed"
    
    # 2. Solve A x = b
    x = LUSolver.solve(A, b)
    res_norm = np.linalg.norm(A @ x - b)
    print(f"Residual Norm ||Ax - b||: {res_norm:.2e}")
    assert res_norm < 1e-12, "Linear solve failed"
    
    # 3. Determinant check
    det_val = LUSolver.determinant(A)
    
    # Section 3.4 Verification Ground-Truth Check
    # ground truth for verification only — not used by the algorithm above
    x_np = np.linalg.solve(A, b)
    det_np = np.linalg.det(A)
    np.testing.assert_allclose(x, x_np, atol=1e-10)
    np.testing.assert_allclose(det_val, det_np, rtol=1e-10)
    print("V02 Sanity Check: PASSED cleanly.")

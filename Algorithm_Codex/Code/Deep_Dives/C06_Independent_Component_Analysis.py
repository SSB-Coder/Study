"""
C06. Independent Component Analysis (FastICA via Negentropy)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class FastICAScratch:
    """FastICA algorithm for blind source separation using tanh contrast function."""
    def __init__(self, n_components: int = 2, max_iter: int = 200, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None # Unmixing matrix
        self.K = None # Whitening matrix
        self.mean_ = None

    def _whiten(self, X: np.ndarray):
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_
        m = len(X_c)
        
        # SVD-based whitening: X_c = U S V^T -> X_white = sqrt(m) * U <- eq. 3.1
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.K = (Vt[:self.n_components].T / (S[:self.n_components] / np.sqrt(m))).T
        X_white = X_c @ self.K.T
        return X_white

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        X_white = self._whiten(X)
        p = self.n_components

        # Initialize random orthonormal unmixing matrix
        W = np.random.randn(p, p)
        # Symmetric decorrelation W = (W @ W^T)^(-1/2) @ W <- eq. 3.3
        s, u = np.linalg.eigh(W @ W.T)
        W = (u @ np.diag(1.0 / np.sqrt(s)) @ u.T) @ W

        for iteration in range(self.max_iter):
            # Projections u = X_white @ W.T
            U_proj = X_white @ W.T
            
            # Contrast function g(u) = tanh(u), g'(u) = 1 - tanh(u)^2 <- eq. 3.2
            g_u = np.tanh(U_proj)
            g_prime_u = 1.0 - g_u ** 2
            
            # Fixed point update: W+ = (1/m) * g(u)^T @ X_white - diag(mean(g'(u))) @ W <- eq. 3.3
            W_new = (g_u.T @ X_white) / m - np.diag(np.mean(g_prime_u, axis=0)) @ W
            
            # Symmetric decorrelation
            s, u = np.linalg.eigh(W_new @ W_new.T)
            W_new = (u @ np.diag(1.0 / np.sqrt(s)) @ u.T) @ W_new

            # Check convergence: max absolute dot product of old and new rows
            lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
            W = W_new
            if lim < self.tol:
                break

        self.W = W
        # Recovered sources: S = X_white @ W.T
        return X_white @ self.W.T

if __name__ == "__main__":
    print("--- C06: FastICA Sanity Check ---")
    np.random.seed(42)
    # Synthesize two distinct non-Gaussian signals: a sine wave and a square-like wave
    time = np.linspace(0, 8, 500)
    s1 = np.sin(2 * time) # Signal 1: harmonic wave
    s2 = np.sign(np.sin(3 * time)) # Signal 2: square wave
    S_true = np.c_[s1, s2]

    # Mix them linearly with mixing matrix A
    A_true = np.array([[1.0, 0.5], 
                       [0.5, 2.0]])
    X_mixed = S_true @ A_true.T

    # Separate with FastICA
    ica = FastICAScratch(n_components=2).fit_transform(X_mixed)
    
    # Check correlation between recovered signals and true signals
    corrs = np.abs(np.corrcoef(S_true.T, ica.T))[:2, 2:]
    print("Correlation matrix between True Sources and FastICA Recovered Sources:")
    print(np.round(corrs, 4))
    # Each true signal should correlate strongly (>0.90) with one of the recovered signals
    max_corr1 = np.max(corrs[0, :])
    max_corr2 = np.max(corrs[1, :])
    print(f"Max Source 1 correlation: {max_corr1:.4f}")
    print(f"Max Source 2 correlation: {max_corr2:.4f}")
    assert max_corr1 > 0.90
    assert max_corr2 > 0.90
    print("C06 Sanity Check: PASSED cleanly.")

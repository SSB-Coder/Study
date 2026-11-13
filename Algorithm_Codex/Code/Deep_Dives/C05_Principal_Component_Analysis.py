"""
C05. Principal Component Analysis (PCA via Eigendecomposition and SVD)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PCAScratch:
    """Principal Component Analysis implementing both Covariance Eigendecomposition and SVD."""
    def __init__(self, n_components: int = 2, method: str = "svd"):
        self.n_components = n_components
        self.method = method # 'svd' or 'covariance'
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        m, d = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.method == "covariance":
            # Method A: Eigendecomposition of sample covariance matrix <- eq. 3.1
            cov = (X_centered.T @ X_centered) / m
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort descending
            sort_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

        elif self.method == "svd":
            # Method B: Singular Value Decomposition X = U S V^T <- eq. 3.2
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Eigenvalues relate to singular values via S^2 / m
            eigenvalues = (S ** 2) / m
            
            self.components_ = Vt[:self.n_components, :]
            self.explained_variance_ = eigenvalues[:self.n_components]
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components_ + self.mean_

if __name__ == "__main__":
    print("--- C05: PCA Sanity Check (Covariance vs SVD Agreement) ---")
    np.random.seed(42)
    m, d = 100, 5
    # Generate data with strong correlation in first 2 dimensions
    X_synth = np.random.randn(m, d)
    X_synth[:, 1] = X_synth[:, 0] * 2.5 + np.random.normal(0, 0.2, m)
    X_synth[:, 2] = X_synth[:, 0] * -1.8 + np.random.normal(0, 0.2, m)

    pca_cov = PCAScratch(n_components=2, method="covariance").fit(X_synth)
    pca_svd = PCAScratch(n_components=2, method="svd").fit(X_synth)

    print(f"Covariance Explained Variance: {pca_cov.explained_variance_}")
    print(f"SVD Explained Variance:        {pca_svd.explained_variance_}")
    # SVD and Covariance eigenvalues must match within floating point precision
    assert np.allclose(pca_cov.explained_variance_, pca_svd.explained_variance_)
    
    # Check transformed coordinates agreement (allowing for sign flips along eigenvectors)
    Z_cov = pca_cov.transform(X_synth)
    Z_svd = pca_svd.transform(X_synth)
    assert np.allclose(np.abs(Z_cov), np.abs(Z_svd))
    print("SVD and Covariance Eigendecomposition agree exactly!")
    print("C05 Sanity Check: PASSED cleanly.")

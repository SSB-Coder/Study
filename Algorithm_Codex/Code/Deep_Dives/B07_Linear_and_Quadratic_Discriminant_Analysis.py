"""
B07. Linear and Quadratic Discriminant Analysis (LDA & QDA)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LinearDiscriminantAnalysisScratch:
    """LDA with shared pooled covariance matrix."""
    def __init__(self, reg: float = 1e-4):
        self.reg = reg
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.cov_inv_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        N, d = X.shape
        K = len(self.classes_)

        self.priors_ = np.zeros(K)
        self.means_ = np.zeros((K, d))
        pooled_cov = np.zeros((d, d))

        for k, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            N_k = len(X_c)
            self.priors_[k] = N_k / N
            self.means_[k, :] = np.mean(X_c, axis=0) # <- eq. 3.2
            diff = X_c - self.means_[k, :]
            pooled_cov += diff.T @ diff

        # Unbiased pooled covariance <- eq. 3.2
        pooled_cov /= (N - K)
        # Add regularization for numerical stability
        pooled_cov += self.reg * np.eye(d)
        self.cov_inv_ = np.linalg.pinv(pooled_cov)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        scores = []
        for k in range(len(self.classes_)):
            mu_k = self.means_[k, :]
            prior = self.priors_[k]
            # delta_k(x) = x @ Sigma^(-1) @ mu_k - 0.5 * mu_k @ Sigma^(-1) @ mu_k + ln(prior) <- eq. 3.2
            w_k = self.cov_inv_ @ mu_k
            b_k = -0.5 * (mu_k @ w_k) + np.log(prior)
            scores.append(X @ w_k + b_k)
        scores = np.vstack(scores).T
        return self.classes_[np.argmax(scores, axis=1)]

class QuadraticDiscriminantAnalysisScratch:
    """QDA with per-class covariance matrices."""
    def __init__(self, reg: float = 1e-4):
        self.reg = reg
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.cov_invs_ = []
        self.log_dets_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        N, d = X.shape
        K = len(self.classes_)

        self.priors_ = np.zeros(K)
        self.means_ = np.zeros((K, d))
        self.cov_invs_ = []
        self.log_dets_ = []

        for k, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            N_k = len(X_c)
            self.priors_[k] = N_k / N
            self.means_[k, :] = np.mean(X_c, axis=0)
            diff = X_c - self.means_[k, :]
            cov_k = (diff.T @ diff) / (N_k - 1) + self.reg * np.eye(d) # <- eq. 3.3
            self.cov_invs_.append(np.linalg.pinv(cov_k))
            # SVD-based pseudo log-determinant
            s = np.linalg.svd(cov_k, compute_uv=False)
            self.log_dets_.append(np.sum(np.log(s[s > 0])))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        scores = []
        for k in range(len(self.classes_)):
            diff = X - self.means_[k, :]
            cov_inv = self.cov_invs_[k]
            log_det = self.log_dets_[k]
            prior = self.priors_[k]
            # delta_k(x) = -0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^(-1) (x - mu_k) + ln(pi_k) <- eq. 3.3
            quad_term = np.sum((diff @ cov_inv) * diff, axis=1)
            score_k = -0.5 * log_det - 0.5 * quad_term + np.log(prior)
            scores.append(score_k)
        scores = np.vstack(scores).T
        return self.classes_[np.argmax(scores, axis=1)]

if __name__ == "__main__":
    print("--- B07: LDA & QDA Sanity Check ---")
    np.random.seed(42)
    # Generate two 2D Gaussian clusters with distinct covariances
    cov1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov2 = np.array([[2.0, -0.8], [-0.8, 1.5]])
    X1 = np.random.multivariate_normal([2, 2], cov1, size=60)
    X2 = np.random.multivariate_normal([-2, -2], cov2, size=60)
    X_da = np.vstack([X1, X2])
    y_da = np.array([0]*60 + [1]*60)

    lda = LinearDiscriminantAnalysisScratch().fit(X_da, y_da)
    qda = QuadraticDiscriminantAnalysisScratch().fit(X_da, y_da)
    
    acc_lda = np.mean(lda.predict(X_da) == y_da)
    acc_qda = np.mean(qda.predict(X_da) == y_da)
    print(f"LDA Accuracy: {acc_lda * 100:.2f}%")
    print(f"QDA Accuracy: {acc_qda * 100:.2f}%")
    assert acc_lda > 0.95
    assert acc_qda > 0.95
    print("B07 Sanity Check: PASSED cleanly.")

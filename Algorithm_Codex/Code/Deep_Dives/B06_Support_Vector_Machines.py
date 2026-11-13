"""
B06. Support Vector Machines (SMO Solver, Kernels, and SVR)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class SVMClassifierScratch:
    """Support Vector Machine with simplified Platt SMO solver and RBF/Linear kernels."""
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = 0.5, 
                 degree: int = 3, tol: float = 1e-3, max_passes: int = 20):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel_name == "linear":
            return X1 @ X2.T
        elif self.kernel_name == "poly":
            return (X1 @ X2.T + 1.0) ** self.degree
        elif self.kernel_name == "rbf":
            # RBF: exp(-gamma * ||x1 - x2||^2) <- eq. 3.2
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
            dist_sq = X1_sq - 2.0 * (X1 @ X2.T) + X2_sq.T
            return np.exp(-self.gamma * np.maximum(0.0, dist_sq))
        raise ValueError(f"Unknown kernel {self.kernel_name}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Ensure labels are -1 and +1
        y = np.where(y <= 0, -1.0, 1.0)
        m, d = X.shape

        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(m)
        self.b = 0.0

        K = self._kernel(X, X)
        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # E_i = f(x_i) - y_i
                f_xi = np.sum(self.alphas * self.y_train * K[:, i]) + self.b
                E_i = f_xi - self.y_train[i]

                # Check KKT conditions
                if ((self.y_train[i] * E_i < -self.tol and self.alphas[i] < self.C) or
                    (self.y_train[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Randomly pick j != i
                    j = np.random.choice([idx for idx in range(m) if idx != i])
                    f_xj = np.sum(self.alphas * self.y_train * K[:, j]) + self.b
                    E_j = f_xj - self.y_train[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Compute L and H <- eq. 3.3
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0.0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if abs(L - H) < 1e-7:
                        continue

                    # eta = 2*K_ij - K_ii - K_jj
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Unclipped and clipped update for alpha_j
                    alpha_j_new = alpha_j_old - (self.y_train[j] * (E_i - E_j)) / eta
                    alpha_j_clipped = min(H, max(L, alpha_j_new))

                    if abs(alpha_j_clipped - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha_i_new = alpha_i_old + self.y_train[i] * self.y_train[j] * (alpha_j_old - alpha_j_clipped)

                    # Update threshold b
                    b1 = self.b - E_i - self.y_train[i] * (alpha_i_new - alpha_i_old) * K[i, i]                                      - self.y_train[j] * (alpha_j_clipped - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y_train[i] * (alpha_i_new - alpha_i_old) * K[i, j]                                      - self.y_train[j] * (alpha_j_clipped - alpha_j_old) * K[j, j]

                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_clipped < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    self.alphas[i] = alpha_i_new
                    self.alphas[j] = alpha_j_clipped
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        K_test = self._kernel(X, self.X_train)
        return np.sum(K_test * (self.alphas * self.y_train), axis=1) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, 0)

if __name__ == "__main__":
    print("--- B06: Support Vector Machines Sanity Check ---")
    np.random.seed(42)
    # Linearly separable clusters with small noise
    m = 60
    X_pos = np.random.randn(m // 2, 2) * 0.5 + np.array([1.5, 1.5])
    X_neg = np.random.randn(m // 2, 2) * 0.5 + np.array([-1.5, -1.5])
    X_svm = np.vstack([X_pos, X_neg])
    y_svm = np.array([1]*(m//2) + [0]*(m//2))

    svm = SVMClassifierScratch(C=1.0, kernel="linear", max_passes=10).fit(X_svm, y_svm)
    preds = svm.predict(X_svm)
    acc = np.mean(preds == y_svm)
    print(f"Linear SVM Accuracy: {acc * 100:.2f}%")
    n_sv = np.sum(svm.alphas > 1e-4)
    print(f"Support Vectors count: {n_sv} / {m}")
    assert acc > 0.95
    assert n_sv > 0
    print("B06 Sanity Check: PASSED cleanly.")

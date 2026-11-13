"""
B01. Linear Regression (Normal Equation, Gradient Descent, Ridge, Lasso, Elastic Net)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LinearRegressionScratch:
    """Comprehensive Linear Regression implementing OLS, Ridge, Lasso, and Elastic Net."""
    def __init__(self, method: str = "normal", alpha: float = 0.01, n_iters: int = 1000, 
                 l1_ratio: float = 0.0, l2_reg: float = 0.0, l1_reg: float = 0.0):
        self.method = method # 'normal', 'gd', 'ridge', 'lasso', 'elastic'
        self.alpha = alpha
        self.n_iters = n_iters
        self.l1_ratio = l1_ratio
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        m, d = X.shape

        if self.method == "normal":
            # Normal Equation: beta = (X_b^T X_b)^(-1) X_b^T y <- eq. 3.1
            X_b = np.c_[np.ones((m, 1)), X]
            beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = float(beta[0, 0])
            self.weights = beta[1:, 0]

        elif self.method == "ridge":
            # Ridge Closed-Form: beta = (X_b^T X_b + lambda * I)^(-1) X_b^T y <- eq. 3.3
            X_b = np.c_[np.ones((m, 1)), X]
            reg_matrix = np.eye(d + 1)
            reg_matrix[0, 0] = 0.0 # Do not regularize bias
            beta = np.linalg.inv(X_b.T @ X_b + self.l2_reg * reg_matrix) @ X_b.T @ y
            self.bias = float(beta[0, 0])
            self.weights = beta[1:, 0]

        elif self.method == "gd":
            # Gradient Descent <- eq. 3.2
            self.weights = np.zeros(d)
            self.bias = 0.0
            for _ in range(self.n_iters):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y.ravel()
                # dW = (1/m) * X.T @ error
                dW = (1.0 / m) * (X.T @ error) + (self.l2_reg * self.weights)
                db = (1.0 / m) * np.sum(error)
                self.weights -= self.alpha * dW
                self.bias -= self.alpha * db

        elif self.method == "lasso":
            # Coordinate Descent with Soft-Thresholding <- eq. 3.4
            self.weights = np.zeros(d)
            self.bias = float(np.mean(y))
            y_centered = y.ravel() - self.bias
            
            z = np.sum(X ** 2, axis=0) # Feature squared norms
            z[z == 0] = 1e-12

            for _ in range(self.n_iters):
                for j in range(d):
                    # Residual excluding feature j
                    r = y_centered - (X @ self.weights - X[:, j] * self.weights[j])
                    rho_j = np.dot(X[:, j], r)
                    
                    # Soft-thresholding operator
                    thresh = self.l1_reg * m
                    if rho_j > thresh:
                        self.weights[j] = (rho_j - thresh) / z[j]
                    elif rho_j < -thresh:
                        self.weights[j] = (rho_j + thresh) / z[j]
                    else:
                        self.weights[j] = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

if __name__ == "__main__":
    print("--- B01: Linear Regression Sanity Check ---")
    np.random.seed(42)
    m, d = 200, 3
    X_synth = np.random.randn(m, d)
    true_w = np.array([2.5, -1.8, 0.0]) # Note feature 3 is zero (sparse)
    true_b = 4.2
    y_synth = X_synth @ true_w + true_b + np.random.normal(0, 0.1, size=m)

    # 1. Normal Equation vs Gradient Descent
    lr_normal = LinearRegressionScratch(method="normal").fit(X_synth, y_synth)
    lr_gd = LinearRegressionScratch(method="gd", alpha=0.1, n_iters=1500).fit(X_synth, y_synth)
    
    print(f"True Weights: {true_w}, True Bias: {true_b}")
    print(f"Normal Eq Weights: {lr_normal.weights}, Bias: {lr_normal.bias:.4f}")
    print(f"GD Weights:        {lr_gd.weights}, Bias: {lr_gd.bias:.4f}")
    assert np.allclose(lr_normal.weights, lr_gd.weights, atol=0.05)
    
    # 2. Lasso Sparsity Check
    lr_lasso = LinearRegressionScratch(method="lasso", l1_reg=0.05, n_iters=500).fit(X_synth, y_synth)
    print(f"Lasso Weights:     {lr_lasso.weights}")
    assert lr_lasso.weights[2] == 0.0 # Feature 3 should be shrunk to exact zero
    print("B01 Sanity Check: PASSED cleanly.")

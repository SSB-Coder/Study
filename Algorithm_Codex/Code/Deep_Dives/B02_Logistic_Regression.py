"""
B02. Logistic Regression (Binary & Softmax Multinomial)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LogisticRegressionScratch:
    """Binary and Multinomial Logistic Regression via batch gradient descent."""
    def __init__(self, lr: float = 0.1, n_iters: int = 1000, l2_reg: float = 1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.W = None
        self.b = None
        self.is_multinomial = False
        self.classes_ = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid <- eq. 3.1
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        # Numerically stable softmax <- eq. 3.2
        Z_shift = Z - np.max(Z, axis=-1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        m, d = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            self.is_multinomial = False
            y_binary = np.where(y == self.classes_[1], 1.0, 0.0).reshape(-1, 1)
            self.W = np.zeros((d, 1))
            self.b = 0.0

            for _ in range(self.n_iters):
                z = X @ self.W + self.b
                y_hat = self._sigmoid(z)
                # Gradient: dW = (1/m) * X.T @ (y_hat - y) <- eq. 3.1
                residual = y_hat - y_binary
                dW = (1.0 / m) * (X.T @ residual) + self.l2_reg * self.W
                db = float((1.0 / m) * np.sum(residual))
                self.W -= self.lr * dW
                self.b -= self.lr * db
        else:
            self.is_multinomial = True
            # One-hot encoding
            Y_onehot = np.zeros((m, n_classes))
            for k, cls in enumerate(self.classes_):
                Y_onehot[y == cls, k] = 1.0

            self.W = np.zeros((d, n_classes))
            self.b = np.zeros((1, n_classes))

            for _ in range(self.n_iters):
                Z = X @ self.W + self.b
                P = self._softmax(Z)
                # Gradient: dW = (1/m) * X.T @ (P - Y) <- eq. 3.2
                residual = P - Y_onehot
                dW = (1.0 / m) * (X.T @ residual) + self.l2_reg * self.W
                db = (1.0 / m) * np.sum(residual, axis=0, keepdims=True)
                self.W -= self.lr * dW
                self.b -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.is_multinomial:
            p1 = self._sigmoid(X @ self.W + self.b)
            return np.hstack([1.0 - p1, p1])
        else:
            return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]

if __name__ == "__main__":
    print("--- B02: Logistic Regression Sanity Check ---")
    np.random.seed(42)
    # Binary classification synthetic data
    m = 150
    X_bin = np.random.randn(m, 2)
    y_bin = (X_bin[:, 0] + X_bin[:, 1] > 0).astype(int)

    clf_bin = LogisticRegressionScratch(lr=0.5, n_iters=1000).fit(X_bin, y_bin)
    preds_bin = clf_bin.predict(X_bin)
    acc_bin = np.mean(preds_bin == y_bin)
    print(f"Binary Classification Accuracy: {acc_bin * 100:.2f}%")
    assert acc_bin > 0.90

    # Multinomial 3-class synthetic data
    X_multi = np.vstack([
        np.random.randn(50, 2) + np.array([-2, -2]),
        np.random.randn(50, 2) + np.array([2, 2]),
        np.random.randn(50, 2) + np.array([-2, 2])
    ])
    y_multi = np.array([0]*50 + [1]*50 + [2]*50)
    clf_multi = LogisticRegressionScratch(lr=0.5, n_iters=1000).fit(X_multi, y_multi)
    preds_multi = clf_multi.predict(X_multi)
    acc_multi = np.mean(preds_multi == y_multi)
    print(f"Multinomial Classification Accuracy: {acc_multi * 100:.2f}%")
    assert acc_multi > 0.90
    print("B02 Sanity Check: PASSED cleanly.")

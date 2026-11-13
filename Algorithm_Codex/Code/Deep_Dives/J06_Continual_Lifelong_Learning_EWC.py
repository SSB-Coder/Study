"""
J06: Continual Lifelong Learning (Elastic Weight Consolidation - EWC)
Pure NumPy implementation of Kirkpatrick et al. (PNAS 2017).
Demonstrates catastrophic forgetting and its mitigation via diagonal
Fisher Information Matrix quadratic penalties.
"""

import numpy as np

class EWCModel:
    """
    MLP classifier with Elastic Weight Consolidation.
    Architecture: input_dim -> hidden_dim (tanh) -> num_classes (softmax)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, lr: float = 0.05):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(num_classes)

        # Conserved optimal parameters and Fisher information for past tasks
        self.optimal_params = None
        self.fisher_diag = None

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        # Softmax
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs, (X, z1, a1, probs)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, ewc_lambda: float = 0.0):
        N = X.shape[0]
        probs, (X_in, z1, a1, _) = self.forward(X)

        # Categorical cross-entropy gradient
        d_out = probs.copy()
        d_out[np.arange(N), y] -= 1.0
        d_out /= N

        dW2 = a1.T @ d_out
        db2 = np.sum(d_out, axis=0)

        da1 = d_out @ self.W2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        dW1 = X_in.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Add EWC quadratic penalty gradient: lambda * F_ii * (theta - theta_A*)
        if ewc_lambda > 0.0 and self.optimal_params is not None:
            opt_W1, opt_b1, opt_W2, opt_b2 = self.optimal_params
            F_W1, F_b1, F_W2, F_b2 = self.fisher_diag

            dW1 += ewc_lambda * F_W1 * (self.W1 - opt_W1)
            db1 += ewc_lambda * F_b1 * (self.b1 - opt_b1)
            dW2 += ewc_lambda * F_W2 * (self.W2 - opt_W2)
            db2 += ewc_lambda * F_b2 * (self.b2 - opt_b2)

        # Gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def compute_fisher_information(self, X: np.ndarray, y: np.ndarray):
        """Computes empirical diagonal Fisher Information Matrix F_ii."""
        N = X.shape[0]
        probs, (X_in, z1, a1, _) = self.forward(X)

        F_W1 = np.zeros_like(self.W1)
        F_b1 = np.zeros_like(self.b1)
        F_W2 = np.zeros_like(self.W2)
        F_b2 = np.zeros_like(self.b2)

        # Compute expectation over squared sample gradients
        for i in range(N):
            d_out = probs[i:i+1].copy()
            d_out[0, y[i]] -= 1.0

            dW2_i = a1[i:i+1].T @ d_out
            db2_i = d_out[0]

            da1_i = d_out @ self.W2.T
            dz1_i = da1_i * (1.0 - a1[i:i+1] ** 2)
            dW1_i = X_in[i:i+1].T @ dz1_i
            db1_i = dz1_i[0]

            F_W1 += dW1_i ** 2
            F_b1 += db1_i ** 2
            F_W2 += dW2_i ** 2
            F_b2 += db2_i ** 2

        self.fisher_diag = (F_W1 / N, F_b1 / N, F_W2 / N, F_b2 / N)
        self.optimal_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        probs, _ = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == y))


if __name__ == "__main__":
    print("--- J06: Continual Learning & EWC Sanity Check ---")
    np.random.seed(42)

    D, K = 8, 3
    # Task A dataset
    X_A = np.random.randn(150, D)
    y_A = np.random.randint(0, K, size=150)

    # Task B dataset (permuted input features, different manifold)
    perm = np.random.permutation(D)
    X_B = X_A[:, perm] + np.random.randn(150, D) * 0.5
    y_B = (y_A + 1) % K

    # 1. Train on Task A
    model_ewc = EWCModel(input_dim=D, hidden_dim=24, num_classes=K, lr=0.1)
    for _ in range(50):
        model_ewc.train_epoch(X_A, y_A)
    acc_A_initial = model_ewc.evaluate(X_A, y_A)
    print(f"Task A Initial Accuracy: {acc_A_initial * 100:.1f}%")

    # Compute Fisher Information Matrix on Task A
    model_ewc.compute_fisher_information(X_A, y_A)

    # 2. Train on Task B with EWC protection (lambda = 20.0)
    for _ in range(50):
        model_ewc.train_epoch(X_B, y_B, ewc_lambda=20.0)

    acc_A_post_ewc = model_ewc.evaluate(X_A, y_A)
    acc_B_post_ewc = model_ewc.evaluate(X_B, y_B)
    print(f"Post-Task B with EWC -> Task A Acc: {acc_A_post_ewc * 100:.1f}% | Task B Acc: {acc_B_post_ewc * 100:.1f}%")

    # Baseline comparison: model WITHOUT EWC shows catastrophic forgetting
    model_naive = EWCModel(input_dim=D, hidden_dim=24, num_classes=K, lr=0.1)
    model_naive.W1, model_naive.b1 = model_ewc.optimal_params[0].copy(), model_ewc.optimal_params[1].copy()
    model_naive.W2, model_naive.b2 = model_ewc.optimal_params[2].copy(), model_ewc.optimal_params[3].copy()
    for _ in range(50):
        model_naive.train_epoch(X_B, y_B, ewc_lambda=0.0)
    acc_A_naive = model_naive.evaluate(X_A, y_A)
    print(f"Post-Task B WITHOUT EWC (Naive) -> Task A Acc: {acc_A_naive * 100:.1f}% (Catastrophic Forgetting)")

    assert acc_A_post_ewc > acc_A_naive, "EWC must preserve Task A performance significantly better than naive training"
    print("J06 Sanity Check: PASSED cleanly.")

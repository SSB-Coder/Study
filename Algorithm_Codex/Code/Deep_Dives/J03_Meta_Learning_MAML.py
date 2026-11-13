"""
J03: Meta-Learning (Model-Agnostic Meta-Learning - MAML)
Pure NumPy implementation of Finn, Abbeel, & Levine (ICML 2017)
for few-shot regression on sinusoidal task distributions.
"""

import numpy as np

class MAMLRegressor:
    """
    Two-layer MLP meta-learner trained to adapt in 1 gradient step to new sine tasks.
    Architecture: 1 -> hidden_dim (tanh) -> 1
    """
    def __init__(self, hidden_dim: int = 32, alpha: float = 0.02, beta: float = 0.005):
        self.alpha = alpha # Inner loop adaptation learning rate
        self.beta = beta   # Outer loop meta-optimization learning rate

        # Meta-parameters theta = {W1, b1, W2, b2}
        self.W1 = np.random.randn(1, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.5
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray, W1, b1, W2, b2):
        z1 = x @ W1 + b1
        a1 = np.tanh(z1)
        out = a1 @ W2 + b2
        return out, (x, z1, a1)

    def compute_loss_and_grads(self, x: np.ndarray, y: np.ndarray, W1, b1, W2, b2):
        n = x.shape[0]
        pred, (x_in, z1, a1) = self.forward(x, W1, b1, W2, b2)
        diff = pred - y
        loss = 0.5 * np.mean(diff ** 2)

        # Backpropagation
        d_out = diff / n
        dW2 = a1.T @ d_out
        db2 = np.sum(d_out, axis=0)

        da1 = d_out @ W2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        dW1 = x_in.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return loss, (dW1, db1, dW2, db2)

    def inner_adapt(self, x_supp: np.ndarray, y_supp: np.ndarray, num_steps: int = 1):
        """Inner loop: single or multi-step task adaptation from meta-parameters theta."""
        W1, b1, W2, b2 = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()
        for _ in range(num_steps):
            _, (dW1, db1, dW2, db2) = self.compute_loss_and_grads(x_supp, y_supp, W1, b1, W2, b2)
            W1 -= self.alpha * dW1
            b1 -= self.alpha * db1
            W2 -= self.alpha * dW2
            b2 -= self.alpha * db2
        return W1, b1, W2, b2

    def meta_train_step(self, tasks: list):
        """
        Outer loop: First-Order MAML (FOMAML) meta-update across task query sets.
        Each task is a tuple: (x_supp, y_supp, x_query, y_query)
        """
        meta_dW1 = np.zeros_like(self.W1)
        meta_db1 = np.zeros_like(self.b1)
        meta_dW2 = np.zeros_like(self.W2)
        meta_db2 = np.zeros_like(self.b2)

        total_query_loss = 0.0

        for x_supp, y_supp, x_query, y_query in tasks:
            # 1. Inner adaptation
            W1_p, b1_p, W2_p, b2_p = self.inner_adapt(x_supp, y_supp, num_steps=1)

            # 2. Query loss and gradients on adapted parameters
            q_loss, (dW1, db1, dW2, db2) = self.compute_loss_and_grads(
                x_query, y_query, W1_p, b1_p, W2_p, b2_p
            )
            total_query_loss += q_loss

            # First-order MAML accumulates dL_query / dtheta' directly
            meta_dW1 += dW1
            meta_db1 += db1
            meta_dW2 += dW2
            meta_db2 += db2

        n_tasks = len(tasks)
        self.W1 -= self.beta * (meta_dW1 / n_tasks)
        self.b1 -= self.beta * (meta_db1 / n_tasks)
        self.W2 -= self.beta * (meta_dW2 / n_tasks)
        self.b2 -= self.beta * (meta_db2 / n_tasks)

        return total_query_loss / n_tasks


def generate_sine_task():
    """Generates a sinusoidal task: y = A * sin(x + phase)."""
    A = np.random.uniform(0.5, 3.0)
    phase = np.random.uniform(0.0, np.pi)

    # 10 support points, 15 query points in [-5, 5]
    x_all = np.random.uniform(-5.0, 5.0, size=(25, 1))
    y_all = A * np.sin(x_all + phase)

    return x_all[:10], y_all[:10], x_all[10:], y_all[10:]


if __name__ == "__main__":
    print("--- J03: Meta-Learning (MAML) Sanity Check ---")
    np.random.seed(42)

    maml = MAMLRegressor(hidden_dim=32, alpha=0.03, beta=0.01)

    # Meta-train over 80 iterations
    for it in range(80):
        batch_tasks = [generate_sine_task() for _ in range(8)]
        q_loss = maml.meta_train_step(batch_tasks)

    # Evaluate on a held-out test task
    x_s, y_s, x_q, y_q = generate_sine_task()

    # Pre-adaptation query loss
    pre_pred, _ = maml.forward(x_q, maml.W1, maml.b1, maml.W2, maml.b2)
    pre_mse = float(0.5 * np.mean((pre_pred - y_q) ** 2))

    # 1-step adapted query loss
    W1_p, b1_p, W2_p, b2_p = maml.inner_adapt(x_s, y_s, num_steps=1)
    post_pred, _ = maml.forward(x_q, W1_p, b1_p, W2_p, b2_p)
    post_mse = float(0.5 * np.mean((post_pred - y_q) ** 2))

    print(f"Test Task Few-Shot Adaptation: Pre-adapt MSE = {pre_mse:.4f} -> Post-adapt MSE = {post_mse:.4f}")
    assert post_mse < pre_mse, "Few-shot inner-loop adaptation must reduce test query error"

    print("J03 Sanity Check: PASSED cleanly.")

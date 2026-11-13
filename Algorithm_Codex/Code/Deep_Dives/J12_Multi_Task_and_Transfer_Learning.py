"""
J12: Multi-Task & Transfer Learning (Shared Trunk & MMD Domain Distance)
Pure NumPy implementation of Multi-Task Learning and Maximum Mean Discrepancy (MMD).
"""

import numpy as np

def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD^2) with RBF kernel between samples X and Y.
    MMD^2 = E[k(X,X)] - 2*E[k(X,Y)] + E[k(Y,Y)]
    """
    n = X.shape[0]
    m = Y.shape[0]

    def rbf_k(A, B):
        diff = A[:, None, :] - B[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-gamma * dist_sq)

    k_xx = rbf_k(X, X)
    k_yy = rbf_k(Y, Y)
    k_xy = rbf_k(X, Y)

    mmd2 = (np.sum(k_xx) / (n * n)) - 2.0 * (np.sum(k_xy) / (n * m)) + (np.sum(k_yy) / (m * m))
    return float(max(0.0, mmd2))


class MultiTaskLinearTrunk:
    """Hard parameter sharing multi-task regressor with 2 heads."""
    def __init__(self, in_dim: int, hidden_dim: int, lr: float = 0.05):
        rng = np.random.default_rng(42)
        self.W_shared = rng.normal(0, 0.5, size=(hidden_dim, in_dim))
        self.w1 = rng.normal(0, 0.5, size=hidden_dim)
        self.w2 = rng.normal(0, 0.5, size=hidden_dim)
        self.lr = lr

    def forward(self, x: np.ndarray):
        z = np.maximum(0, self.W_shared @ x)  # ReLU
        y1 = float(np.dot(self.w1, z))
        y2 = float(np.dot(self.w2, z))
        return y1, y2, z

    def train_step(self, x: np.ndarray, y1_true: float, y2_true: float):
        y1, y2, z = self.forward(x)
        err1 = y1 - y1_true
        err2 = y2 - y2_true

        # Gradients w.r.t heads
        grad_w1 = err1 * z
        grad_w2 = err2 * z

        # Gradients w.r.t shared trunk
        dz = err1 * self.w1 + err2 * self.w2
        d_relu = dz * (z > 0)
        grad_W_shared = np.outer(d_relu, x)

        # Update
        self.w1 -= self.lr * grad_w1
        self.w2 -= self.lr * grad_w2
        self.W_shared -= self.lr * grad_W_shared


if __name__ == "__main__":
    print("--- J12: Multi-Task Learning & MMD Sanity Check ---")
    np.random.seed(42)

    # 1. Test Maximum Mean Discrepancy (MMD)
    # Samples from identical distribution: Normal(0, 1)
    dist_same_1 = np.random.normal(0, 1, size=(60, 2))
    dist_same_2 = np.random.normal(0, 1, size=(60, 2))
    # Samples from shifted distribution: Normal(3, 1)
    dist_shifted = np.random.normal(3, 1, size=(60, 2))

    mmd_same = compute_mmd(dist_same_1, dist_same_2, gamma=0.5)
    mmd_shifted = compute_mmd(dist_same_1, dist_shifted, gamma=0.5)

    print(f"MMD between identical distributions: {mmd_same:.5f}")
    print(f"MMD between shifted distributions:   {mmd_shifted:.5f}")

    assert mmd_same < 0.05, f"MMD between identical distributions should be near 0, got {mmd_same}"
    assert mmd_shifted > 0.50, f"MMD between shifted distributions must be high, got {mmd_shifted}"

    # 2. Test Multi-Task Shared Trunk
    mtl = MultiTaskLinearTrunk(in_dim=2, hidden_dim=4, lr=0.1)
    # Task 1: y1 = 2*x0 + x1
    # Task 2: y2 = -x0 + 3*x1
    for _ in range(400):
        x_sample = np.random.uniform(-1, 1, size=2)
        t1 = 2.0 * x_sample[0] + x_sample[1]
        t2 = -x_sample[0] + 3.0 * x_sample[1]
        mtl.train_step(x_sample, t1, t2)

    test_x = np.array([0.5, -0.5])
    true_y1 = 2.0 * 0.5 - 0.5  # 0.5
    true_y2 = -0.5 + 3.0 * (-0.5)  # -2.0
    pred_y1, pred_y2, _ = mtl.forward(test_x)

    print(f"Task 1 Target: {true_y1:.2f} | Predicted: {pred_y1:.2f}")
    print(f"Task 2 Target: {true_y2:.2f} | Predicted: {pred_y2:.2f}")

    assert abs(pred_y1 - true_y1) < 0.25, "Head 1 must converge"
    assert abs(pred_y2 - true_y2) < 0.25, "Head 2 must converge"
    print("J12 Sanity Check: PASSED cleanly.")

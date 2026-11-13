"""
J02: Self-Supervised Contrastive Learning (InfoNCE / SimCLR)
Pure NumPy implementation of InfoNCE contrastive loss on a hand-derived
projection network with explicit matrix calculus backpropagation.
"""

import numpy as np

class ContrastiveEncoder:
    """
    Two-layer MLP projection head trained via InfoNCE contrastive loss.
    Architecture: input_dim -> hidden_dim (tanh) -> proj_dim (L2 normalized)
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, lr: float = 0.05, tau: float = 0.1):
        self.lr = lr
        self.tau = tau
        # He/Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, proj_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(proj_dim)

    def forward(self, X: np.ndarray):
        """Forward pass with L2-normalized projection representations."""
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        # L2 normalize embeddings onto unit hypersphere
        norm = np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-12
        Z_proj = Z2 / norm
        return Z_proj, (X, Z1, A1, Z2, norm)

    def compute_infonce_loss_and_grad(self, Z_a: np.ndarray, Z_b: np.ndarray):
        """
        Computes InfoNCE loss between view A and view B for batch size N.
        Positive pair: (Z_a[i], Z_b[i]). Negatives: all other batch entries.
        """
        N = Z_a.shape[0]
        # Cosine similarity matrix between all pairs: S[i, j] = Z_a[i] @ Z_b[j] / tau
        S = (Z_a @ Z_b.T) / self.tau

        # Numerically stable softmax probabilities
        S_max = np.max(S, axis=1, keepdims=True)
        exp_S = np.exp(S - S_max)
        denom = np.sum(exp_S, axis=1, keepdims=True)
        P = exp_S / denom # shape (N, N)

        # Loss: - 1/N sum_i log P[i, i]
        pos_probs = np.clip(np.diag(P), 1e-15, 1.0)
        loss = -np.mean(np.log(pos_probs))

        # Gradient wrt Z_a: dL/dZ_a[i] = 1/(N * tau) * sum_j (P[i, j] - delta_{ij}) * Z_b[j]
        grad_Za = np.zeros_like(Z_a)
        for i in range(N):
            p_i = P[i].copy()
            p_i[i] -= 1.0 # (P - I)
            grad_Za[i] = (p_i @ Z_b) / (N * self.tau)

        return loss, grad_Za

    def backward(self, grad_Z_proj: np.ndarray, cache):
        """Backpropagates through L2 normalization and 2-layer MLP."""
        X, Z1, A1, Z2, norm = cache

        # Gradient through L2 normalization:
        # d/dZ2 [Z2 / ||Z2||] = (I - Z_proj Z_proj^T) / ||Z2||
        # Vectorized form: grad_Z2 = (grad_Z_proj - Z_proj * sum(grad_Z_proj * Z_proj, axis=1, keepdims=True)) / norm
        Z_proj = Z2 / norm
        proj_dot = np.sum(grad_Z_proj * Z_proj, axis=1, keepdims=True)
        grad_Z2 = (grad_Z_proj - Z_proj * proj_dot) / norm

        # Layer 2 gradients
        dW2 = A1.T @ grad_Z2
        db2 = np.sum(grad_Z2, axis=0)

        # Layer 1 gradients
        grad_A1 = grad_Z2 @ self.W2.T
        grad_Z1 = grad_A1 * (1.0 - A1**2) # d/dz tanh(z) = 1 - tanh^2(z)
        dW1 = X.T @ grad_Z1
        db1 = np.sum(grad_Z1, axis=0)

        # Gradient descent parameter updates
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_step(self, X_a: np.ndarray, X_b: np.ndarray):
        Za, cache_a = self.forward(X_a)
        Zb, cache_b = self.forward(X_b)
        loss, grad_Za = self.compute_infonce_loss_and_grad(Za, Zb)
        self.backward(grad_Za, cache_a)
        return loss


if __name__ == "__main__":
    print("--- J02: Self-Supervised Contrastive Learning (InfoNCE) Sanity Check ---")
    np.random.seed(42)

    N, D = 64, 16
    # Create base semantic clusters
    cluster_centers = np.random.randn(4, D) * 2.0
    labels = np.random.randint(0, 4, size=N)
    X_base = cluster_centers[labels]

    # Two stochastic augmentations: random additive noise and feature dropout
    X_view1 = X_base + np.random.randn(N, D) * 0.1
    X_view2 = X_base + np.random.randn(N, D) * 0.1

    model = ContrastiveEncoder(input_dim=D, hidden_dim=32, proj_dim=8, lr=0.08, tau=0.1)

    initial_loss = None
    final_loss = None
    for epoch in range(100):
        loss = model.train_step(X_view1, X_view2)
        if epoch == 0:
            initial_loss = loss
        final_loss = loss

    print(f"InfoNCE Loss: Initial = {initial_loss:.4f} -> Final = {final_loss:.4f}")
    assert final_loss < initial_loss, "Contrastive optimization must decrease InfoNCE loss"

    # Positive vs negative cosine similarities
    Za, _ = model.forward(X_view1)
    Zb, _ = model.forward(X_view2)
    pos_sims = np.diag(Za @ Zb.T)
    neg_mask = ~np.eye(N, dtype=bool)
    neg_sims = (Za @ Zb.T)[neg_mask]

    mean_pos = float(np.mean(pos_sims))
    mean_neg = float(np.mean(neg_sims))
    print(f"Mean Positive Pair Similarity: {mean_pos:.4f} | Mean Negative Pair Similarity: {mean_neg:.4f}")
    assert mean_pos > mean_neg, "Positive representations must be significantly closer than negative pairs"

    print("J02 Sanity Check: PASSED cleanly.")

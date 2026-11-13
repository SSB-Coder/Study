"""
K04: Capsule Networks (Dynamic Routing-by-Agreement)
Pure NumPy implementation of Sabour, Frosst, & Hinton (NeurIPS 2017).
Vector-valued capsule activations with squashing non-linearity.
"""

import numpy as np

def squash(s: np.ndarray, axis: int = -1) -> np.ndarray:
    """Non-linear squashing function."""
    norm_sq = np.sum(s ** 2, axis=axis, keepdims=True)
    norm = np.sqrt(norm_sq) + 1e-12
    scale = norm_sq / (1.0 + norm_sq)
    return scale * (s / norm)


class CapsuleLayer:
    """
    Capsule layer with dynamic routing-by-agreement.
    Maps n_in capsules of dim d_in to n_out capsules of dim d_out.
    """
    def __init__(self, n_in: int, d_in: int, n_out: int, d_out: int, n_routing: int = 3):
        self.n_in = n_in
        self.d_in = d_in
        self.n_out = n_out
        self.d_out = d_out
        self.n_routing = n_routing

        # Transformation weight tensor: (n_in, n_out, d_out, d_in)
        self.W = np.random.randn(n_in, n_out, d_out, d_in) * 0.1

    def forward(self, u: np.ndarray) -> np.ndarray:
        B = u.shape[0]

        u_expanded = u[:, :, None, :, None] # (B, n_in, 1, d_in, 1)
        W_expanded = self.W[None, :, :, :, :] # (1, n_in, n_out, d_out, d_in)
        u_hat = np.squeeze(W_expanded @ u_expanded, axis=-1) # (B, n_in, n_out, d_out)

        b = np.zeros((B, self.n_in, self.n_out))

        v_j = None
        for r in range(self.n_routing):
            exp_b = np.exp(b - np.max(b, axis=2, keepdims=True))
            c = exp_b / np.sum(exp_b, axis=2, keepdims=True)

            s_j = np.sum(c[:, :, :, None] * u_hat, axis=1)
            v_j = squash(s_j, axis=-1)

            if r < self.n_routing - 1:
                agreement = np.sum(u_hat * v_j[:, None, :, :], axis=-1)
                b += agreement

        return v_j


if __name__ == "__main__":
    print("--- K04: Capsule Networks (Dynamic Routing) Sanity Check ---")
    np.random.seed(42)

    B = 4
    n_in, d_in = 8, 4
    n_out, d_out = 3, 6

    caps_layer = CapsuleLayer(n_in=n_in, d_in=d_in, n_out=n_out, d_out=d_out, n_routing=3)

    u = np.random.randn(B, n_in, d_in)
    v = caps_layer.forward(u)

    print(f"Input Capsule Tensor: {u.shape} -> Output Capsule Tensor: {v.shape}")

    capsule_lengths = np.linalg.norm(v, axis=-1)
    print(f"Output Capsule Lengths (sample 0): {capsule_lengths[0].round(3)}")

    assert v.shape == (B, n_out, d_out), "Output shape must match (B, n_out, d_out)"
    assert np.all(capsule_lengths >= 0.0) and np.all(capsule_lengths < 1.0), "Capsule norms must strictly lie in [0, 1)"

    print("K04 Sanity Check: PASSED cleanly.")

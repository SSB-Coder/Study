"""
J01: Semi-Supervised Learning & Label Propagation
Pure NumPy implementation of Harmonic Graph Label Propagation.
"""

import numpy as np

class LabelPropagation:
    """Graph-based Label Propagation using Harmonic Function closed-form solution."""
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def fit_predict(self, X: np.ndarray, y_labeled: dict[int, int], n_classes: int = 2) -> np.ndarray:
        # X: (N, d), y_labeled: {index: class_label}
        N = X.shape[0]
        labeled_indices = sorted(list(y_labeled.keys()))
        unlabeled_indices = sorted(list(set(range(N)) - set(labeled_indices)))

        l = len(labeled_indices)
        u = len(unlabeled_indices)

        # 1. Compute RBF Affinity matrix W
        diff = X[:, None, :] - X[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        W = np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        np.fill_diagonal(W, 0.0)

        # 2. Compute stochastic transition matrix P = D^(-1) W
        D = np.sum(W, axis=1)
        D[D == 0] = 1.0
        P = W / D[:, None]

        # 3. Rearrange into labeled and unlabeled blocks
        perm = labeled_indices + unlabeled_indices
        P_perm = P[perm, :][:, perm]
        P_uu = P_perm[l:, l:]
        P_ul = P_perm[l:, :l]

        # One-hot encoded labeled targets Y_l: (l, n_classes)
        Y_l = np.zeros((l, n_classes))
        for i, idx in enumerate(labeled_indices):
            Y_l[i, y_labeled[idx]] = 1.0

        # 4. Closed-form harmonic solution: Y_u = (I - P_uu)^(-1) * P_ul * Y_l
        I = np.eye(u)
        Y_u = np.linalg.solve(I - P_uu, P_ul @ Y_l)

        # Recombine full predictions
        final_preds = np.zeros(N, dtype=int)
        for i, idx in enumerate(labeled_indices):
            final_preds[idx] = y_labeled[idx]
        for i, idx in enumerate(unlabeled_indices):
            final_preds[idx] = int(np.argmax(Y_u[i]))

        return final_preds


if __name__ == "__main__":
    print("--- J01: Label Propagation Sanity Check ---")
    # Two distinct 2D Gaussian clusters: Class 0 at (-2, 0), Class 1 at (+2, 0)
    np.random.seed(42)
    c0 = np.random.normal(loc=[-2.0, 0.0], scale=0.4, size=(20, 2))
    c1 = np.random.normal(loc=[ 2.0, 0.0], scale=0.4, size=(20, 2))
    X = np.vstack([c0, c1])
    true_labels = np.array([0] * 20 + [1] * 20)

    # Provide only ONE labeled seed per cluster (2 labeled points out of 40 = 5% supervision)
    labeled_seeds = {0: 0, 20: 1}

    lp = LabelPropagation(sigma=1.0)
    preds = lp.fit_predict(X, labeled_seeds, n_classes=2)

    accuracy = np.mean(preds == true_labels)
    print(f"Propagated Accuracy with 5% Labeled Seeds: {accuracy:.2%}")

    assert accuracy >= 0.95, f"Label propagation should achieve >= 95% accuracy on separated clusters, got {accuracy}"
    print("J01 Sanity Check: PASSED cleanly.")

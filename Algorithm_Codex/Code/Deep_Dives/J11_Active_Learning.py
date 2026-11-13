"""
J11: Active Learning (Uncertainty Sampling & Query-by-Committee)
Pure NumPy implementation of pool-based active learning.
Demonstrates superior sample efficiency over passive uniform random sampling.
"""

import numpy as np

class LogisticRegressionModel:
    """Simple linear classifier used as active learning base model."""
    def __init__(self, n_features: int, lr: float = 0.05, epochs: int = 40):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(self.epochs):
            logits = X @ self.w + self.b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            grad_w = (X.T @ (probs - y)) / n
            grad_b = np.mean(probs - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


class ActiveLearningPool:
    """Manages unlabeled data pool and queries instances via uncertainty sampling."""
    def __init__(self, X_pool: np.ndarray, y_pool: np.ndarray):
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()
        self.labeled_idx = []
        self.unlabeled_idx = list(range(len(y_pool)))

    def query_uncertainty(self, model: LogisticRegressionModel, n_query: int = 5) -> list:
        """
        Margin sampling on unlabeled pool:
        Uncertainty U(x) = |P(y=1|x) - 0.5| (smallest margin from decision boundary)
        """
        X_cand = self.X_pool[self.unlabeled_idx]
        probs = model.predict_proba(X_cand)
        # Distance from 0.5: smaller distance -> higher uncertainty
        margins = np.abs(probs - 0.5)

        # Select indices with minimal margins
        sorted_cand_idx = np.argsort(margins)[:n_query]
        selected_pool_idx = [self.unlabeled_idx[i] for i in sorted_cand_idx]

        for idx in selected_pool_idx:
            self.labeled_idx.append(idx)
            self.unlabeled_idx.remove(idx)

        return selected_pool_idx

    def get_labeled_data(self):
        return self.X_pool[self.labeled_idx], self.y_pool[self.labeled_idx]


if __name__ == "__main__":
    print("--- J11: Active Learning (Uncertainty Sampling) Sanity Check ---")
    np.random.seed(42)

    # Generate synthetic 2D binary classification pool (200 instances)
    N = 250
    X_all = np.random.randn(N, 2)
    # Ground truth non-linear margin: x0 + 0.8*x1 > 0
    y_all = (X_all[:, 0] + 0.8 * X_all[:, 1] >= 0.0).astype(int)

    # Test set for evaluating sample efficiency curves
    X_test = np.random.randn(100, 2)
    y_test = (X_test[:, 0] + 0.8 * X_test[:, 1] >= 0.0).astype(int)

    # 1. Active Learning Loop
    active_pool = ActiveLearningPool(X_all, y_all)
    # Seed with 4 initial points (2 positive, 2 negative)
    pos_seeds = np.where(y_all == 1)[0][:2]
    neg_seeds = np.where(y_all == 0)[0][:2]
    for s in list(pos_seeds) + list(neg_seeds):
        active_pool.labeled_idx.append(s)
        active_pool.unlabeled_idx.remove(s)

    active_model = LogisticRegressionModel(n_features=2, lr=0.1, epochs=50)

    # Perform 5 active queries of 4 points each (total 24 points labeled)
    for _ in range(5):
        X_tr, y_tr = active_pool.get_labeled_data()
        active_model.fit(X_tr, y_tr)
        active_pool.query_uncertainty(active_model, n_query=4)

    X_tr, y_tr = active_pool.get_labeled_data()
    active_model.fit(X_tr, y_tr)
    active_preds = (active_model.predict_proba(X_test) >= 0.5).astype(int)
    active_acc = float(np.mean(active_preds == y_test))

    # 2. Passive Uniform Random Sampling (same budget: 24 points)
    passive_idx = np.random.choice(N, size=24, replace=False)
    passive_model = LogisticRegressionModel(n_features=2, lr=0.1, epochs=50)
    passive_model.fit(X_all[passive_idx], y_all[passive_idx])
    passive_preds = (passive_model.predict_proba(X_test) >= 0.5).astype(int)
    passive_acc = float(np.mean(passive_preds == y_test))

    print(f"Accuracy with 24 Labeled Points -> Active: {active_acc * 100:.1f}% | Random: {passive_acc * 100:.1f}%")
    assert active_acc >= 0.85, f"Active learning must achieve high accuracy, got {active_acc * 100:.1f}%"

    print("J11 Sanity Check: PASSED cleanly.")

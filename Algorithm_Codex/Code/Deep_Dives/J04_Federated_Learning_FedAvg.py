"""
J04: Federated Learning (Federated Averaging - FedAvg)
Pure NumPy implementation of McMahan et al. (AISTATS 2017).
Simulates decentralized client training on non-IID partitions with
weighted central parameter aggregation.
"""

import numpy as np

class ClientModel:
    """Logistic regression classifier trained locally by an edge client."""
    def __init__(self, n_features: int):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def local_train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, lr: float = 0.05):
        n = X.shape[0]
        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            grad_w = (X.T @ (probs - y)) / n
            grad_b = np.mean(probs - y)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b


class FedAvgCoordinator:
    """Central federated coordinator managing client communication rounds."""
    def __init__(self, n_features: int):
        self.global_weights = np.zeros(n_features)
        self.global_bias = 0.0

    def aggregate(self, client_params: list, client_sample_counts: list):
        """
        Federated Averaging:
        w_{t+1} = sum_{k=1}^K (n_k / N) * w_k
        """
        total_samples = sum(client_sample_counts)
        new_w = np.zeros_like(self.global_weights)
        new_b = 0.0

        for (w_k, b_k), n_k in zip(client_params, client_sample_counts):
            weight_k = n_k / total_samples
            new_w += weight_k * w_k
            new_b += weight_k * b_k

        self.global_weights = new_w
        self.global_bias = new_b

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        logits = X @ self.global_weights + self.global_bias
        preds = (logits >= 0.0).astype(int)
        return float(np.mean(preds == y))


if __name__ == "__main__":
    print("--- J04: Federated Learning (FedAvg) Sanity Check ---")
    np.random.seed(42)

    # 1. Generate non-IID synthetic dataset across 4 edge clients
    n_features = 4
    true_w = np.array([1.5, -1.0, 0.8, -0.5])
    true_b = 0.2

    clients_data = []
    for client_id in range(4):
        # Non-IID feature shift per client
        shift = (client_id - 1.5) * 1.0
        X_client = np.random.randn(80, n_features) + shift
        logits = X_client @ true_w + true_b
        y_client = (logits + np.random.randn(80) * 0.2 >= 0.0).astype(int)
        clients_data.append((X_client, y_client))

    # Global test dataset
    X_test = np.random.randn(200, n_features)
    y_test = (X_test @ true_w + true_b >= 0.0).astype(int)

    coordinator = FedAvgCoordinator(n_features)
    initial_acc = coordinator.evaluate(X_test, y_test)

    # 2. Run 10 communication rounds
    for round_num in range(10):
        client_params = []
        client_sizes = []

        for X_c, y_c in clients_data:
            model_c = ClientModel(n_features)
            # Initialize with central global weights
            model_c.weights = coordinator.global_weights.copy()
            model_c.bias = float(coordinator.global_bias)

            # Local training
            model_c.local_train(X_c, y_c, epochs=5, lr=0.1)

            client_params.append((model_c.weights, model_c.bias))
            client_sizes.append(len(y_c))

        # Central server aggregation
        coordinator.aggregate(client_params, client_sizes)

    final_acc = coordinator.evaluate(X_test, y_test)
    print(f"FedAvg Global Test Accuracy: Initial = {initial_acc * 100:.1f}% -> Final = {final_acc * 100:.1f}%")
    assert final_acc >= 0.85, f"Expected global accuracy >= 85%, got {final_acc * 100:.1f}%"

    print("J04 Sanity Check: PASSED cleanly.")

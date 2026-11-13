"""
J09: Recommender Systems (Collaborative Filtering & Matrix Factorization)
Pure NumPy implementation of user-item collaborative filtering and
regularized Matrix Factorization with latent biases trained via SGD.
"""

import numpy as np

class MatrixFactorizationSGD:
    """
    SVD-style Matrix Factorization with global, user, and item bias terms:
    r_hat_{ui} = mu + b_u + b_i + p_u^T q_i
    """
    def __init__(self, n_factors: int = 8, lr: float = 0.02, reg: float = 0.02, epochs: int = 50):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.mu = 0.0
        self.b_u = None
        self.b_i = None
        self.P = None # User latent matrix (n_users, n_factors)
        self.Q = None # Item latent matrix (n_items, n_factors)

    def fit(self, R: np.ndarray):
        """
        R: sparse rating matrix (n_users, n_items), with 0 representing unobserved ratings.
        """
        n_users, n_items = R.shape
        observed_u, observed_i = np.where(R > 0)
        ratings = R[observed_u, observed_i]

        self.mu = float(np.mean(ratings))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P = np.random.randn(n_users, self.n_factors) * 0.1
        self.Q = np.random.randn(n_items, self.n_factors) * 0.1

        n_ratings = len(ratings)

        for ep in range(self.epochs):
            perm = np.random.permutation(n_ratings)
            for idx in perm:
                u = observed_u[idx]
                i = observed_i[idx]
                r = ratings[idx]

                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred

                # SGD updates with L2 regularization
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

    def predict(self, u: int, i: int) -> float:
        pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_matrix(self) -> np.ndarray:
        return np.clip(self.mu + self.b_u[:, None] + self.b_i[None, :] + (self.P @ self.Q.T), 1.0, 5.0)


if __name__ == "__main__":
    print("--- J09: Recommender Systems (Matrix Factorization) Sanity Check ---")
    np.random.seed(42)

    n_users, n_items = 30, 20
    # Generate low-rank ground truth preference structure
    true_P = np.random.randn(n_users, 3)
    true_Q = np.random.randn(n_items, 3)
    true_matrix = np.clip(3.0 + true_P @ true_Q.T, 1.0, 5.0)

    # Observe only 40% of ratings
    mask = np.random.rand(n_users, n_items) < 0.4
    R = np.zeros((n_users, n_items))
    R[mask] = true_matrix[mask]

    # Held-out test set from remaining 60%
    test_mask = ~mask
    test_actual = true_matrix[test_mask]

    mf = MatrixFactorizationSGD(n_factors=4, lr=0.03, reg=0.01, epochs=60)
    mf.fit(R)

    pred_matrix = mf.predict_matrix()
    test_preds = pred_matrix[test_mask]

    test_rmse = float(np.sqrt(np.mean((test_preds - test_actual) ** 2)))
    print(f"Matrix Factorization Test RMSE on Held-out Ratings: {test_rmse:.4f}")

    assert test_rmse < 1.0, f"Expected Test RMSE < 1.0, got {test_rmse:.4f}"
    print("J09 Sanity Check: PASSED cleanly.")

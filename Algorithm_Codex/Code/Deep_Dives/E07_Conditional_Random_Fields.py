"""
E07. Conditional Random Fields (Linear-Chain CRF)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LinearChainCRFScratch:
    """Linear-Chain Conditional Random Field with Forward-Backward training and Viterbi decoding."""
    def __init__(self, n_states: int = 2, n_features: int = 2, lr: float = 0.1, n_iters: int = 50):
        self.n_states = n_states
        self.n_features = n_features
        self.lr = lr
        self.n_iters = n_iters
        # Weights:
        # 1. State weights: (n_features, n_states)
        self.W_state = np.zeros((n_features, n_states))
        # 2. Transition weights: (n_states, n_states) from state i to state j
        self.W_trans = np.zeros((n_states, n_states))

    def _potentials(self, X: np.ndarray):
        """Computes transition-emission potential matrices M_t for sequence <- eq. 3.2"""
        T = len(X)
        # State potentials: exp(X @ W_state)
        state_scores = X @ self.W_state # Shape (T, n_states)
        
        # M_t[i, j] = exp(W_trans[i, j] + state_scores[t, j])
        M = np.zeros((T, self.n_states, self.n_states))
        for t in range(T):
            M[t] = np.exp(np.clip(self.W_trans + state_scores[t, :], -30, 30))
        return M

    def forward_backward(self, M: np.ndarray):
        """Computes alpha, beta, and partition function Z <- eq. 3.2"""
        T = len(M)
        alpha = np.zeros((T, self.n_states))
        # Base case t = 0
        alpha[0] = np.ones(self.n_states) @ M[0]
        alpha[0] /= np.sum(alpha[0])

        for t in range(1, T):
            alpha[t] = alpha[t - 1] @ M[t]
            alpha[t] /= np.sum(alpha[t])

        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = M[t + 1] @ beta[t + 1]
            beta[t] /= np.sum(beta[t])

        return alpha, beta

    def fit(self, X_seqs: list, y_seqs: list):
        """Trains CRF parameters via gradient ascent on the log-likelihood <- eq. 3.1"""
        for iteration in range(self.n_iters):
            grad_state = np.zeros_like(self.W_state)
            grad_trans = np.zeros_like(self.W_trans)

            for X, y in zip(X_seqs, y_seqs):
                T = len(X)
                M = self._potentials(X)
                alpha, beta = self.forward_backward(M)

                # 1. Empirical feature counts
                for t in range(T):
                    grad_state[:, y[t]] += X[t]
                    if t > 0:
                        grad_trans[y[t - 1], y[t]] += 1.0

                # 2. Expected feature counts under model
                for t in range(T):
                    # Marginal state prob P(y_t = j | X)
                    gamma_t = alpha[t] * beta[t]
                    gamma_t /= np.sum(gamma_t) + 1e-15
                    grad_state -= np.outer(X[t], gamma_t)

                    if t > 0:
                        # Marginal transition prob P(y_{t-1}=i, y_t=j | X)
                        xi_t = alpha[t - 1, :, None] * M[t] * beta[t, None, :]
                        xi_t /= np.sum(xi_t) + 1e-15
                        grad_trans -= xi_t

            # Gradient ascent update
            self.W_state += self.lr * grad_state / len(X_seqs)
            self.W_trans += self.lr * grad_trans / len(X_seqs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for highest scoring label sequence."""
        T = len(X)
        M = self._potentials(X)
        log_M = np.log(np.maximum(1e-15, M))
        
        V = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)

        V[0] = np.log(M[0, 0, :])
        for t in range(1, T):
            for j in range(self.n_states):
                scores = V[t - 1] + log_M[t, :, j]
                best_prev = np.argmax(scores)
                V[t, j] = scores[best_prev]
                backpointer[t, j] = best_prev

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path

if __name__ == "__main__":
    print("--- E07: Linear-Chain CRF Sanity Check ---")
    np.random.seed(42)
    # Sequence labeling toy task: State 0 tends to follow State 0, State 1 follows State 1
    # Feature 0 indicates State 0; Feature 1 indicates State 1
    seq_len = 10
    X1 = np.vstack([np.array([1.0, 0.0]) for _ in range(5)] + [np.array([0.0, 1.0]) for _ in range(5)])
    y1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    crf = LinearChainCRFScratch(n_states=2, n_features=2, lr=0.2, n_iters=40)
    crf.fit([X1, X1], [y1, y1])

    decoded = crf.predict(X1)
    print(f"Decoded Sequence: {decoded}")
    print(f"True Sequence:    {y1}")
    assert np.array_equal(decoded, y1)
    print("E07 Sanity Check: PASSED cleanly.")

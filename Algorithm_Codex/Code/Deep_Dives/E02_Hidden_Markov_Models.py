"""
E02. Hidden Markov Models (Forward-Backward, Viterbi, Baum-Welch)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class HiddenMarkovModelScratch:
    """Discrete Hidden Markov Model built from scratch."""
    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations
        # Initialize uniformly or randomly
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.B = np.full((n_states, n_observations), 1.0 / n_observations)

    def forward(self, obs: np.ndarray):
        """Forward algorithm computing alpha matrix and scaling factors <- eq. 3.1"""
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)

        # Base case t = 0
        alpha[0] = self.pi * self.B[:, obs[0]]
        scale[0] = np.sum(alpha[0]) + 1e-15
        alpha[0] /= scale[0]

        # Induction
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scale[t] = np.sum(alpha[t]) + 1e-15
            alpha[t] /= scale[t]

        log_likelihood = np.sum(np.log(scale))
        return alpha, scale, log_likelihood

    def backward(self, obs: np.ndarray, scale: np.ndarray):
        """Backward algorithm computing beta matrix using forward scaling factors <- eq. 3.1"""
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1.0 / scale[T - 1]

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            beta[t] /= scale[t]

        return beta

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        """Exact max-path decoding in log-probability space <- eq. 3.2"""
        T = len(obs)
        V = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)

        log_pi = np.log(np.maximum(1e-15, self.pi))
        log_A = np.log(np.maximum(1e-15, self.A))
        log_B = np.log(np.maximum(1e-15, self.B))

        # Initialization
        V[0] = log_pi + log_B[:, obs[0]]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                # V_{t-1}(i) + log A_{ij}
                trans_probs = V[t - 1] + log_A[:, j]
                best_prev = np.argmax(trans_probs)
                V[t, j] = trans_probs[best_prev] + log_B[j, obs[t]]
                backpointer[t, j] = best_prev

        # Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(V[T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path

    def fit(self, obs: np.ndarray, n_iters: int = 20, tol: float = 1e-4):
        """Baum-Welch (EM) parameter re-estimation <- eq. 3.3"""
        T = len(obs)
        prev_ll = -np.inf

        for _ in range(n_iters):
            alpha, scale, log_ll = self.forward(obs)
            beta = self.backward(obs, scale)

            if abs(log_ll - prev_ll) < tol:
                break
            prev_ll = log_ll

            # Compute gamma_t(i) and xi_t(i, j)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                numerator = alpha[t, :, None] * self.A * self.B[None, :, obs[t + 1]] * beta[t + 1, None, :]
                xi[t] = numerator / (np.sum(numerator) + 1e-15)

            # M-Step Parameter Updates
            self.pi = gamma[0]
            self.A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0, keepdims=True).T + 1e-15)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            for k in range(self.n_observations):
                mask = (obs == k)
                self.B[:, k] = np.sum(gamma[mask], axis=0) / (np.sum(gamma, axis=0) + 1e-15)
            self.B /= np.sum(self.B, axis=1, keepdims=True)

        return self

if __name__ == "__main__":
    print("--- E02: Hidden Markov Models Sanity Check ---")
    np.random.seed(42)
    # Define an HMM: 2 states (Sunny=0, Rainy=1), 2 emissions (Clean=0, Muddy=1)
    hmm = HiddenMarkovModelScratch(n_states=2, n_observations=2)
    hmm.pi = np.array([0.6, 0.4])
    hmm.A = np.array([[0.8, 0.2],
                      [0.3, 0.7]])
    hmm.B = np.array([[0.9, 0.1],  # Sunny -> 90% Clean boots
                      [0.2, 0.8]])  # Rainy -> 80% Muddy boots

    # Observed muddy/clean sequence: Muddy, Muddy, Clean, Clean, Muddy
    observations = np.array([1, 1, 0, 0, 1])
    
    # 1. Test Forward-Backward
    alpha, scale, log_ll = hmm.forward(observations)
    print(f"HMM Sequence Log-Likelihood: {log_ll:.4f}")
    assert np.isfinite(log_ll)
    
    # 2. Test Viterbi Decoding
    best_path = hmm.viterbi(observations)
    print(f"Decoded Hidden State Sequence: {best_path} (0=Sunny, 1=Rainy)")
    # Days 0,1 are Muddy -> likely Rainy(1); Days 2,3 are Clean -> likely Sunny(0)
    assert best_path[0] == 1 and best_path[1] == 1
    assert best_path[2] == 0 and best_path[3] == 0
    print("E02 Sanity Check: PASSED cleanly.")

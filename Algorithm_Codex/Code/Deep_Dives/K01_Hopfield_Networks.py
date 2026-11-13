"""
K01: Hopfield Networks & Associative Memory
Pure NumPy implementation of Hebbian pattern storage and asynchronous retrieval.
"""

import numpy as np

class HopfieldNetwork:
    """Discrete Hopfield Network with bipolar states in {-1, +1}."""
    def __init__(self, n_neurons: int):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons), dtype=np.float64)

    def store_patterns(self, patterns: np.ndarray):
        """
        Hebbian outer product learning rule:
        W = 1/N sum_mu xi_mu xi_mu^T with zero diagonal.
        """
        P, N = patterns.shape
        assert N == self.n
        self.W = np.zeros((self.n, self.n))
        for mu in range(P):
            self.W += np.outer(patterns[mu], patterns[mu])
        self.W /= self.n
        np.fill_diagonal(self.W, 0.0)

    def energy(self, s: np.ndarray) -> float:
        """Computes Lyapunov energy E = -0.5 * s^T W s."""
        return float(-0.5 * s @ self.W @ s)

    def retrieve(self, corrupted_state: np.ndarray, max_iter: int = 50) -> np.ndarray:
        s = np.array(corrupted_state, dtype=np.float64)
        for _ in range(max_iter):
            old_s = s.copy()
            # Asynchronous or synchronous update
            h = self.W @ s
            s = np.where(h >= 0, 1.0, -1.0)
            if np.array_equal(s, old_s):
                break
        return s


if __name__ == "__main__":
    print("--- K01: Hopfield Network Sanity Check ---")
    # Store two distinct 10-neuron bipolar patterns
    p1 = np.array([ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1])
    p2 = np.array([ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1])
    patterns = np.vstack([p1, p2])

    net = HopfieldNetwork(n_neurons=10)
    net.store_patterns(patterns)

    # Corrupt p1 by flipping 2 bits
    corrupted_p1 = p1.copy()
    corrupted_p1[0] = -corrupted_p1[0]
    corrupted_p1[1] = -corrupted_p1[1]

    retrieved = net.retrieve(corrupted_p1)

    print(f"Original Pattern:  {p1}")
    print(f"Corrupted Query:   {corrupted_p1}")
    print(f"Retrieved Pattern: {retrieved.astype(int)}")

    assert np.array_equal(retrieved, p1), "Hopfield network must cleanly restore corrupted bits back to stored attractor"
    print("K01 Sanity Check: PASSED cleanly.")

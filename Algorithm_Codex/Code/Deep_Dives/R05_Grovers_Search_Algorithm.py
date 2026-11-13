"""
R05: Grover's Search Algorithm
Pure NumPy implementation of Grover's quantum search algorithm for unstructured databases.
Demonstrates oracle marking, diffusion operator (inversion about mean), and quadratic speedup.
"""

import numpy as np

class GroverSearch:
    """
    Simulator for Grover's Unstructured Search Algorithm on n qubits (N = 2^n states).
    """
    def __init__(self, n_qubits: int, target_state: int):
        self.n_qubits = n_qubits
        self.N = 1 << n_qubits
        self.target = target_state
        assert 0 <= target_state < self.N, "Target state index out of bounds"
        
        # Optimal number of Grover iterations R approx (pi / 4) * sqrt(N)
        self.optimal_iterations = int(np.round((np.pi / 4.0) * np.sqrt(self.N)))

    def initialize_state(self) -> np.ndarray:
        """Create uniform superposition state |s> = (1/sqrt(N)) sum_x |x>."""
        return np.ones(self.N, dtype=complex) / np.sqrt(self.N)

    def apply_oracle(self, state: np.ndarray) -> np.ndarray:
        """
        Oracle reflection O_w = I - 2|w><w|:
        Flips the phase of the target state |w>.
        """
        state_copy = state.copy()
        state_copy[self.target] *= -1.0
        return state_copy

    def apply_diffusion(self, state: np.ndarray) -> np.ndarray:
        """
        Diffusion operator (inversion about mean) D = 2|s><s| - I:
        Maps each amplitude a_i to 2*mean(a) - a_i.
        """
        mean_amp = np.mean(state)
        return 2.0 * mean_amp - state

    def run_search(self, iterations: int | None = None) -> tuple[np.ndarray, list[float]]:
        """
        Execute Grover iterations G = D * O.
        Returns final state vector and history of target state probabilities.
        """
        if iterations is None:
            iterations = self.optimal_iterations
            
        state = self.initialize_state()
        prob_history = [float(np.abs(state[self.target]) ** 2)]
        
        for _ in range(iterations):
            state = self.apply_oracle(state)
            state = self.apply_diffusion(state)
            prob_history.append(float(np.abs(state[self.target]) ** 2))
            
        return state, prob_history

if __name__ == "__main__":
    print("--- R05: Grover's Search Algorithm Sanity Check ---")
    
    # 1. 3-qubit search (N = 8 items)
    # Target = 5 (|101>) -> R = round((pi/4)*sqrt(8)) = 2 iterations
    grover3 = GroverSearch(n_qubits=3, target_state=5)
    state3, probs3 = grover3.run_search()
    final_prob3 = probs3[-1]
    
    print(f"3-Qubit Search (N=8): Optimal Iterations = {grover3.optimal_iterations}")
    print(f"Probability of Target (|101>) after {grover3.optimal_iterations} steps: {final_prob3:.4f}")
    assert final_prob3 > 0.94, "Grover 3-qubit search must achieve > 94% success probability"
    
    # 2. 4-qubit search (N = 16 items)
    # Target = 11 (|1011>) -> R = round((pi/4)*sqrt(16)) = 3 iterations
    grover4 = GroverSearch(n_qubits=4, target_state=11)
    state4, probs4 = grover4.run_search()
    final_prob4 = probs4[-1]
    
    print(f"4-Qubit Search (N=16): Optimal Iterations = {grover4.optimal_iterations}")
    print(f"Probability of Target (|1011>) after {grover4.optimal_iterations} steps: {final_prob4:.4f}")
    assert final_prob4 > 0.95, "Grover 4-qubit search must achieve > 95% success probability"
    
    # Verify unitary normalization throughout evolution
    assert np.isclose(np.sum(np.abs(state4) ** 2), 1.0), "State normalization must be preserved"
    
    print("Grover's Search verifications PASSED cleanly.")

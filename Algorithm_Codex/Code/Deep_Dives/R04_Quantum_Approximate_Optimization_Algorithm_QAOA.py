"""
R04: Quantum Approximate Optimization Algorithm (QAOA)
Variational quantum algorithm for combinatorial optimization.
Solves MaxCut on graphs using alternating cost and mixer Hamiltonians.
"""

import numpy as np
from R01_Qubit_State_Vectors_and_Quantum_Gates import H

class QAOA:
    """
    Quantum Approximate Optimization Algorithm for MaxCut on unweighted graphs.
    """
    def __init__(self, n_nodes: int, edges: list[tuple[int, int]], p_layers: int = 1):
        self.n_nodes = n_nodes
        self.edges = edges
        self.p_layers = p_layers
        self.dim = 1 << n_nodes
        
        # Precompute cost function for each computational basis state
        # Cost(x) = sum_{(u, v) in E} (x_u ^ x_v) = number of cut edges
        self.cut_values = np.zeros(self.dim, dtype=float)
        for state_idx in range(self.dim):
            cut = 0
            for u, v in edges:
                bit_u = (state_idx >> u) & 1
                bit_v = (state_idx >> v) & 1
                if bit_u != bit_v:
                    cut += 1
            self.cut_values[state_idx] = cut

    def state_preparation(self, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Prepare QAOA state:
        |psi(gamma, beta)> = U(B, beta_p) U(C, gamma_p) ... U(B, beta_1) U(C, gamma_1) |+>^n
        """
        # Initial uniform superposition |+>^n
        state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        
        for layer in range(self.p_layers):
            g = gamma[layer]
            b = beta[layer]
            
            # 1. Cost unitary U(C, gamma): diagonal phase shift e^(-i * gamma * Cut(x))
            state = state * np.exp(-1.0j * g * self.cut_values)
            
            # 2. Mixer unitary U(B, beta) = prod_k e^(-i * beta * X_k)
            # Tensor reshape and apply Rx(2*beta) to each qubit
            shape = [2] * self.n_nodes
            tensor = state.reshape(shape)
            
            # e^(-i * beta * X) = cos(beta)*I - i*sin(beta)*X
            c = np.cos(b)
            s = -1.0j * np.sin(b)
            rx_gate = np.array([[c, s], [s, c]], dtype=complex)
            
            for q in range(self.n_nodes):
                tensor = np.tensordot(rx_gate, tensor, axes=(1, q))
                tensor = np.moveaxis(tensor, 0, q)
                
            state = tensor.reshape(self.dim)
            
        return state

    def expected_cut(self, gamma: np.ndarray, beta: np.ndarray) -> float:
        """Compute expectation value <psi| C |psi> = sum_x |psi(x)|^2 * Cut(x)."""
        psi = self.state_preparation(gamma, beta)
        probs = np.abs(psi) ** 2
        return float(np.sum(probs * self.cut_values))

    def optimize_parameters(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Grid/coordinate search for optimal (gamma, beta) at p=1.
        """
        best_exp = -1.0
        best_gamma = np.zeros(self.p_layers)
        best_beta = np.zeros(self.p_layers)
        
        # Grid search over [0, 2*pi] x [0, pi]
        gamma_range = np.linspace(0, 2 * np.pi, 25)
        beta_range = np.linspace(0, np.pi, 25)
        
        for g in gamma_range:
            for b in beta_range:
                val = self.expected_cut(np.array([g]), np.array([b]))
                if val > best_exp:
                    best_exp = val
                    best_gamma[0] = g
                    best_beta[0] = b
                    
        return best_gamma, best_beta, best_exp

if __name__ == "__main__":
    print("--- R04: QAOA MaxCut Sanity Check ---")
    
    # 4-node ring graph (C4): Edges (0,1), (1,2), (2,3), (3,0)
    # Maximum cut is 4 (bipartition: 0,2 vs 1,3 -> bitstring 0101 or 1010)
    edges_ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    qaoa = QAOA(n_nodes=4, edges=edges_ring, p_layers=1)
    
    best_g, best_b, max_expected_cut = qaoa.optimize_parameters()
    
    # Maximum possible cut in C4 is 4
    max_cut = 4.0
    approx_ratio = max_expected_cut / max_cut
    
    print(f"Optimal Gamma : {best_g[0]:.4f}, Beta: {best_b[0]:.4f}")
    print(f"Expected Cut  : {max_expected_cut:.4f} / {max_cut:.1f}")
    print(f"Approx Ratio  : {approx_ratio:.4f}")
    
    # For p=1 QAOA on C4, theoretical approximation ratio is >= 0.75
    assert approx_ratio >= 0.75, "QAOA p=1 must achieve approximation ratio >= 0.75 on C4"
    
    # Sample state and verify probability peaks on optimal solutions (0101=5 and 1010=10)
    opt_state = qaoa.state_preparation(best_g, best_b)
    probs = np.abs(opt_state) ** 2
    peak_prob = probs[5] + probs[10]
    print(f"Probability of exact MaxCut bitstrings (0101 & 1010): {peak_prob:.4f}")
    assert peak_prob > 0.40, "Optimal cut states must dominate probability distribution"
    
    print("QAOA MaxCut verification PASSED cleanly.")

"""
R03: Variational Quantum Eigensolver (VQE)
NISQ hybrid quantum-classical algorithm to compute molecular ground state energies.
Calculates H2 ground state using Jordan-Wigner Hamiltonian and parameterized ansatz.
"""

import numpy as np
from R01_Qubit_State_Vectors_and_Quantum_Gates import I2, X, Y, Z, ry, CNOT, tensor_product

class VQE:
    """
    Variational Quantum Eigensolver for 2-qubit molecular Hamiltonians.
    """
    def __init__(self, hamiltonian_terms: list[tuple[float, list[np.ndarray]]]):
        """
        hamiltonian_terms: list of (coefficient, [gate_q0, gate_q1]) tuples.
        H = sum_i c_i (P_{i,0} (x) P_{i,1})
        """
        self.hamiltonian_terms = hamiltonian_terms
        # Construct explicit Hamiltonian matrix for expectation & exact baseline
        self.H_matrix = sum(c * tensor_product(*ops) for c, ops in hamiltonian_terms)

    def ansatz(self, params: np.ndarray) -> np.ndarray:
        """
        Hardware-efficient ansatz:
        |00> -> Ry(theta_0) (x) Ry(theta_1) -> CNOT -> Ry(theta_2) (x) Ry(theta_3)
        """
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        
        # Layer 1
        u1 = tensor_product(ry(params[0]), ry(params[1]))
        state = u1 @ state
        
        # Entangler
        state = CNOT @ state
        
        # Layer 2
        u2 = tensor_product(ry(params[2]), ry(params[3]))
        state = u2 @ state
        
        return state

    def energy_expectation(self, params: np.ndarray) -> float:
        """
        Compute <psi(theta)|H|psi(theta)> = sum_i c_i <psi(theta)|P_i|psi(theta)>.
        """
        psi = self.ansatz(params)
        # Directly compute expectation: <psi|H|psi>
        energy = np.real(psi.conj().T @ self.H_matrix @ psi)
        return float(energy)

    def optimize(self, initial_params: np.ndarray, lr: float = 0.05, max_iter: int = 150) -> tuple[np.ndarray, float]:
        """
        Classical parameter optimization using finite-difference gradient descent.
        """
        params = np.array(initial_params, dtype=float).copy()
        eps = 1e-5
        
        for _ in range(max_iter):
            grad = np.zeros_like(params)
            base_e = self.energy_expectation(params)
            for i in range(len(params)):
                params_shifted = params.copy()
                params_shifted[i] += eps
                e_shifted = self.energy_expectation(params_shifted)
                grad[i] = (e_shifted - base_e) / eps
                
            params -= lr * grad
            
        final_energy = self.energy_expectation(params)
        return params, final_energy

def get_h2_hamiltonian() -> list[tuple[float, list[np.ndarray]]]:
    """
    Standard Jordan-Wigner mapped STO-3G H2 Hamiltonian at R = 0.7414 Angstrom (2 qubits):
    H = g0*I + g1*Z0 + g2*Z1 + g3*(Z0 (x) Z1) + g4*(X0 (x) X1) + g5*(Y0 (x) Y1)
    """
    g0 = -0.3210
    g1 = 0.1712
    g2 = 0.1712
    g3 = 0.1205
    g4 = 0.1658
    g5 = 0.1658
    
    return [
        (g0, [I2, I2]),
        (g1, [Z, I2]),
        (g2, [I2, Z]),
        (g3, [Z, Z]),
        (g4, [X, X]),
        (g5, [Y, Y])
    ]

if __name__ == "__main__":
    print("--- R03: Variational Quantum Eigensolver (VQE) Sanity Check ---")
    
    h2_terms = get_h2_hamiltonian()
    vqe = VQE(h2_terms)
    
    # Exact ground state via diagonalization
    exact_eigenvalues, _ = np.linalg.eigh(vqe.H_matrix)
    exact_ground_energy = float(exact_eigenvalues[0])
    
    # Run VQE optimization
    np.random.seed(42)
    init_theta = np.random.uniform(0, 2 * np.pi, 4)
    opt_params, vqe_energy = vqe.optimize(init_theta, lr=0.1, max_iter=200)
    
    print(f"Exact Ground State Energy : {exact_ground_energy:.6f} Hartree")
    print(f"VQE Optimized Energy       : {vqe_energy:.6f} Hartree")
    print(f"Absolute Energy Error      : {abs(vqe_energy - exact_ground_energy):.6e} Hartree")
    
    # Assert chemical precision (error < 1.6e-3 Hartree ~ 1 kcal/mol)
    assert abs(vqe_energy - exact_ground_energy) < 1.6e-3, "VQE must reach chemical accuracy"
    print("VQE verification PASSED cleanly.")

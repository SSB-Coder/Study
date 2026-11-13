"""
R08: Quantum Error Correction (3-Qubit Bit-Flip Code)
Pure NumPy implementation of the 3-qubit repetition code protecting against bit-flip (X) errors.
Demonstrates non-destructive syndrome measurements, Pauli error correction, and fidelity gain.
"""

import numpy as np
from R01_Qubit_State_Vectors_and_Quantum_Gates import I2, X, Z, CNOT, tensor_product

class BitFlipCode:
    """
    3-qubit repetition code protecting one logical qubit against Pauli X noise.
    Logical basis: |0_L> = |000>, |1_L> = |111>.
    """
    def __init__(self):
        # Stabilizer generators: S1 = Z (x) Z (x) I, S2 = I (x) Z (x) Z
        self.S1 = tensor_product(Z, Z, I2)
        self.S2 = tensor_product(I2, Z, Z)

    def encode(self, alpha: complex, beta: complex) -> np.ndarray:
        """
        Encode logical state alpha|0> + beta|1> into 3-qubit entangled state:
        |psi_L> = alpha|000> + beta|111>.
        """
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm
        
        state = np.zeros(8, dtype=complex)
        state[0] = alpha  # |000>
        state[7] = beta   # |111>
        return state

    def inject_noise(self, state: np.ndarray, error_qubit: int | None) -> np.ndarray:
        """
        Apply Pauli X bit-flip to the specified qubit (0, 1, or 2).
        If None, no error is injected.
        """
        if error_qubit is None:
            return state.copy()
            
        ops = [I2, I2, I2]
        ops[error_qubit] = X
        error_gate = tensor_product(*ops)
        return error_gate @ state

    def measure_syndrome(self, state: np.ndarray) -> tuple[int, int]:
        """
        Measure eigenvalues of stabilizers S1 and S2:
        Eigenvalue +1 -> syndrome bit 0; Eigenvalue -1 -> syndrome bit 1.
        """
        # Since the error commutes with stabilizers up to sign, the noisy state
        # is an eigenstate of S1 and S2 with eigenvalue +/- 1
        s1_exp = np.real(state.conj().T @ self.S1 @ state)
        s2_exp = np.real(state.conj().T @ self.S2 @ state)
        
        bit1 = 0 if s1_exp > 0 else 1
        bit2 = 0 if s2_exp > 0 else 1
        return bit1, bit2

    def correct_error(self, state: np.ndarray, syndrome: tuple[int, int]) -> np.ndarray:
        """
        Apply recovery operation based on error syndrome:
        (0, 0) -> No error
        (1, 0) -> Error on Qubit 0 (S1=-1, S2=+1)
        (1, 1) -> Error on Qubit 1 (S1=-1, S2=-1)
        (0, 1) -> Error on Qubit 2 (S1=+1, S2=-1)
        """
        syndrome_map = {
            (0, 0): None,
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2
        }
        faulty_qubit = syndrome_map[syndrome]
        if faulty_qubit is None:
            return state.copy()
            
        ops = [I2, I2, I2]
        ops[faulty_qubit] = X
        correction_gate = tensor_product(*ops)
        return correction_gate @ state

    def decode(self, state: np.ndarray) -> tuple[complex, complex]:
        """Extract alpha and beta amplitudes from logical codeword."""
        alpha = state[0]
        beta = state[7]
        return alpha, beta

if __name__ == "__main__":
    print("--- R08: Quantum Error Correction (3-Qubit Code) Sanity Check ---")
    
    qec = BitFlipCode()
    alpha_init = 0.8 + 0.1j
    beta_init = 0.5 - 0.3j
    norm_init = np.sqrt(abs(alpha_init)**2 + abs(beta_init)**2)
    alpha_init /= norm_init
    beta_init /= norm_init
    
    # Encode state
    encoded_state = qec.encode(alpha_init, beta_init)
    
    # Test all single-qubit error cases: None, Qubit 0, Qubit 1, Qubit 2
    for error_q in [None, 0, 1, 2]:
        noisy_state = qec.inject_noise(encoded_state, error_q)
        syndrome = qec.measure_syndrome(noisy_state)
        corrected_state = qec.correct_error(noisy_state, syndrome)
        
        # State fidelity
        fidelity = abs(np.vdot(encoded_state, corrected_state)) ** 2
        print(f"Injected error on Qubit {error_q} -> Syndrome {syndrome} -> Fidelity: {fidelity:.6f}")
        assert np.isclose(fidelity, 1.0), f"Recovery failed for error on qubit {error_q}"
        
    print("All 3-Qubit Bit-Flip QEC syndrome and recovery tests PASSED cleanly.")

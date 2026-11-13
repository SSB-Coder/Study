"""
R02: Quantum Circuit Simulator
Full n-qubit state vector quantum circuit simulator supporting arbitrary single/two-qubit gates,
projective measurements, state collapse, and quantum teleportation.
"""

import numpy as np
from R01_Qubit_State_Vectors_and_Quantum_Gates import I2, X, Z, H, CNOT, tensor_product

class QuantumCircuit:
    """
    Pure NumPy state-vector simulator for n qubits.
    State is stored as a 1D complex vector of length 2^n.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        # Initialize to ground state |0...0>
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0 + 0.0j

    def apply_single_gate(self, gate: np.ndarray, target: int) -> None:
        """Apply 2x2 gate to the specified target qubit (0 to n-1)."""
        # Reshape state to n-dimensional tensor (2, 2, ..., 2)
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        # Contract along target axis
        # tensor has axes 0, 1, ..., n-1. We contract gate (out, in) with tensor along axis target
        tensor = np.tensordot(gate, tensor, axes=(1, target))
        # tensordot puts the gate's new index at axis 0. Move it back to target
        tensor = np.moveaxis(tensor, 0, target)
        self.state = tensor.reshape(self.dim)

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply Controlled-NOT with control and target qubit indices."""
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        
        # We perform conditional Pauli-X on target when control == 1
        # Slicing tensor where control axis == 1
        idx_ctrl1 = [slice(None)] * self.n_qubits
        idx_ctrl1[control] = 1
        
        sub_tensor = tensor[tuple(idx_ctrl1)]
        # Apply X to target in sub_tensor
        # Target index in sub_tensor shifts if control < target
        sub_target = target if target < control else target - 1
        sub_tensor = np.tensordot(X, sub_tensor, axes=(1, sub_target))
        sub_tensor = np.moveaxis(sub_tensor, 0, sub_target)
        
        tensor[tuple(idx_ctrl1)] = sub_tensor
        self.state = tensor.reshape(self.dim)

    def get_probabilities(self) -> np.ndarray:
        """Compute Born rule probabilities P(i) = |<i|psi>|^2."""
        return np.abs(self.state) ** 2

    def measure(self, target: int, seed: int | None = None) -> int:
        """
        Perform projective measurement on target qubit.
        Returns outcome 0 or 1, and collapses the state vector.
        """
        rng = np.random.default_rng(seed)
        shape = [2] * self.n_qubits
        tensor = self.state.reshape(shape)
        
        # Marginal probability of target being 0
        idx_0 = [slice(None)] * self.n_qubits
        idx_0[target] = 0
        p0 = float(np.sum(np.abs(tensor[tuple(idx_0)]) ** 2))
        
        outcome = 0 if rng.random() < p0 else 1
        
        # State collapse: zero out non-matching outcomes and renormalize
        idx_kill = [slice(None)] * self.n_qubits
        idx_kill[target] = 1 - outcome
        tensor[tuple(idx_kill)] = 0.0
        
        norm = np.linalg.norm(tensor)
        if norm > 0:
            tensor /= norm
        self.state = tensor.reshape(self.dim)
        return outcome

def simulate_teleportation(alpha: complex, beta: complex) -> bool:
    """
    Simulate full quantum teleportation protocol of state |psi> = alpha|0> + beta|1>.
    Qubit 0: Alice's unknown input state |psi>
    Qubit 1: Alice's half of Bell pair
    Qubit 2: Bob's half of Bell pair
    """
    # Normalize input
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha /= norm
    beta /= norm
    
    qc = QuantumCircuit(3)
    # Prepare Qubit 0 in state |psi>
    qc.state[0] = alpha
    qc.state[4] = beta  # Qubit 0 is the most significant bit in binary 100 = 4 if MSB=q0
    # Let's verify indexing: in [2,2,2], index (q0, q1, q2) -> q0*4 + q1*2 + q2
    # So q0=1, q1=0, q2=0 is index 4.
    
    # Step 1: Create Bell state between Qubit 1 and Qubit 2
    qc.apply_single_gate(H, 1)
    qc.apply_cnot(1, 2)
    
    # Step 2: Alice applies CNOT(q0 -> q1) and H(q0)
    qc.apply_cnot(0, 1)
    qc.apply_single_gate(H, 0)
    
    # Step 3: Alice measures Qubit 0 and Qubit 1
    m0 = qc.measure(0)
    m1 = qc.measure(1)
    
    # Step 4: Bob applies Pauli corrections based on measurements
    # If m1 == 1, Bob applies X to Qubit 2
    if m1 == 1:
        qc.apply_single_gate(X, 2)
    # If m0 == 1, Bob applies Z to Qubit 2
    if m0 == 1:
        qc.apply_single_gate(Z, 2)
        
    # Bob's qubit is at qubit 2. After collapse, the joint state is |m0> |m1> |psi_bob>
    # Slicing state at q0=m0, q1=m1 gives Bob's state
    shape = [2, 2, 2]
    tensor = qc.state.reshape(shape)
    bob_state = tensor[m0, m1, :]
    
    # Compute fidelity with original state
    target_state = np.array([alpha, beta], dtype=complex)
    fidelity = abs(np.vdot(target_state, bob_state)) ** 2
    return bool(np.isclose(fidelity, 1.0, atol=1e-7))

if __name__ == "__main__":
    print("--- R02: Quantum Circuit Simulator Sanity Check ---")
    
    # 1. GHZ State preparation: (|000> + |111>) / sqrt(2)
    qc_ghz = QuantumCircuit(3)
    qc_ghz.apply_single_gate(H, 0)
    qc_ghz.apply_cnot(0, 1)
    qc_ghz.apply_cnot(1, 2)
    
    probs = qc_ghz.get_probabilities()
    assert np.isclose(probs[0], 0.5) and np.isclose(probs[7], 0.5), "GHZ state generation failed"
    assert np.isclose(np.sum(probs), 1.0), "Probabilities must sum to 1"
    
    # 2. Quantum Teleportation Protocol
    # Teleport arbitrary superposition
    alpha_in = 0.6 + 0.3j
    beta_in = 0.8 - 0.2j
    success = simulate_teleportation(alpha_in, beta_in)
    assert success, "Quantum teleportation fidelity must be 1.0"
    
    print("GHZ state and Quantum Teleportation verifications PASSED cleanly.")

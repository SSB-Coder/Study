"""
R01: Qubit State Vectors and Quantum Gates
Pure NumPy implementation of single- and two-qubit quantum states,
Bloch sphere representations, unitary quantum gates, and multi-qubit tensor products.
"""

import numpy as np

# Elementary Single-Qubit Gates (2x2 complex matrices)
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
T = np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=complex)

def rx(theta: float) -> np.ndarray:
    """Rotation around X-axis: Rx(theta) = cos(theta/2)*I - i*sin(theta/2)*X."""
    return np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * X

def ry(theta: float) -> np.ndarray:
    """Rotation around Y-axis: Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y."""
    return np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * Y

def rz(theta: float) -> np.ndarray:
    """Rotation around Z-axis: Rz(theta) = cos(theta/2)*I - i*sin(theta/2)*Z."""
    return np.cos(theta / 2.0) * I2 - 1.0j * np.sin(theta / 2.0) * Z

# Two-Qubit and Multi-Qubit Gates
CNOT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=complex)

SWAP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=complex)

def is_unitary(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if matrix U satisfies U^dagger U = I."""
    identity = np.eye(matrix.shape[0], dtype=complex)
    product = matrix.conj().T @ matrix
    return bool(np.allclose(product, identity, atol=tol))

def bloch_coordinates(state: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Bloch sphere coordinates (x, y, z) for a 2-element normalized pure state.
    x = <psi|X|psi>, y = <psi|Y|psi>, z = <psi|Z|psi>.
    """
    psi = np.asarray(state, dtype=complex).reshape(2, 1)
    norm = np.linalg.norm(psi)
    psi = psi / norm
    
    x = float(np.real(psi.conj().T @ X @ psi)[0, 0])
    y = float(np.real(psi.conj().T @ Y @ psi)[0, 0])
    z = float(np.real(psi.conj().T @ Z @ psi)[0, 0])
    return x, y, z

def state_from_bloch(theta: float, phi: float) -> np.ndarray:
    """Generate pure state vector |psi> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>."""
    alpha = np.cos(theta / 2.0)
    beta = np.exp(1.0j * phi) * np.sin(theta / 2.0)
    return np.array([alpha, beta], dtype=complex)

def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """Compute Kronecker tensor product of an arbitrary list of matrices/vectors."""
    res = matrices[0]
    for m in matrices[1:]:
        res = np.kron(res, m)
    return res

if __name__ == "__main__":
    print("--- R01: Qubit State Vectors & Quantum Gates Sanity Check ---")
    
    # 1. Unitarity of standard gates
    for name, gate in [("H", H), ("X", X), ("Y", Y), ("Z", Z), ("S", S), ("T", T), ("CNOT", CNOT), ("SWAP", SWAP)]:
        assert is_unitary(gate), f"Gate {name} must be unitary"
    
    # 2. Parametric rotations unitarity
    theta = np.pi / 3.7
    assert is_unitary(rx(theta)) and is_unitary(ry(theta)) and is_unitary(rz(theta))
    
    # 3. Bloch sphere mapping
    # State |0>: theta=0 -> (0, 0, 1)
    state_0 = np.array([1.0, 0.0], dtype=complex)
    bx, by, bz = bloch_coordinates(state_0)
    assert np.isclose(bx, 0.0) and np.isclose(by, 0.0) and np.isclose(bz, 1.0)
    
    # State |+> = H|0>: theta=pi/2, phi=0 -> (1, 0, 0)
    state_plus = H @ state_0
    bx, by, bz = bloch_coordinates(state_plus)
    assert np.isclose(bx, 1.0) and np.isclose(by, 0.0) and np.isclose(bz, 0.0)
    
    # State |R> = (|0> + i|1>)/sqrt(2): theta=pi/2, phi=pi/2 -> (0, 1, 0)
    state_r = state_from_bloch(np.pi / 2.0, np.pi / 2.0)
    bx, by, bz = bloch_coordinates(state_r)
    assert np.isclose(bx, 0.0) and np.isclose(by, 1.0) and np.isclose(bz, 0.0)
    
    # 4. Entanglement: Bell state |Phi+> = CNOT (H (x) I) |00>
    state_00 = np.array([1, 0, 0, 0], dtype=complex)
    h_kron_i = tensor_product(H, I2)
    bell_state = CNOT @ (h_kron_i @ state_00)
    expected_bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=complex)
    assert np.allclose(bell_state, expected_bell), "Bell state generation failed"
    
    print("All unitary, Bloch sphere, and Bell gate tests PASSED.")

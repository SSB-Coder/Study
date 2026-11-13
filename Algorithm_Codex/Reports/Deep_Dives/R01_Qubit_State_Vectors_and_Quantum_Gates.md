# R01: Qubit State Vectors and Quantum Gates

## 1. Mathematical Formulation & Architecture
A quantum state vector of an isolated $n$-qubit system resides in a $2^n$-dimensional complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$:
$$\lvert\psi\rangle = \sum_{x \in \{0,1\}^n} c_x \lvert x\rangle, \quad \sum_{x} \lvert c_x\rvert^2 = 1$$

For a single qubit ($n=1$), any pure state is represented on the **Bloch Sphere**:
$$\lvert\psi\rangle = \cos\left(\frac{\theta}{2}\right)\lvert 0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)\lvert 1\rangle, \quad \theta \in [0, \pi], \; \phi \in [0, 2\pi)$$
The Bloch coordinates $(r_x, r_y, r_z) \in S^2$ correspond to the Pauli expectation values:
$$r_x = \langle\psi\lvert X \rvert\psi\rangle = \sin\theta\cos\phi, \quad r_y = \langle\psi\lvert Y \rvert\psi\rangle = \sin\theta\sin\phi, \quad r_z = \langle\psi\lvert Z \rvert\psi\rangle = \cos\theta$$

### Quantum Gates and Unitarity
Reversible evolution in closed quantum systems is governed by unitary operators $U \in U(2^n)$, where:
$$U^\dagger U = U U^\dagger = I$$
The elementary single-qubit generators are the Pauli matrices:
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$
Arbitrary single-qubit rotations are parameterized by axis-angle exponentiation:
$$R_{\hat{n}}(\theta) = \exp\left(-i \frac{\theta}{2} \hat{n}\cdot \vec{\sigma}\right) = \cos\left(\frac{\theta}{2}\right) I - i \sin\left(\frac{\theta}{2}\right) (n_x X + n_y Y + n_z Z)$$

Multi-qubit operators are constructed via the Kronecker tensor product:
$$(A \otimes B)_{i k, j l} = A_{i, j} B_{k, l}$$

## 2. Implementation Details
The Python module `R01_Qubit_State_Vectors_and_Quantum_Gates.py` provides:
- Exact representations for $I, X, Y, Z, H, S, T$, and parametric rotations $R_x, R_y, R_z(\theta)$.
- Two-qubit gates including CNOT and SWAP.
- Numerical unitarity validator `is_unitary` with tolerance check $\lVert U^\dagger U - I \rVert_\infty < 10^{-9}$.
- `bloch_coordinates` and `state_from_bloch` for spherical coordinate conversions.
- Arbitrary tensor products via `tensor_product`.

## 3. Asymptotic & Numerical Analysis
- **Gate-State Multiplication**: For an $n$-qubit system, naive state vector multiplication requires $\mathcal{O}(2^{2n})$ operations using full matrices, or $\mathcal{O}(2^n)$ when applying $2\times 2$ single-qubit operations directly to the tensor-reshaped state vector.
- **Precision Limits**: Complex 128-bit floating-point precision yields roundoff errors on the order of $\epsilon_{\text{mach}} \sim 10^{-16}$, well within our $10^{-9}$ assertion threshold.

## 4. Edge Cases & Numerical Stability
- **Global Phase Equivalence**: Two states $\lvert\psi_1\rangle$ and $e^{i\gamma}\lvert\psi_1\rangle$ correspond to the exact same physical state. Bloch sphere projections eliminate global phase ambiguities by mapping to gauge-invariant density matrix expectations $\operatorname{Tr}(\rho \sigma_i)$.
- **Pole Singularities**: At the poles $\theta = 0$ ($|0\rangle$) and $\theta = \pi$ ($|1\rangle$), the azimuthal angle $\phi$ is degenerate; normalization handles this stably without branch cuts.

## 5. Standalone Verification
The test suite in `R01_Qubit_State_Vectors_and_Quantum_Gates.py` verifies:
- Unitarity of all standard gates and parametric rotations.
- Coordinate accuracy for $|0\rangle, |1\rangle, |+\rangle$, and $|R\rangle$.
- Exact analytical fidelity of Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$.

## 6. Real-World Applications & Context
Foundational primitive for all quantum algorithms, NISQ circuit compilers, quantum chemistry (VQE), and quantum hardware control.

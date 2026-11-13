# R03: Variational Quantum Eigensolver (VQE)

## 1. Mathematical Formulation & Architecture
The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the minimum eigenvalue (ground state energy) of a Hamiltonian $H$:
$$E_0 = \min_{\lvert\psi\rangle} \frac{\langle\psi\lvert H \rvert\psi\rangle}{\langle\psi\lvert\psi\rangle} \le \langle\psi(\vec{\theta})\lvert H \rvert\psi(\vec{\theta})\rangle$$
By the Ritz variational principle, any parameterized trial wavefunction $\lvert\psi(\vec{\theta})\rangle$ provides an upper bound on $E_0$.

### Molecular Hamiltonian & Jordan-Wigner Transformation
Second-quantized electronic Hamiltonians:
$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r$$
are mapped onto qubit operators via the Jordan-Wigner isomorphism:
$$a_j^\dagger = \frac{1}{2}(X_j - i Y_j) \otimes Z_{j-1} \dots Z_0$$
yielding a linear combination of Pauli strings:
$$H = \sum_k c_k P_k, \quad P_k \in \{I, X, Y, Z\}^{\otimes n}$$

### Expectation Evaluation & Ansatz
The total energy expectation is computed term-by-term:
$$\langle H \rangle_{\vec{\theta}} = \sum_k c_k \langle\psi(\vec{\theta})\lvert P_k \rvert\psi(\vec{\theta})\rangle$$
We deploy a Hardware-Efficient Ansatz consisting of alternating single-qubit rotations $R_y(\theta)$ and entangling CNOT gates.

## 2. Implementation Details
The `VQE` class in `R03_Variational_Quantum_Eigensolver_VQE.py`:
- Decomposes the STO-3G minimal-basis $H_2$ Hamiltonian into 6 Pauli terms.
- Implements hardware-efficient 4-parameter ansatz state preparation.
- Employs gradient descent with finite-difference numerical gradients.
- Compares computed eigenvalues with exact Hermitian diagonalization via `np.linalg.eigh`.

## 3. Asymptotic & Numerical Analysis
- **Measurement Overhead**: Measuring $M$ Pauli terms with statistical precision $\epsilon$ requires $\mathcal{O}(M / \epsilon^2)$ quantum circuit executions (shots).
- **Chemical Accuracy**: The standard chemical benchmark requires precision within $1 \text{ kcal/mol} \approx 1.59 \times 10^{-3} \text{ Hartree}$.

## 4. Edge Cases & Numerical Stability
- **Barren Plateaus**: In deep random ansätze, gradients vanish exponentially in $n$. Using shallow, symmetry-preserving hardware-efficient structures maintains healthy gradient norms.
- **Hermiticity**: The decomposed Hamiltonian matrix is strictly Hermitian ($H = H^\dagger$), guaranteeing real eigenvalues.

## 5. Standalone Verification
The test verifies:
- Ground energy convergence from arbitrary random initial parameter initialization.
- Absolute error against exact diagonalization $< 1.6 \times 10^{-3} \text{ Hartree}$ (reaching chemical accuracy).

## 6. Real-World Applications & Context
Primary algorithm for near-term NISQ devices in quantum chemistry, catalyst discovery, battery materials, and condensed matter physics.

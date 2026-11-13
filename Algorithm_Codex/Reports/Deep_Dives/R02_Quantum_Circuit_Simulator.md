# R02: Quantum Circuit Simulator

## 1. Mathematical Formulation & Architecture
An $n$-qubit quantum state vector $\lvert\psi\rangle$ contains $2^n$ complex amplitudes. Rather than forming explicit $(2^n \times 2^n)$ unitary matrices, an efficient state-vector simulator treats the state as an $n$-dimensional tensor $\mathcal{T} \in (\mathbb{C}^2)^{\otimes n}$.

### Tensor Gate Application
Applying a single-qubit gate $U \in \mathbb{C}^{2 \times 2}$ to target qubit $k$ corresponds to tensor contraction:
$$\mathcal{T}'_{i_0, \dots, i_{k-1}, j, i_{k+1}, \dots, i_{n-1}} = \sum_{i_k=0}^1 U_{j, i_k} \mathcal{T}_{i_0, \dots, i_{k-1}, i_k, i_{k+1}, \dots, i_{n-1}}$$

### Projective Measurement & Born Rule
Measuring qubit $k$ yields binary outcome $m \in \{0, 1\}$ with probability:
$$P(m) = \sum_{i_0, \dots, i_{k-1}, i_{k+1}, \dots, i_{n-1}} \lvert\mathcal{T}_{i_0, \dots, i_{k-1}, m, i_{k+1}, \dots, i_{n-1}}\rvert^2$$
Upon measurement, the state collapses via non-unitary projection:
$$\mathcal{T} \leftarrow \frac{\Pi_m^{(k)} \mathcal{T}}{\sqrt{P(m)}}, \quad \Pi_m^{(k)} = \lvert m\rangle\langle m\rvert_k$$

### Quantum Teleportation Protocol
Given an unknown state $\lvert\psi\rangle = \alpha\lvert 0\rangle + \beta\lvert 1\rangle$ and an EPR pair $\lvert\Phi^+\rangle_{12} = \frac{1}{\sqrt{2}}(\lvert 00\rangle + \lvert 11\rangle)$:
1. Alice couples $\lvert\psi\rangle_0$ with her entangled qubit $\lvert\Phi^+\rangle_1$ via $\text{CNOT}_{01}$ followed by $H_0$.
2. The joint 3-qubit state expands into Bell-basis decompositions:
   $$\lvert\Psi_{012}\rangle = \frac{1}{2}\left(\lvert 00\rangle (\alpha\lvert 0\rangle + \beta\lvert 1\rangle) + \lvert 01\rangle (\alpha\lvert 1\rangle + \beta\lvert 0\rangle) + \lvert 10\rangle (\alpha\lvert 0\rangle - \beta\lvert 1\rangle) + \lvert 11\rangle (\alpha\lvert 1\rangle - \beta\lvert 0\rangle)\right)$$
3. Alice measures qubits 0 and 1, transmitting 2 classical bits $(m_0, m_1)$ to Bob.
4. Bob applies correction $Z^{m_0} X^{m_1}$ to qubit 2, recovering exact state $\lvert\psi\rangle$ with fidelity $F = 1$.

## 2. Implementation Details
The `QuantumCircuit` class in `R02_Quantum_Circuit_Simulator.py`:
- Employs `np.tensordot` and `np.moveaxis` to avoid full Kronecker expansions.
- Supports arbitrary multi-qubit CNOT with flexible control-target assignments.
- Implements exact state collapse and projective measurement.
- Implements and verifies end-to-end teleportation.

## 3. Asymptotic & Numerical Analysis
- **Time Complexity**: A single-qubit gate on an $n$-qubit state requires $\mathcal{O}(2^n)$ complex multiply-adds.
- **Space Complexity**: $\mathcal{O}(2^n)$ complex entries (e.g. $n=20$ takes $pprox 16$ MB).

## 4. Edge Cases & Numerical Stability
- Zero Probability Collapse: When projecting onto an outcome with $P(m) = 0$, normalization is safeguarded against division by zero.
- Preserving Phase Coherence: All matrix multiplications use `complex128`.

## 5. Standalone Verification
The test suite validates:
- GHZ tripartite entangled state generation: $P(000) = 0.5, P(111) = 0.5$.
- Quantum teleportation across random complex amplitudes, asserting $F > 1 - 10^{-7}$.

## 6. Real-World Applications & Context
Simulates quantum algorithms up to $\sim 25$ qubits classically, serving as the execution backend for VQE, QAOA, and error-correction benchmarking.

# R08: Quantum Error Correction (3-Qubit Bit-Flip Code)

## 1. Mathematical Formulation & Architecture
Quantum Error Correction (QEC) overcomes the no-cloning theorem and environmental decoherence by encoding logical quantum information non-locally into entangled multi-qubit states.

### 3-Qubit Bit-Flip Code
The bit-flip code protects a single logical qubit against Pauli $X$ errors:
$$\lvert 0_L\rangle = \lvert 000\rangle, \quad \lvert 1_L\rangle = \lvert 111\rangle$$
Arbitrary logical states take the form:
$$\lvert\psi_L\rangle = \alpha \lvert 000\rangle + \beta \lvert 111\rangle$$

### Stabilizer Formalism & Non-Destructive Syndromes
The code space $\mathcal{C}$ is stabilized by the Abelian stabilizer group $\mathcal{S} = \langle S_1, S_2\rangle$:
$$S_1 = Z_1 Z_2 I_3, \quad S_2 = I_1 Z_2 Z_3$$
Stabilizers satisfy $S_i \lvert\psi_L\rangle = +1 \lvert\psi_L\rangle$. When a single bit-flip $X_j$ occurs, it anti-commutes with specific stabilizers:
- **No error**: $S_1 = +1, S_2 = +1 \implies (0, 0)$
- **Error on Qubit 1 ($X_1$)**: $S_1 = -1, S_2 = +1 \implies (1, 0)$
- **Error on Qubit 2 ($X_2$)**: $S_1 = -1, S_2 = -1 \implies (1, 1)$
- **Error on Qubit 3 ($X_3$)**: $S_1 = +1, S_2 = -1 \implies (0, 1)$

Because measuring stabilizers $Z_i Z_j$ measures parity without measuring individual $Z_i$, the superposition amplitudes $\alpha$ and $\beta$ remain completely uncollapsed and coherent.

## 2. Implementation Details
The Python module `R08_Quantum_Error_Correction_Bit_Flip_Code.py`:
- Prepares logical 3-qubit codewords.
- Simulates Pauli $X$ errors across physical qubits.
- Evaluates expectation values of stabilizer generators $S_1$ and $S_2$.
- Maps measured syndromes $(b_1, b_2)$ to targeted single-qubit Pauli recovery gates.
- Measures state recovery fidelity $F = \lvert\langle\psi_L\lvert\psi_{\text{rec}}\rangle\rvert^2$.

## 3. Asymptotic & Numerical Analysis
- **Error Suppression**: For independent physical error rate $p < 1/2$, the logical error probability drops to $P_L = 3p^2(1-p) + p^3 = 3p^2 - 2p^3 < p$, establishing the foundational threshold theorem of fault-tolerant quantum computing.

## 4. Edge Cases & Numerical Stability
- Handles multi-qubit error events: A 2-qubit bit-flip is diagnosed as a single bit-flip on the unaffected qubit, demonstrating the fundamental code distance $d = 1$ limits of repetition codes.
- Numerical inner products retain exact $1.0$ fidelity under double-precision arithmetic.

## 5. Standalone Verification
The test suite validates:
- Correct syndrome extraction across all single-qubit error cases $(0, 1, 2)$ and clean states.
- Exact $F = 1.000000$ quantum fidelity recovery for all single bit-flip scenarios.

## 6. Real-World Applications & Context
The foundational building block of the 9-qubit Shor code, Steane code, and topological Surface Codes running on Google Quantum AI, IBM Quantum, and Quantinuum hardware.

# R07: Tensor Networks and Matrix Product States (MPS)

## 1. Mathematical Formulation & Architecture
Tensor networks provide a computationally tractable ansatz for quantum many-body systems by exploiting the area law of entanglement entropy. In 1D spin systems, any state $\lvert\psi\rangle \in (\mathbb{C}^d)^{\otimes L}$ can be decomposed as a **Matrix Product State (MPS)**:
$$\lvert\psi\rangle = \sum_{s_1, \dots, s_L=1}^d A^{[1]}_{s_1} A^{[2]}_{s_2} \dots A^{[L]}_{s_L} \lvert s_1 s_2 \dots s_L\rangle$$
where each $A^{[i]}_{s_i}$ is a matrix of dimension $\chi_{i-1} \times \chi_i$ with boundary conditions $\chi_0 = \chi_L = 1$. The parameter $\chi = \max_i \chi_i$ is the **bond dimension**.

### Schmidt Decomposition & Entanglement Entropy
For a bipartition of the chain into subsystems $A = [1, \dots, k]$ and $B = [k+1, \dots, L]$, the Schmidt decomposition is:
$$\lvert\psi\rangle = \sum_{\alpha=1}^{\min(d^k, d^{L-k})} \lambda_\alpha \lvert\phi_\alpha^A\rangle \lvert\phi_\alpha^B\rangle, \quad \sum_\alpha \lambda_\alpha^2 = 1$$
The von Neumann entanglement entropy across the cut is:
$$S(A) = -\operatorname{Tr}(\rho_A \ln \rho_A) = -\sum_\alpha \lambda_\alpha^2 \ln(\lambda_\alpha^2)$$
According to the 1D Area Law (Hastings 2007), gapped ground states satisfy $S(A) \le \text{const}$, guaranteeing that $\lambda_\alpha$ decay exponentially and can be truncated to finite bond dimension $\chi$ with bounded error.

## 2. Implementation Details
The `MatrixProductState` class in `R07_Tensor_Networks_and_Matrix_Product_States.py`:
- Converts full quantum states into canonical MPS representations via successive Singular Value Decompositions (SVD).
- Enforces bond dimension truncation $\chi \le \chi_{\max}$.
- Contracts the tensor train via `np.tensordot` to reconstruct the full state vector.
- Computes exact Schmidt singular values and von Neumann entanglement entropy.

## 3. Asymptotic & Numerical Analysis
- **Storage Complexity**: Compressed from exponential $\mathcal{O}(d^L)$ down to linear $\mathcal{O}(L d \chi^2)$.
- **Contraction Complexity**: Scalar product $\langle\phi\lvert\psi\rangle$ and local observables evaluate in $\mathcal{O}(L d \chi^3)$.

## 4. Edge Cases & Numerical Stability
- **Zero Singular Values**: Small singular values ($\lambda < 10^{-12}$) are truncated prior to evaluating $x \ln x$ to prevent $\text{NaN}$ in logarithmic entropy calculations.
- **Orthonormality**: Left and right orthogonal canonical forms guarantee numerical conditioning during iterative sweeps.

## 5. Standalone Verification
The test suite validates:
- Zero entanglement entropy on product states ($S = 0.0$).
- Exact bipartite entanglement entropy for GHZ state: $S = \ln(2) \approx 0.693147$.
- Exact wave function reconstruction from full SVD MPS representation ($F = 1.0$).

## 6. Real-World Applications & Context
Core mathematical architecture underlying DMRG (Density Matrix Renormalization Group), TEBD (Time-Evolving Block Decimation), and modern quantum circuit classical emulation.

# R04: Quantum Approximate Optimization Algorithm (QAOA)

## 1. Mathematical Formulation & Architecture
The Quantum Approximate Optimization Algorithm (QAOA) is a polynomial-time variational heuristic designed by Farhi, Goldstone, and Gutmann (2014) to solve combinatorial optimization problems (e.g. MaxCut).

### MaxCut Formulation
Given an undirected graph $G = (V, E)$, MaxCut seeks a binary assignment $s_i \in \{-1, +1\}$ maximizing:
$$C = \sum_{(u, v) \in E} \frac{1 - s_u s_v}{2}$$
Promoted to a diagonal quantum operator in computational basis:
$$H_C = \sum_{(u, v) \in E} \frac{1}{2}(I - Z_u Z_v)$$

### Alternating Quantum Circuit
Starting from the uniform superposition $\lvert +\rangle^{\otimes n}$, QAOA applies $p$ alternating unitary layers:
$$\lvert\psi(\vec{\gamma}, \vec{\beta})\rangle = \left( \prod_{k=1}^p e^{-i \beta_k H_M} e^{-i \gamma_k H_C} \right) \lvert +\rangle^{\otimes n}$$
where the mixer Hamiltonian is the transverse magnetic field:
$$H_M = \sum_{j \in V} X_j, \quad e^{-i \beta_k X_j} = \cos(\beta_k) I - i \sin(\beta_k) X_j$$
The classical optimization objective is to maximize:
$$F_p(\vec{\gamma}, \vec{\beta}) = \langle\psi(\vec{\gamma}, \vec{\beta})\lvert H_C \rvert\psi(\vec{\gamma}, \vec{\beta})\rangle$$

## 2. Implementation Details
The `QAOA` implementation in `R04_Quantum_Approximate_Optimization_Algorithm_QAOA.py`:
- Encodes arbitrary graph topologies via node and edge specifications.
- Evaluates $U(C, \gamma)$ via basis phase-shifts in $\mathcal{O}(2^n)$ time.
- Applies mixer layers $U(B, \beta)$ across tensor axes via `np.tensordot`.
- Performs parameter optimization to maximize cut expectation and tests approximation ratio $\alpha = \langle C \rangle / C_{\max}$.

## 3. Asymptotic & Numerical Analysis
- **Approximation Guarantees**: As $p \to \infty$, QAOA converges to adiabatic state preparation, yielding the exact ground state (approximation ratio $1.0$). For $p=1$ on 3-regular graphs, QAOA achieves $\alpha \ge 0.692$.
- **Circuit Depth**: For depth $p$, circuit complexity scales as $\mathcal{O}(p \lvert E\rvert)$.

## 4. Edge Cases & Numerical Stability
- **Periodicity & Symmetry**: The expectation $F_1(\gamma, \beta)$ is periodic in $\gamma \in [0, 2\pi)$ and $\beta \in [0, \pi)$, bounding the parameter search space.
- **Graph Disconnectedness**: Handles isolated nodes gracefully since isolated qubits experience only the mixer rotations.

## 5. Standalone Verification
The test suite validates:
- Performance on 4-cycle graph $C_4$.
- Approximation ratio $\alpha \ge 0.75$.
- Strong probability concentration ($P > 40\%$) onto exact optimal partitions ($0101$ and $1010$).

## 6. Real-World Applications & Context
Applicable to NP-hard scheduling, network partitioning, Portfolio optimization, and logistics on near-term NISQ processors.

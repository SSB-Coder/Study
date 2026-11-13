# Deep Dive: U05 — Circuit Modified Nodal Analysis (MNA)

## 1. The Beginner's Analogy: Water Pipe Pressure Networks
Imagine a network of water pipes, pumps, and valves supplying a skyscraper:
- Water flows from high pressure to low pressure through narrow pipes (resistors).
- At every pipe junction (node), water cannot vanish into thin air: total flow entering must equal total flow leaving (Kirchhoff's Current Law).
- Certain booster pumps force an exact pressure jump across their flanges regardless of flow rate (independent voltage sources).
- **Modified Nodal Analysis (MNA)** is the exact mathematical formulation used inside SPICE to assemble every junction's flow conservation law into a linear system of equations and solve for every pressure and flow rate in the network.

---

## 2. The Deep Architecture/Algorithm Breakdown
Developed by Ho, Ruehli, and Brennan (1975), Modified Nodal Analysis is the mathematical core of SPICE circuit simulators:
1. **Limitation of Standard Nodal Analysis:** Standard nodal analysis uses Kirchhoff's Current Law (KCL) $\sum i = 0$ with $i = G \Delta v$. However, ideal voltage sources have zero internal resistance ($G = \infty$), making admittance matrices undefined.
2. **MNA Formulation:** Augments the node voltage variables $\mathbf{v}$ with the unknown branch currents $\mathbf{j}$ flowing through voltage sources:
   $$\begin{bmatrix} G & B \\ B^T & D \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ \mathbf{j} \end{bmatrix} = \begin{bmatrix} \mathbf{i} \\ \mathbf{e} \end{bmatrix}$$
   - $G \in \mathbb{R}^{n \times n}$: Inter-node conductance matrix.
   - $B \in \mathbb{R}^{n \times m}$: Incidence matrix linking voltage sources to nodes ($+1$ for positive terminal, $-1$ for negative).
   - $D \in \mathbb{R}^{m \times m}$: Zero matrix for independent voltage sources.
   - $\mathbf{i} \in \mathbb{R}^n$: Current injection vector.
   - $\mathbf{e} \in \mathbb{R}^m$: Voltage values vector.

---

## 3. The Hardcore Mathematics

### 3.1 Conductance Stamping Rules
For a resistor $R$ connected between nodes $k$ and $l$ with conductance $g = 1/R$:
$$G_{kk} \leftarrow G_{kk} + g, \quad G_{ll} \leftarrow G_{ll} + g$$
$$G_{kl} \leftarrow G_{kl} - g, \quad G_{lk} \leftarrow G_{lk} - g$$

### 3.2 Voltage Source Stamping Rules
For voltage source $V_p$ with value $E$ between node $k$ (+) and node $l$ (-):
$$B_{k, p} = +1, \quad B_{l, p} = -1, \quad e_p = E$$
The lower equation $B^T \mathbf{v} = \mathbf{e}$ directly enforces the branch constraint $v_k - v_l = E$.

---

## 4. The Algorithm Step-by-Step
1. **Index Nodes:** Assign integer IDs to all circuit nodes, designating Node 0 as Ground reference.
2. **Allocate Matrices:** Size $(N-1 + M) \times (N-1 + M)$.
3. **Stamp Passive Elements:** For each resistor, stamp conductance $g$ into $G$.
4. **Stamp Active Elements:** Stamp independent current sources into $\mathbf{i}$ and voltage sources into $B$ and $\mathbf{e}$.
5. **Solve:** Solve linear system using LU decomposition (V02).

---

## 5. Limitations / When It Breaks
- **Non-Linear Components:** Diodes and transistors (MOSFETs, BJTs) have non-linear $I-V$ relationships ($I = I_s (e^{V/V_t} - 1)$), requiring multi-variable Newton-Raphson iterations (I02, V05) wrapped around the MNA linear solver.

---

## 6. Where This Fits
- **Cross-Reference LU Decomposition (V02):** Directly validates the linear system solving machinery.

# Deep Dive: V02 — LU Decomposition & Linear System Solving

## 1. The Beginner's Analogy: Gaussian Elimination on Record
Imagine solving high school algebra systems with multiple unknowns ($x, y, z$). Instead of erasing and re-solving every time the right-hand numbers ($b$) change:
- **Gaussian elimination** eliminates variables row by row, converting the mess into an organized "staircase" (upper triangular matrix $U$).
- **LU Decomposition** records every single subtraction multiplier you used during that elimination inside a lower-triangular matrix $L$.
- **Partial Pivoting ($P$)** acts like a safety inspector: before dividing by any number on the diagonal, it scans the column below, finds the largest value, and swaps that row to the top. This prevents dividing by tiny numbers that would blow up floating-point roundoff errors.

---

## 2. The Deep Architecture/Algorithm Breakdown
Given non-singular $A \in \mathbb{R}^{n \times n}$, LU factorization with partial row pivoting computes:
$$P A = L U$$
where:
1. $P$ is an $n \times n$ permutation matrix with $\det(P) \in \{+1, -1\}$.
2. $L$ is unit lower-triangular ($L_{ii} = 1$, $L_{ij} = 0$ for $j > i$).
3. $U$ is upper-triangular ($U_{ij} = 0$ for $i > j$).

To solve $A x = b$:
1. Permute right-hand side: $P A x = L U x = P b$.
2. Forward substitution: solve $L y = P b$ in $\mathcal{O}(n^2)$ operations.
3. Backward substitution: solve $U x = y$ in $\mathcal{O}(n^2)$ operations.

---

## 3. The Hardcore Mathematics

### 3.1 Partial Pivoting Criterion
At step $k \in \{1, \dots, n-1\}$, identify pivot row $p$:
$$p = \arg\max_{i \in \{k, \dots, n\}} |U_{ik}|$$
Swap rows $k$ and $p$ in $U, P,$ and the already-computed subdiagonal elements of $L$.

### 3.2 Elimination Update
For each row $i \in \{k+1, \dots, n\}$:
$$l_{ik} = \frac{U_{ik}}{U_{kk}}$$
$$U_{i, k:n} \leftarrow U_{i, k:n} - l_{ik} U_{k, k:n}$$

### 3.3 Substitution Formulas
Forward substitution ($L y = b'$):
$$y_i = b'_i - \sum_{j=1}^{i-1} L_{ij} y_j, \quad i = 1, \dots, n$$
Backward substitution ($U x = y$):
$$x_i = \frac{1}{U_{ii}} \left( y_i - \sum_{j=i+1}^n U_{ij} x_j \right), \quad i = n, \dots, 1$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** $P = I_n, L = I_n, U = A.copy(), \text{swaps} = 0$.
2. **Triangularize ($k = 1 \dots n-1$):**
   - Find maximum $|U_{ik}|$ for $i \ge k$.
   - If $|U_{pk}| < \epsilon$, matrix is numerically singular.
   - Swap rows $k$ and $p$ in $U$, $P$, and subdiagonal $L$. Increment swaps if $p \ne k$.
   - Compute multipliers $L_{ik} = U_{ik} / U_{kk}$ and subtract $L_{ik} U_{k, k:n}$ from $U_{i, k:n}$.
3. **Solve for any RHS $b$:**
   - Compute $Pb$. Solve $Ly = Pb$ forward.
   - Solve $Ux = y$ backward. Return $x$.

---

## 5. Limitations / When It Breaks
- **Near-Singular Matrices:** If $\kappa(A) \gg 1$, numerical roundoff causes substantial loss of precision. Condition number estimation or SVD/QR should be used.
- **Sparse Structures:** For large sparse matrices (e.g. from PDE discretizations or circuit netlists), naive dense LU produces fill-in that destroys sparsity. Graph-coloring and fill-reducing reorderings (Reverse Cuthill-McKee, AMD) are essential.

---

## 6. Where This Fits
LU decomposition is the backbone for:
1. **Circuit Nodal Analysis (U05):** MNA formulations create $G v = i$ linear systems solved by LU.
2. **Finite Element Analysis (W10):** Structural truss equations $K u = f$ are assembled and solved via LU.
3. **Newton-Raphson Optimization (I02, V05):** Solving Hessian Newton systems $H \Delta \theta = -g$.

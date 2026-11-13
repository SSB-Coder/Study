# Deep Dive: V01 — QR Decomposition (Gram-Schmidt & Householder)

## 1. The Beginner's Analogy: Building Orthogonal Shadow Boxes
Imagine building a 3D wooden sculpture where every strut must meet at perfect 90-degree right angles:
- **Classical Gram-Schmidt** takes a set of tilted wooden poles. It fixes the first pole vertically, then casts shadows of subsequent poles onto the first and saws off the overlapping shadow so the remainder is orthogonal. However, tiny sawing errors accumulate, and by the 10th pole, the poles drift away from true 90 degrees.
- **Modified Gram-Schmidt (MGS)** immediately subtracts the projection from *all* remaining unbuilt poles as soon as a new orthogonal axis is locked down, preventing catastrophic loss of orthogonality.
- **Householder Reflections** take a completely different, mirror-like approach: instead of projecting vector by vector, it builds a multi-dimensional mirror that reflects an entire submatrix column in one strike so all its entries below the diagonal vanish into zero. Householder reflections are unconditionally stable to machine precision.

---

## 2. The Deep Architecture/Algorithm Breakdown
Given a matrix $A \in \mathbb{R}^{m \times n}$ with $m \ge n$, QR decomposition factors $A$ into:
$$A = Q R$$
where $Q \in \mathbb{R}^{m \times m}$ is orthogonal ($Q^T Q = Q Q^T = I_m$), and $R \in \mathbb{R}^{m \times n}$ is upper-triangular ($R_{ij} = 0$ for $i > j$). In "thin" or "reduced" QR, $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^T Q = I_n$) and $R \in \mathbb{R}^{n \times n}$.

### 2.1 Modified Gram-Schmidt vs. Classical Gram-Schmidt
In Classical Gram-Schmidt (CGS), rounding errors lead to loss of orthogonality: $\|Q^T Q - I\| = \mathcal{O}(\kappa(A) \epsilon_{\text{mach}})$. In Modified Gram-Schmidt (MGS), the intermediate projections are updated progressively:
$$v_j^{(k+1)} = v_j^{(k)} - \langle q_k, v_j^{(k)} \rangle q_k$$
which significantly improves stability: $\|Q^T Q - I\| = \mathcal{O}(\epsilon_{\text{mach}})$.

### 2.2 Householder Triangularization
Householder transformations apply a sequence of orthogonal elementary reflectors $H_k = I - 2 v_k v_k^T / \|v_k\|^2$ such that:
$$H_n \dots H_2 H_1 A = R$$
Since each $H_k$ is orthogonal and symmetric, $Q = H_1 H_2 \dots H_n$.

---

## 3. The Hardcore Mathematics

### 3.1 Gram-Schmidt Projection
For column $k$, the normalized basis vector is:
$$q_k = \frac{v_k}{\|v_k\|}, \quad R_{kk} = \|v_k\|$$
For every subsequent column $j > k$:
$$R_{kj} = q_k^T v_j, \quad v_j \leftarrow v_j - R_{kj} q_k$$

### 3.2 Householder Vector Derivation
To zero out all subdiagonal entries of vector $x = [x_1, x_2, \dots, x_p]^T$, we seek a reflection vector $v$ such that:
$$H x = (I - 2 \frac{v v^T}{v^T v}) x = \alpha e_1$$
Because orthogonal transformations preserve Euclidean norms, $\|H x\| = \|x\| \implies |\alpha| = \|x\|_2$.
To prevent catastrophic numerical cancellation when $x_1$ has the same sign as $\alpha$, we choose:
$$\alpha = -\text{sgn}(x_1) \|x\|_2$$
$$v = x - \alpha e_1 = \begin{bmatrix} x_1 - \alpha \\ x_2 \\ \vdots \\ x_p \end{bmatrix}$$
Normalized reflector: $P = I - 2 \frac{v v^T}{v^T v}$.

---

## 4. The Algorithm Step-by-Step

### 4.1 Householder QR Algorithm
1. **Initialize:** Set $R = A$, $Q = I_m$.
2. **Loop over columns $k = 1, \dots, \min(m-1, n)$:**
   - Extract sub-vector $x = R[k:m, k]$.
   - Compute $\alpha = -\text{sgn}(x_1) \|x\|_2$.
   - Form Householder vector $v = x - \alpha e_1$, normalize $\hat{v} = v / \|v\|_2$.
   - Apply update to submatrix: $R[k:m, k:] \leftarrow R[k:m, k:] - 2 \hat{v} (\hat{v}^T R[k:m, k:])$.
   - Accumulate into orthogonal matrix: $Q[:, k:m] \leftarrow Q[:, k:m] - 2 (Q[:, k:m] \hat{v}) \hat{v}^T$.
3. **Return:** $Q$ and upper-triangular $R$.

---

## 5. Limitations / When It Breaks
- **Rank-Deficient Matrices:** When $A$ has linearly dependent columns, $R_{kk} = 0$ or falls below precision. Pivoting (Column-Pivoted QR / RRQR) is required to determine the numerical rank.
- **Storage Complexity:** Explicitly forming the $m \times m$ matrix $Q$ requires $\mathcal{O}(m^2)$ memory. In large-scale systems, $Q$ is stored implicitly as the sequence of Householder vectors $v_1, \dots, v_n$.

---

## 6. Where This Fits
QR decomposition is the foundational numerical linear algebra workhorse underlying:
1. **V03 Eigenvalue Algorithm:** QR iteration drives general matrix eigensolvers.
2. **Ordinary Least Squares (B01):** Solving $A x = b$ via $R x = Q^T b$ avoids the condition-squaring disaster of normal equations $A^T A x = A^T b$ (since $\kappa(A^T A) = \kappa(A)^2$).
3. **Kalman Filters (E06, X04):** Square-root information filters maintain covariance matrices in factored square-root form via QR to guarantee positive semi-definiteness.

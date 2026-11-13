# Deep Dive: V03 — The QR Algorithm for Eigenvalues

## 1. The Beginner's Analogy: The Shuffling Deck of Diagonalization
Imagine a deck of cards where numbers on adjacent cards bleed into each other:
- If you repeatedly split the deck (QR factorize: $A = QR$) and put it back together in reverse order ($A_{k+1} = RQ$), a strange and beautiful mathematical phenomenon occurs: the cross-talk numbers off the diagonal diminish with every shuffle!
- As the shuffles progress, all off-diagonal values shrink to zero, leaving only pure, isolated eigenvalues gleaming along the main diagonal.
- Applying a **Wilkinson shift** is like magnetizing the bottom cards: it pulls the trailing diagonal elements toward their true values at cubic convergence speed instead of sluggish linear crawls.

---

## 2. The Deep Architecture/Algorithm Breakdown
The QR algorithm computes the Schur decomposition (or spectral decomposition for symmetric matrices):
$$A_0 = A$$
$$A_k - \mu_k I = Q_k R_k \implies A_{k+1} = R_k Q_k + \mu_k I$$
Notice that:
$$A_{k+1} = Q_k^T (A_k - \mu_k I) Q_k + \mu_k I = Q_k^T A_k Q_k$$
Thus, $A_{k+1}$ is strictly **orthogonally similar** to $A_k$, preserving all eigenvalues identically:
$$\text{spec}(A_{k+1}) = \text{spec}(A_k)$$
As $k \to \infty$, $A_k$ converges to upper-triangular form (Schur form) whose diagonal contains the eigenvalues.

---

## 3. The Hardcore Mathematics

### 3.1 Orthogonal Similarity Invariant
Let $Q^{(k)} = Q_0 Q_1 \dots Q_k$. Then:
$$A_{k+1} = (Q^{(k)})^T A (Q^{(k)})$$
Because orthogonal transformations preserve matrix trace, determinant, and spectral norm:
$$\text{tr}(A_{k+1}) = \text{tr}(A), \quad \det(A_{k+1}) = \det(A)$$

### 3.2 Wilkinson Shift Derivation
For symmetric matrices, convergence without shifts is linear: $|a_{n, n-1}^{(k)}| = \mathcal{O}(|\lambda_n / \lambda_{n-1}|^k)$.
To accelerate convergence to cubic order, we pick shift $\mu$ as the eigenvalue of the trailing $2 \times 2$ submatrix closest to $a_{nn}$:
$$\begin{bmatrix} a_{n-1, n-1} & a_{n-1, n} \\ a_{n, n-1} & a_{nn} \end{bmatrix}$$
Let $d = \frac{a_{n-1, n-1} - a_{nn}}{2}$. The Wilkinson shift is:
$$\mu = a_{nn} - \frac{\text{sgn}(d) a_{n, n-1}^2}{|d| + \sqrt{d^2 + a_{n, n-1}^2}}$$

---

## 4. The Algorithm Step-by-Step
1. **Tridiagonalize/Hessenberg Pre-solve:** For efficient implementation, reduce $A$ to tridiagonal (or upper Hessenberg) form via Householder similarity transforms in $\mathcal{O}(n^3)$.
2. **Iterate:** For iteration $k = 1, 2, \dots, \text{max\_iter}$:
   - Check off-diagonal norm: $\sum_{i \ne j} A_{ij}^2 < \text{tol}^2$. If converged, terminate.
   - Compute shift $\mu_k$ from trailing block.
   - Decompose $A_k - \mu_k I = Q R$.
   - Form $A_{k+1} = R Q + \mu_k I$.
   - Accumulate eigenvectors: $V \leftarrow V Q$.
3. **Extract:** $\lambda_i = A_{ii}$. Sort descending.

---

## 5. Limitations / When It Breaks
- **Complex Conjugate Pairs:** For non-symmetric real matrices, eigenvalues can be complex numbers, appearing as $2 \times 2$ blocks on the block upper-triangular real Schur form (requiring double-shift Francis QR steps).
- **Multiple Eigenvalues:** When eigenvalues cluster, convergence of individual vectors slows, requiring deflation techniques.

---

## 6. Where This Fits
1. **PCA & Spectral Clustering (C02, C05):** Eigendecomposition extracts principal components and graph Laplacian eigenvectors.
2. **PageRank & Power Iteration (L07):** Power iteration computes only the dominant eigenvalue; the QR algorithm extracts the entire spectrum simultaneously.
3. **Quantum Mechanics (R01, R03):** Observables are Hermitian matrices whose eigenvalues represent physical energy levels.

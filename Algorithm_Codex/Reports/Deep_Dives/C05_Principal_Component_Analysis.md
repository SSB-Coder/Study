# Deep Dive: Principal Component Analysis (Eigendecomposition vs. SVD)

## 1. The Beginner's Analogy: Flattening a 3D Sculpture into a 2D Shadow Portrait
Imagine you sculpt a wire statue of a running cheetah in full 3D:
- You want to photograph it from a single angle so that anyone looking at the 2D photo can see as much detail, muscle tone, and body length as possible.
- If you photograph it directly from the front (head-on), the entire 6-foot length of the cheetah collapses into a tiny, squished circle (massive information loss).
- **Principal Component Analysis (PCA)** rotates the camera around the statue to find the exact angle where the cheetah's shadow stretches out the furthest (maximizing variance).
- **First Principal Axis (PC1):** The spine of the cheetah (largest variance).
- **Second Principal Axis (PC2):** Perpendicular to PC1, capturing the leg extension.
- **Third Axis (PC3):** The thickness of the wire, which is tiny and can be safely discarded without losing the essence of the cheetah.

---

## 2. The Deep Architecture/Algorithm Breakdown
PCA is an orthogonal linear transformation projecting data onto a lower-dimensional subspace while preserving maximal variance:
1. **Centering:** Subtraction of the empirical mean $\mu = \frac{1}{m}\sum x_i$ is mandatory; uncentered PCA projects toward the origin rather than the data center.
2. **Covariance Formulation:**
   $$\Sigma = \frac{1}{m} X^T X = V \Lambda V^T$$
   Where $V$ are orthonormal eigenvectors and $\Lambda$ are eigenvalues.
3. **Singular Value Decomposition (SVD) Formulation:**
   $$X = U S V^T$$
   Where $V$ are the exact same right singular vectors, and singular values relate via $\lambda_j = \frac{s_j^2}{m}$.
4. **Explained Variance Ratio:** Fraction of total variance captured by the top $k$ components:
   $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}$$
5. **Reconstruction:** Low-rank approximation $\hat{X} = X_k V_k^T + \mu$.

---

## 3. The Hardcore Mathematics

### 3.1 Maximum Variance Derivation
Let centered data be $X \in \mathbb{R}^{m \times d}$. We seek unit projection vector $w \in \mathbb{R}^d$ ($\|w\|_2 = 1$) maximizing sample variance:
$$\max_w \frac{1}{m} \sum_{i=1}^m (x_i^T w)^2 = \max_w w^T \left( \frac{1}{m} X^T X \right) w = \max_w w^T \Sigma w \quad \text{s.t.} \quad w^T w = 1$$
Formulate the Lagrangian:
$$\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)$$
Differentiating and setting to zero:
$$\nabla_w \mathcal{L} = 2 \Sigma w - 2 \lambda w = 0 \implies \Sigma w = \lambda w$$
This is the classic matrix **eigenvalue equation**. The vector $w$ maximizing variance is the eigenvector corresponding to the largest eigenvalue $\lambda_1$.

### 3.2 Proof of Equivalence: Eigendecomposition vs. SVD
Let $X = U S V^T$, where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{d \times d}$ have orthonormal columns ($U^T U = I, V^T V = I$), and $S = \text{diag}(s_1, \dots, s_r)$ are singular values.
Compute the sample covariance:
$$\Sigma = \frac{1}{m} X^T X = \frac{1}{m} (U S V^T)^T (U S V^T) = \frac{1}{m} V S^T U^T U S V^T = V \left( \frac{S^2}{m} \right) V^T$$
Since $\Lambda = \frac{S^2}{m}$ is diagonal, this is the exact spectral decomposition of $\Sigma$:
$$\lambda_j = \frac{s_j^2}{m}$$
The right singular vectors $V$ of data matrix $X$ are mathematically identical to the eigenvectors of covariance matrix $\Sigma$. SVD computes this without forming $X^T X$, avoiding $O(d^2)$ memory and precision loss.

---

## 4. The Algorithm Step-by-Step
1. **Center Data:** $X_c = X - \text{mean}(X, \text{axis}=0)$.
2. **Compute Components:**
   - **Method A (Covariance):** Form $\Sigma = \frac{1}{m} X_c^T X_c$, compute `eig(Sigma)`, sort eigenvalues descending.
   - **Method B (SVD):** Compute `svd(X_c)`, extract right singular vectors $V^T$.
3. **Select Top $k$ Components:** Form projection matrix $W_k = V_{:, 1:k} \in \mathbb{R}^{d \times k}$.
4. **Transform:** $Z = X_c W_k \in \mathbb{R}^{m \times k}$.
5. **Reconstruction:** $\hat{X} = Z W_k^T + \mu$.

---

## 5. Limitations / When It Breaks
- **Linearity Constraint:** Cannot discover non-linear manifolds (e.g. Swiss roll); collapses complex geometries onto overlapping planes.
- **Scale Sensitivity:** Features with arbitrarily large units (e.g. income in dollars vs. age in years) dominate principal components unless standardized to unit variance.
- **Historical Fix:** Kernel PCA, Isomap (Module C09), and t-SNE / UMAP (Modules C07 & C10).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/38_Quantization_and_LoRA.md`. Low-Rank Adaptation (LoRA) decomposes large weight updates $\Delta W = B A$ into low-rank matrices based on the exact same low-rank intrinsic dimensionality principles formalized by PCA.

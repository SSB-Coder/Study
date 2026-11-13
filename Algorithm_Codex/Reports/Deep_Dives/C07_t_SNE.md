# Deep Dive: t-Distributed Stochastic Neighbor Embedding (t-SNE)

## 1. The Beginner's Analogy: Unrolling a Tangled Ball of Wool
Imagine high-dimensional data as a tangled ball of multicolored wool:
- If you simply stomp on the ball from above (PCA linear projection), all the red, blue, and yellow strands crush into an uninterpretable pile in the center (the "Crowding Problem").
- **t-SNE** treats the points like beads connected by soft, elastic springs:
  - In the high-dimensional ball, it measures the probability that bead $i$ would pick bead $j$ as its neighbor under a Gaussian distribution.
  - In the flat 2D tray, it places beads randomly and simulates a **Student-t distribution** (heavy tails).
  - The heavy tails give points extra room to breathe; beads that were close in the 3D ball are pulled together by attractive forces, while beads that were far apart are vigorously pushed away by repulsive forces until distinct colored clusters settle naturally onto the table.

---

## 2. The Deep Architecture/Algorithm Breakdown
t-SNE is a non-linear, probabilistic dimensionality reduction technique designed for cluster visualization:
1. **High-Dimensional Affinities ($p_{j|i}$):** Probability that $x_i$ picks $x_j$ as neighbor under a Gaussian centered at $x_i$ with variance $\sigma_i^2$ determined by **perplexity**.
2. **Symmetric Probabilities ($p_{ij}$):** Joint distribution $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2m}$.
3. **Low-Dimensional Affinities ($q_{ij}$):** Student-t distribution with 1 degree of freedom (Cauchy distribution):
   $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
4. **Kullback-Leibler Divergence Objective:** Minimizes information loss between high-dimensional distribution $P$ and low-dimensional distribution $Q$:
   $$\mathcal{L} = D_{KL}(P \parallel Q) = \sum_{i} \sum_{j \neq i} p_{ij} \ln \frac{p_{ij}}{q_{ij}}$$
5. **Exact Analytical Gradient:** $\frac{\partial \mathcal{L}}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$.

---

## 3. The Hardcore Mathematics

### 3.1 Perplexity and Bisection Search for $\sigma_i$
The perplexity of conditional distribution $P_i$ is defined as $2^{H(P_i)}$:
$$\text{Perp}(P_i) = 2^{-\sum_{j \neq i} p_{j|i} \log_2 p_{j|i}}$$
For a user-specified perplexity (typically $5$ to $50$), a 1D bisection root-finding search is conducted on the precision $\beta_i = \frac{1}{2\sigma_i^2}$ until $|\log_2(\text{Perp}) - H(P_i)| < 10^{-5}$.

### 3.2 The Crowding Problem & The Student-t Solution
In high dimensions, the volume of a sphere grows exponentially with dimension $d$. In 10 dimensions, there is 1,000 times more surface area than in 2 dimensions. If we used a Gaussian distribution in 2D space as well, the moderate-distance neighbors would all crowd into the center.
By using the **Cauchy distribution (Student-t with $\nu=1$)**, the density drops off as an inverse square $\frac{1}{1 + d^2}$ rather than an exponential $e^{-d^2}$. This heavy tail allows moderate and distant points to be mapped far apart in 2D space without blowing up the KL divergence penalty.

### 3.3 Gradient Derivation
$$\frac{\partial \mathcal{L}}{\partial y_i} = 4 \sum_{j=1}^m (p_{ij} - q_{ij}) (y_i - y_j) \left( 1 + \|y_i - y_j\|_2^2 \right)^{-1}$$
Notice the physical intuition:
- If $p_{ij} > q_{ij}$: points were close in high dimensions but far in 2D; the term is positive, exerting an **attractive force**.
- If $p_{ij} < q_{ij}$: points were far in high dimensions but close in 2D; the term is negative, exerting a **repulsive force**.

---

## 4. The Algorithm Step-by-Step
1. **Compute Pairwise Squared Distances:** $D_{ij} = \|x_i - x_j\|^2$.
2. **Find $\sigma_i$ via Binary Search:** Match specified perplexity for each sample $i$, compute $p_{j|i}$.
3. **Symmetrize:** $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2m}$. Apply early exaggeration ($p_{ij} \leftarrow 4 p_{ij}$) during first 50 iterations.
4. **Initialize Low-Dim Embedding:** Sample $y_i \sim \mathcal{N}(0, 10^{-4} I)$.
5. **Gradient Descent Loop:**
   - Compute low-dim pairwise distances and Student-t affinities $q_{ij}$.
   - Evaluate gradient $\frac{\partial \mathcal{L}}{\partial y_i}$.
   - Update embeddings with momentum: $y^{(t+1)} = y^{(t)} - \eta \nabla \mathcal{L} + \alpha(t)(y^{(t)} - y^{(t-1)})$.
6. **Return:** 2D coordinates $Y \in \mathbb{R}^{m \times 2}$.

---

## 5. Limitations / When It Breaks
- **Non-Preservation of Global Geometry:** Distances between distant clusters in t-SNE are essentially arbitrary; only local cluster neighborhood topology is meaningful.
- **Computational Complexity:** Exact t-SNE evaluates all pairwise interactions in $O(N^2)$ time per iteration.
- **Historical Fix:** Barnes-Hut quadtree approximation ($O(N \log N)$) and UMAP (Module C10).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/22_Information_Theory_and_Manifolds.md`. t-SNE remains the premier diagnostic visualization tool used across research papers to visualize learned multi-head attention activation clusters.

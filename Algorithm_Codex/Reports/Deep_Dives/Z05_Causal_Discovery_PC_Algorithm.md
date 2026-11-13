# Z05: Causal Discovery & Graph Learning (The PC Algorithm)

## 1. Mathematical Formulation & Architecture
Traditional statistical machine learning models passive conditional correlations $\mathbb{P}(Y \mid X)$. Causal inference seeks the underlying data-generating mechanisms governing interventions $\mathbb{P}(Y \mid \text{do}(X))$.
While [`J07: Causal Inference`](J07_Causal_Inference.md) estimated causal treatment effects assuming a known directed acyclic graph (DAG), **Causal Discovery** reconstructs the graph structure directly from observational data.

### Foundational Assumptions
1. **Causal Sufficiency**: There are no unobserved common confounders.
2. **Causal Markov Condition**: Every variable $X_i$ is conditionally independent of its non-descendants given its direct parents $\text{Pa}(X_i)$ in DAG $G$:
   $$\mathbb{P}(X_1, \dots, X_d) = \prod_{i=1}^d \mathbb{P}(X_i \mid \text{Pa}(X_i))$$
3. **Faithfulness**: All conditional independencies observed in the probability distribution $\mathbb{P}$ are entailed by the d-separation geometry of $G$ (no parameter cancellations).

---

### The Peter-Clark (PC) Algorithm (Spirtes & Glymour 1991)
The PC algorithm efficiently reconstructs the **Markov Equivalence Class** (represented as a Completed Partially Directed Acyclic Graph, CPDAG) in three systematic phases:

#### Phase 1: Skeleton Discovery via Conditional Independence Tests
1. Initialize a complete undirected graph $G = (V, E)$ where every pair of variables is connected.
2. For conditioning set cardinality $\ell = 0, 1, 2, \dots$:
   - For every adjacent pair $(X_i, X_j)$:
     - For every subset $S \subseteq \text{adj}(X_i) \setminus \{X_j\}$ of size $|S| = \ell$:
       - Test hypothesis $H_0: X_i \perp\!\!\!\perp X_j \mid S$.
       - Under linear Gaussian dynamics, conditional independence is equivalent to vanishing partial correlation $\rho_{ij \cdot S} = 0$.
       - Using submatrix inversion of covariance $\Sigma_{\{i,j\}\cup S}$:
         $$\rho_{ij \cdot S} = -\frac{K_{0,1}}{\sqrt{K_{0,0} K_{1,1}}} \quad \text{where } K = \Sigma_{\{i,j\}\cup S}^{-1}$$
       - Evaluate significance via Fisher's $z$-transform:
         $$z = \frac{1}{2} \ln \left( \frac{1 + \rho}{1 - \rho} \right), \quad \sqrt{N - |S| - 3} \cdot |z| \sim \mathcal{N}(0, 1)$$
       - If $p\text{-value} > \alpha$ (cannot reject independence), delete edge $(X_i, X_j)$ and record $S$ as the separating set: $\text{SepSet}(X_i, X_j) = S$.

#### Phase 2: Unshielded Collider (v-structure) Orientation
For every unshielded triple $X_i - X_k - X_j$ (where $X_i$ and $X_j$ are non-adjacent):
- If $X_k \notin \text{SepSet}(X_i, X_j)$, orient as an unshielded collider:
  $$X_i \longrightarrow X_k \longleftarrow X_j$$
*(Explanation: Conditioning on a common effect creates spurious correlation between independent causes via Berkson's bias; hence, $X_k$ could not have been in the conditioning set that rendered them independent).*

#### Phase 3: Meek's Orientation Rules (Meek 1995)
Iteratively apply logical orientation rules until closure to avoid creating spurious cycles or new colliders:
- **Rule 1**: If $X_i \rightarrow X_k - X_j$ with $X_i \not\sim X_j$, orient $X_k \rightarrow X_j$ (otherwise $X_i \rightarrow X_k \leftarrow X_j$ would form an unshielded collider not detected in Phase 2).
- **Rule 2**: If $X_i \rightarrow X_k \rightarrow X_j$ with $X_i - X_j$, orient $X_i \rightarrow X_j$ (to avoid forming a directed cycle $X_j \rightarrow X_i \rightarrow X_k \rightarrow X_j$).

## 2. Implementation Details
The module `Z05_Causal_Discovery_PC_Algorithm.py`:
- Implements `PCAlgorithm` with exact submatrix pseudo-inverses for partial correlation evaluation.
- Implements Fisher's $z$-transform with complementary error function `erfc` tail probabilities.
- Systematically executes Phase 1 skeleton search, Phase 2 collider orientation, and Phase 3 Meek's transitive propagation.

## 3. Asymptotic & Numerical Analysis
- **Worst-case Complexity**: $\mathcal{O}(d^k)$ conditional independence tests, where $k$ is the maximum degree of the true DAG.
- For sparse graphs ($k \ll d$), the algorithm scales in polynomial time $\mathcal{O}(d^2)$ compared to super-exponential exhaustive DAG search ($\approx \mathcal{O}(2^{d^2})$).

## 4. Edge Cases & Numerical Stability
- In sample covariance matrices, collinearity or high conditioning subset size $|S|$ can induce near-singular submatrices. Handled using Moore-Penrose pseudo-inversion (`np.linalg.pinv`) and numerical correlation clipping $[-0.999999, 0.999999]$.

## 5. Standalone Verification
The test suite validates:
- Generation of synthetic linear Gaussian SCM with ground-truth DAG:
  $X_0 \rightarrow X_2 \leftarrow X_1$ and $X_2 \rightarrow X_3$.
- Elimination of non-edges: identifying $X_0 \perp\!\!\!\perp X_1$ and $X_0 \perp\!\!\!\perp X_3 \mid X_2$.
- Exact recovery of the unshielded collider $X_0 \rightarrow X_2 \leftarrow X_1$.
- Exact orientation of downstream causal edge $X_2 \rightarrow X_3$ via Meek's Rule 1.

## 6. Real-World Applications & Context
Vital for identifying gene regulatory networks from perturbation single-cell genomics, isolating genuine macroeconomic drivers from time-series indicators, root-cause diagnosis in complex distributed microservices, and debiasing recommender systems.

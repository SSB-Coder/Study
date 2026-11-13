# J01: Semi-Supervised Learning & Label Propagation

## 1. Intuition & The "Grand Analogy"
Imagine dropping droplets of red and blue dye onto opposite corners of a damp paper towel. The water molecules on the paper represent thousands of unlabeled data points. You do not need to individually inspect every square millimeter of cellulose to decide which color it belongs to. As the dye diffuses naturally through the connected fibers of the paper (the graph of data affinities), the dye propagates along dense structural clusters, creating natural color boundaries along dry or sparse gaps.

**Semi-Supervised Learning (SSL)** leverages the geometric structure of vast unlabeled datasets $\mathcal{U}$ alongside tiny labeled datasets $\mathcal{L}$ ($|\mathcal{L}| \ll |\mathcal{U}|$). **Label Propagation** (Zhu & Ghahramani, 2002) models data points as nodes on an affinity graph, diffusing known labels through random walk transition matrices.

---

## 2. Architectural & Mathematical Breakdown
Let dataset $\mathcal{D} = \{ (\mathbf{x}_1, y_1), \dots, (\mathbf{x}_l, y_l), \mathbf{x}_{l+1}, \dots, \mathbf{x}_{l+u} \}$.
Construct symmetric affinity matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ ($n = l + u$):
$$W_{ij} = \exp\left( - \frac{\| \mathbf{x}_i - \mathbf{x}_j \|^2}{2 \sigma^2} \right) \quad (i \neq j), \quad W_{ii} = 0$$
Define degree diagonal matrix $D_{ii} = \sum_j W_{ij}$.
The row-stochastic transition probability matrix is $\mathbf{P} = \mathbf{D}^{-1} \mathbf{W}$.

### Block Partitioning
$$\mathbf{P} = \begin{bmatrix} \mathbf{P}_{ll} & \mathbf{P}_{lu} \\ \mathbf{P}_{ul} & \mathbf{P}_{uu} \end{bmatrix}, \quad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_l \\ \mathbf{Y}_u \end{bmatrix}$$
where $\mathbf{Y}_l \in \{0, 1\}^{l \times C}$ is the one-hot labeled ground truth.

### Harmonic Function Closed-Form Solution
At convergence of the random walk, unlabeled probabilities satisfy the Dirichlet problem on graphs:
$$\mathbf{Y}_u = (\mathbf{I} - \mathbf{P}_{uu})^{-1} \mathbf{P}_{ul} \mathbf{Y}_l$$

---

## 3. Hardcore Mathematical Derivations

### Graph Laplacian Energy Minimization
Label propagation minimizes the Dirichlet energy functional:
$$\mathcal{E}(f) \triangleq \frac{1}{2} \sum_{i, j} W_{ij} (f(i) - f(j))^2 = \mathbf{f}^\top \mathbf{L} \mathbf{f}$$
subject to the clamping constraint $f(i) = y_i$ for all $i \in \mathcal{L}$, where $\mathbf{L} = \mathbf{D} - \mathbf{W}$ is the unnormalized Graph Laplacian.
Differentiating with respect to the unlabeled vector $\mathbf{f}_u$:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{f}_u} = 2 (\mathbf{L}_{uu} \mathbf{f}_u + \mathbf{L}_{ul} \mathbf{f}_l) = 0$$
$$\mathbf{L}_{uu} \mathbf{f}_u = - \mathbf{L}_{ul} \mathbf{f}_l$$
Since $\mathbf{L}_{uu}$ is strictly positive definite for connected graphs with at least one labeled node, the solution is unique:
$$\mathbf{f}_u = - \mathbf{L}_{uu}^{-1} \mathbf{L}_{ul} \mathbf{f}_l$$
Labels are smooth: points connected by high-weight edges $W_{ij}$ are penalized heavily for having different label values.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Iterative Label Propagation
Inputs: Affinity matrix W, initial label matrix Y_0 where unlabeled rows are 0
1. Compute transition matrix P = D^(-1) W
2. Loop until convergence:
     Y_{t+1} = P * Y_t
     Clamp labeled nodes: Y_{t+1}[0:l, :] = Y_l
3. Return predicted classes for unlabeled data: y_i = argmax_c Y[i, c]
```

---

## 5. Failure Modes & When It Breaks
1. **Disconnected Components:** If an unlabeled island has no path to any labeled node, transition probabilities never reach it, leaving it unassigned.
2. **Label Imbalance Distortion:** A dominant class with more labeled seeds can wash across the graph, drowning minority clusters.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Semi-Supervised Instruction Tuning:** Propagating human quality ratings across millions of synthetic LLM completions using embedding cosine similarity graphs.
- **Pseudo-Labeling in Self-Training:** Models like Noisy Student propagate teacher predictions onto unlabeled internet text.

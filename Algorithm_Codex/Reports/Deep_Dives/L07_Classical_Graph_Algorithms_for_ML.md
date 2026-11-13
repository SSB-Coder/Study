# L07: Classical Graph Algorithms for ML (Dijkstra, PageRank, Spectral Clustering)

## 1. The Beginner's Analogy
Data points in classical machine learning are often treated as independent vectors floating in empty space. But real-world data is inherently relational: web pages link to each other, molecules connect via chemical bonds, and users interact in social networks.
**Classical Graph Algorithms** reveal the underlying geometry of relations:
- **Dijkstra:** Finds the absolute shortest physical route through a roadmap of weighted roads.
- **PageRank:** Decides which websites are most important by simulating millions of random surfers clicking links across the World Wide Web.
- **Spectral Clustering:** Unravels complex, non-linear geometric shapes (like intertwining rings) by listening to the "vibrational harmonics" (eigenvalues) of the graph's connectivity matrix.

## 2. The Deep Architecture & Mechanics
This module covers three foundational graph algorithmic pillars:
1. **Dijkstra's Shortest Path:** Greedy priority-queue graph traversal relaxing edge weights $d[v] = \min(d[v], d[u] + w(u, v))$.
2. **Google's PageRank:** Power iteration over the stochastic Google transition matrix $G$, balancing link authority with random walk damping.
3. **Normalized Spectral Clustering:** Projects non-linearly separable data into the eigenspace of the Normalized Graph Laplacian $L_{\text{sym}}$, where manifold clusters become linearly separable clusters suitable for $k$-means.

## 3. The Hardcore Mathematics

### Dijkstra Relaxation Invariant
For edge $(u, v)$ with non-negative weight $w(u, v) \ge 0$:
$$d(s, v) \le d(s, u) + w(u, v)$$

### Google PageRank Formulation
Let $M_{ij} = \frac{A_{ji}}{\sum_k A_{ki}}$ be the column-stochastic web adjacency transition matrix.
The Google Matrix $G$ incorporates random damping factor $d \approx 0.85$:
$$G = d M + \frac{1 - d}{N} \mathbf{1} \mathbf{1}^T$$
By the Perron-Frobenius Theorem, $G$ is positive and primitive, guaranteeing a unique stationary probability distribution satisfying:
$$p^* = G p^*, \quad \|p^*\|_1 = 1$$
Solved via Power Iteration: $p^{(t+1)} = G p^{(t)}$.

### Spectral Graph Theory & The Laplacian
Given affinity matrix $W$ with Gaussian kernel $W_{ij} = \exp(-\gamma \|x_i - x_j\|^2)$ and degree matrix $D_{ii} = \sum_j W_{ij}$:
1. **Unnormalized Laplacian:** $L = D - W$.
2. **Symmetric Normalized Laplacian:**
   $$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$$
For any vector $f \in \mathbb{R}^N$:
$$f^T L_{\text{sym}} f = \frac{1}{2} \sum_{i,j=1}^N W_{ij} \left( \frac{f_i}{\sqrt{D_{ii}}} - \frac{f_j}{\sqrt{D_{jj}}} \right)^2 \ge 0$$
Minimizing Rayleigh quotient $f^T L f$ over orthogonal coordinates corresponds to finding the minimal graph cut (RatioCut / Normalized Cut), solved via the $k$ smallest eigenvectors of $L_{\text{sym}}$.

## 4. The Algorithm Step-by-Step
1. **Dijkstra:** Initialize $d[s] = 0, d[v] = \infty$. Repeatedly pop minimal distance node and relax incident edges until all nodes visited.
2. **PageRank:** Construct transition matrix $M$; blend with teleportation matrix; iterate $p^{(t+1)} = G p^{(t)}$ until $\|p^{(t+1)} - p^{(t)}\|_1 < \epsilon$.
3. **Spectral Clustering:**
   - Compute pairwise Gaussian affinity matrix $W$.
   - Compute degree matrix $D$ and normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$.
   - Compute smallest $k$ eigenvectors $U \in \mathbb{R}^{N \times k}$.
   - Normalize rows of $U$ to unit hypersphere $T_{ij} = U_{ij} / \|U_i\|_2$.
   - Cluster rows of $T$ using standard $k$-means.

## 5. Limitations & Failure Modes
- **Dijkstra with Negative Edges:** Fails if graphs contain negative edge cycles (requires Bellman-Ford).
- **Spectral Clustering Eigendecomposition Cost:** Exact full eigendecomposition scales cubically $\mathcal{O}(N^3)$, requiring Lanczos/Nyström approximations for large graphs.

## 6. Where This Fits
Cross-reference: Spectral graph theory and PageRank are the foundational ancestors of Graph Neural Networks (GCNs, GATs). Message-passing layers in modern GNNs generalize normalized Laplacian propagation matrices with learnable weight transformations.

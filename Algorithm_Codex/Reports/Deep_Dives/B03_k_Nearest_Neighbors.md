# Deep Dive: k-Nearest Neighbors & KD-Trees

## 1. The Beginner's Analogy: Asking Your Neighbors
Imagine you move to a new neighborhood and want to decide which internet provider to pick:
- You don't build a complex physical model of the network infrastructure. Instead, you walk outside and ask the **$k=5$ closest houses** to your front door which provider they use.
- You tally their answers: 4 say "FiberNet", 1 says "CableMax". By simple majority vote, you choose FiberNet.
- **The KD-Tree** is an organized real-estate filing system that divides the city into North/South halves, then East/West quarters, then street blocks. Instead of knocking on every single door in the entire city ($O(N)$), you search only your block and adjacent sectors ($O(\log N)$).

---

## 2. The Deep Architecture/Algorithm Breakdown
k-Nearest Neighbors is an instance-based, non-parametric, lazy learning algorithm:
1. **Lazy Evaluation:** No explicit model parameters are estimated during training; training consists merely of storing the dataset $\mathcal{D}$.
2. **Metric Distances:** Measures proximity using Minkowski metrics ($L_1$ Manhattan, $L_2$ Euclidean).
3. **KD-Tree (k-Dimensional Tree):** A space-partitioning binary search tree:
   - Alternates splitting dimensions at each depth level: $\text{axis} = \text{depth} \pmod d$.
   - Splits points at the **median** value of the chosen axis.
   - Enables fast nearest neighbor search via recursive branch-and-bound pruning.
4. **Majority Voting & Distance Weighting:** Predicts class labels via mode voting or inverse-distance weighting $w_i = \frac{1}{d(x, x_i) + \epsilon}$.

---

## 3. The Hardcore Mathematics

### 3.1 Minkowski Distance Metric
Given query vector $x \in \mathbb{R}^d$ and stored exemplar $x_i \in \mathbb{R}^d$:
$$D_p(x, x_i) = \left( \sum_{j=1}^d |x_j - x_{ij}|^p \right)^{1/p}$$
When $p=2$, this is Euclidean distance; when $p=1$, Manhattan distance.

### 3.2 KD-Tree Splitting Invariant
At tree node with depth $h$ and axis $a = h \pmod d$:
Let median coordinate value be $v = \text{median}(\{x_i[a]\}_{i \in \text{node}\})$.
Data is partitioned into left and right subtrees:
$$\text{Left} = \{x_i \mid x_i[a] \le v\}, \quad \text{Right} = \{x_i \mid x_i[a] > v\}$$
Search pruning criterion: If best distance so far $r > |x_q[a] - v|$, the query hypersphere intersects the dividing hyperplane, requiring exploration of both branches; otherwise, the alternate branch is safely pruned.

---

## 4. The Algorithm Step-by-Step
1. **KD-Tree Construction:** Recursively find median along axis $a$, create node, recurse on left and right slices until points per leaf $\le \text{leaf\_size}$.
2. **Querying Nearest Neighbors:**
   - Traverse down tree to locate leaf containing query point $x_q$.
   - Add leaf points to candidate priority queue of size $k$.
   - Backtrack up tree: check if distance from $x_q$ to bounding hyperplane is less than current $k$-th neighbor distance. If yes, explore sibling subtree.
3. **Aggregate Predictions:** Compute mode of top $k$ labels (classification) or mean of target values (regression).

---

## 5. Limitations / When It Breaks
- **The Curse of Dimensionality:** As $d > 20$, the volume of the unit sphere shrinks exponentially relative to the bounding hypercube; all pairwise distances converge to roughly the same distance, and KD-Tree search degenerates to linear scan $O(N)$.
- **Inference Latency:** High memory and computation cost at test time since every prediction depends on historical data points.
- **Historical Fix:** Locality-Sensitive Hashing (LSH), Hierarchical Navigable Small World graphs (HNSW), and Product Quantization.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/39_Vector_Databases_and_HNSW.md`. In modern AI pipelines, Retrieval-Augmented Generation (RAG) uses approximate k-NN across millions of text embeddings to ground LLM generations in factual documents.

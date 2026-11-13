# Deep Dive: W05 — Min-Cut Circuit Partitioning (Kernighan-Lin)

## 1. The Beginner's Analogy: Splitting a Factory Across Two Buildings
Imagine dividing a busy electronic manufacturing factory across two separate buildings:
- Workers inside each building can communicate instantly through open corridors.
- But every time a task requires communication between Building A and Building B, a courier must physically cross the street outside (inter-chip interconnect delay and pin cost).
- **Kernighan-Lin** starts with an arbitrary division of workers and simulates swapping pairs between buildings. It calculates the net reduction in street crossings, locking workers as it tests, and executes the sequence of swaps that minimizes outside courier trips while keeping both buildings equally staffed.

---

## 2. The Deep Architecture/Algorithm Breakdown
Given a circuit netlist represented as graph $G = (V, E)$ with $|V| = 2n$:
1. **Goal:** Partition $V$ into disjoint subsets $A$ and $B$ with $|A| = |B| = n$ minimizing cut size:
   $$\text{Cut}(A, B) = \sum_{u \in A, v \in B} w(u, v)$$
2. **Kernighan-Lin (KL):** An $\mathcal{O}(n^2 \log n)$ iterative improvement heuristic. In each pass, it evaluates all pairs $(a, b) \in A \times B$, greedily locks the pair with maximum gain, and picks the prefix $k$ that maximizes cumulative gain $\sum_{i=1}^k g_i$.
3. **Fiduccia-Mattheyses (FM) Extension:** Generalizes KL to hypergraphs (nets with $>2$ pins) and moves single cells at a time using bucket-list data structures in linear $\mathcal{O}(|E|)$ time per pass.

---

## 3. The Hardcore Mathematics

### 3.1 Node D-Values (Gain Potential)
For node $a \in A$:
$$E(a) = \sum_{v \in B} c(a, v), \quad I(a) = \sum_{u \in A} c(a, u)$$
$$D(a) = E(a) - I(a)$$
$D(a)$ measures the net decrease in cut cost if node $a$ moves from $A$ to $B$.

### 3.2 Pairwise Swap Gain Formula
If we swap $a \in A$ and $b \in B$, the exact reduction in cut size is:
$$g_{ab} = D(a) + D(b) - 2 c(a, b)$$
The $-2 c(a, b)$ term corrects for the edge connecting $a$ and $b$, which changes from an external cut edge to an internal edge.

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Partition nodes into equal halves $A$ and $B$.
2. **Pass Iteration:**
   - Compute $D(u)$ for all nodes. Set $\text{locked} = \emptyset$.
   - For step $i = 1 \dots n$:
     - Find unlocked $a \in A, b \in B$ maximizing $g = D(a) + D(b) - 2 c(a, b)$.
     - Lock $a$ and $b$. Record swap and gain $g_i$.
     - Update $D$ values of remaining unlocked neighbors.
   - Find $k$ maximizing cumulative gain $G_k = \sum_{j=1}^k g_j$.
   - If $G_k > 0$, commit the first $k$ swaps and repeat pass. Else, terminate.

---

## 5. Limitations / When It Breaks
- **Local Optima:** Being an exchange heuristic, KL can get trapped in sub-optimal local basins. Multi-level partitioning (hMETIS) coarsens the netlist through heavy-edge matching, partitions the coarse graph, and uncoarsens with FM refinement.

---

## 6. Where This Fits
- **Top-Level Design Partitioning:** Splitting multi-million gate SoC designs across multiple FPGA chips or chiplets in 2.5D/3D IC packaging.
- **Spectral Graph Partitioning (C02, L07):** Graph Laplacian Fiedler vector methods provide continuous relaxations to this discrete combinatorial problem.

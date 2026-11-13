# Deep Dive: T02 — Phylogenetic Tree Construction (UPGMA & Neighbor-Joining)

## 1. The Beginner's Analogy: Tracing the Tree of Life
Imagine comparing the DNA of four animals: Human, Chimpanzee, Gorilla, and Orangutan:
- Human and Chimp genomes differ by only 1.2%. Gorilla differs by 3%. Orangutan differs by 5%.
- **UPGMA** acts like hierarchical clustering: it merges Human and Chimp into a pair, then merges Gorilla, assuming that all species evolve at the exact same clock speed (the "molecular clock" hypothesis).
- **Neighbor-Joining (NJ)** is smarter: it knows some lineages evolve faster than others (e.g. rodents reproduce 50x faster than whales). NJ uses a mathematical $Q$-matrix that subtracts each species' average divergence rate, preventing fast-evolving lineages from being falsely clustered together.

---

## 2. The Deep Architecture/Algorithm Breakdown
Phylogenetic inference reconstructs the evolutionary history of species from molecular distance matrices:
1. **Distance Matrices:** Pairwise genetic divergence $d(i, j)$ computed using nucleotide substitution models (Jukes-Cantor, Kimura 2-parameter).
2. **UPGMA (Sokal & Michener, 1958):**
   - Assumes **ultrametricity**: constant rate of evolution across all lineages ($d(u, \text{root}) = d(v, \text{root})$).
   - Generates rooted trees where all terminal leaves are equidistant from root.
3. **Neighbor-Joining (Saitou & Nei, 1987):**
   - Assumes **additivity**: $d(i, j) = \sum_{e \in \text{path}(i, j)} w(e)$.
   - Star-decomposition algorithm: starts with a completely unresolved star topology and iteratively isolates pairs of neighbors that minimize total tree length.

---

## 3. The Hardcore Mathematics

### 3.1 UPGMA Cluster Distance Update
When clusters $A$ and $B$ merge into $A \cup B$:
$$d_{(A \cup B), C} = \frac{|A| d_{AC} + |B| d_{BC}}{|A| + |B|}$$

### 3.2 Neighbor-Joining Q-Matrix
For $N$ taxa with distance matrix $D$:
$$r_i = \sum_{k=1}^N D(i, k)$$
$$Q(i, j) = (N - 2) D(i, j) - r_i - r_j$$
The pair $(i, j)$ that minimizes $Q(i, j)$ is guaranteed to be true neighbors in the additive tree!

### 3.3 Limb Length Calculation
Distance from taxon $i$ and $j$ to their common ancestral node $u$:
$$L_{iu} = \frac{1}{2} D(i, j) + \frac{1}{2(N - 2)} (r_i - r_j)$$
$$L_{ju} = D(i, j) - L_{iu}$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Compute distance matrix $D$.
2. **NJ Loop:**
   - Compute row sums $r_i$.
   - Compute $Q$-matrix. Find $(i, j) = \arg\min Q(i, j)$.
   - Create internal node $u$. Compute branch lengths $L_{iu}, L_{ju}$.
   - Update distances from $u$ to all other nodes $k$: $D(u, k) = \frac{1}{2}(D(ik) + D(jk) - D(ij))$.
   - Remove $i, j$ and insert $u$. Repeat until 2 nodes remain.
3. **Connect Final Pair:** Add terminal edge.

---

## 5. Limitations / When It Breaks
- **Long-Branch Attraction (LBA):** When two unrelated lineages both experience exceptionally high mutation rates, distance-based methods mistakenly group them as sister taxa. Maximum Likelihood and Bayesian phylogenetics (MrBayes) model site-specific substitution rates to resolve LBA.

---

## 6. Where This Fits
- **Cross-Reference Hierarchical Clustering (C02):** UPGMA is mathematically identical to average-linkage agglomerative clustering.

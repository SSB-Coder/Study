# K01: Hopfield Networks & Dense Associative Memory

## 1. Intuition & The "Grand Analogy"
Imagine a rubber sheet suspended horizontally. Whenever you press a heavy metal bowling ball into the sheet, it creates a deep conical depression (a memory attractor basin). If you carve three distinct divots into the sheet and then release a tiny marble near the edge of one depression, the marble rolls downhill, automatically settling at the bottom of the nearest basin. Even if you only remember 20% of a childhood photograph (a noisy, corrupted marble position), your brain rolls down the neural energy landscape to reconstitute the complete, crystal-clear memory.

**Hopfield Networks** (Hopfield, 1982) are the quintessential model of **content-addressable associative memory**, operating via Lyapunov energy descent.

---

## 2. Architectural & Mathematical Breakdown
A discrete Hopfield network consists of $N$ binary bipolar neurons $s_i \in \{-1, +1\}$.
The network is recurrently and symmetrically connected:
$$W_{ij} = W_{ji}, \quad W_{ii} = 0$$

### Hebbian Storage Prescription (Outer Product Rule)
To store $M$ orthogonal prototype memory patterns $\mathbf{\xi}^1, \dots, \mathbf{\xi}^M \in \{-1, +1\}^N$:
$$\mathbf{W} = \frac{1}{N} \sum_{\mu=1}^M \mathbf{\xi}^\mu (\mathbf{\xi}^\mu)^\top - \frac{M}{N} \mathbf{I}$$

### Asynchronous Energy Minimization & Retrieval
At each step, select a neuron $i$ and update its state:
$$s_i \leftarrow \text{sgn}\left( \sum_{j=1}^N W_{ij} s_j - \theta_i \right)$$

---

## 3. Hardcore Mathematical Derivations

### The Lyapunov Energy Function
Define the scalar network energy:
$$E(\mathbf{s}) \triangleq -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N W_{ij} s_i s_j = -\frac{1}{2} \mathbf{s}^\top \mathbf{W} \mathbf{s}$$
Consider the change in energy $\Delta E$ when neuron $k$ updates its state from $s_k$ to $s'_k = \text{sgn}(h_k)$, where $h_k = \sum_j W_{kj} s_j$:
$$\Delta E = E(\mathbf{s}') - E(\mathbf{s}) = -\left( s'_k - s_k \right) \sum_{j \neq k} W_{kj} s_j = -\Delta s_k \cdot h_k$$
- If $h_k > 0$, $s'_k = +1$, so $\Delta s_k \ge 0$, meaning $-\Delta s_k \cdot h_k \le 0$.
- If $h_k < 0$, $s'_k = -1$, so $\Delta s_k \le 0$, meaning $-\Delta s_k \cdot h_k \le 0$.
In all cases, $\Delta E \le 0$.
Because the energy is bounded below ($E \ge -\frac{1}{2} \sum_{i, j} |W_{ij}|$), the network is strictly **guaranteed to converge to a stable local minimum (attractor)** in a finite number of steps.

### Memory Capacity Limit
In classical Hebbian Hopfield networks, storing more than $M_{\max} \approx 0.138 N$ patterns causes memory basins to merge, spawning catastrophic "spurious states" that corrupt retrieval.
- **Modern Hopfield Networks (Dense Associative Memory, Krotov & Hopfield 2016):** Replacing quadratic energy with exponential energy yields exponential storage capacity $M \approx 2^{N/2}$.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Hopfield Pattern Storage and Retrieval
1. Storage:
     Given M binary patterns xi_1, ..., xi_M in {-1, 1}^N
     W = 1/N * sum_{mu} xi_mu * xi_mu^T
     Set diagonal W_{ii} = 0
2. Retrieval:
     Given corrupted test query s_0
     Repeat until convergence:
       For each neuron i in random order:
         h_i = sum_j W_{ij} s_j
         s_i = +1 if h_i >= 0 else -1
3. Return converged state vector s
```

---

## 5. Failure Modes & When It Breaks
1. **Spurious Attractors:** Linear combinations and inverted states $-\mathbf{\xi}^\mu$ are also local energy minima.
2. **Correlated Memories:** If stored patterns are non-orthogonal (correlated), the cross-talk interference term destroys retrieval basins.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Transformers ARE Modern Hopfield Networks (Ramsauer et al., 2020):** Softmax self-attention $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$ is mathematically identical to a single-step retrieval update in a continuous Modern Hopfield Network with exponential energy function!

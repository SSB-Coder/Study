# Deep Dive: W02 — Rectilinear Steiner Minimum Tree (RSMT)

## 1. The Beginner's Analogy: Connecting Cities with Junction Hubs
Imagine connecting three towns (A, B, C) with electrical transmission lines:
- **Minimum Spanning Tree (MST)** requires transmission wires to run directly from town to town. You might wire A to B, and B to C.
- **Steiner Tree** asks a smarter question: what if we place a new power substation S in the empty field between them? Wiring A to S, B to S, and C to S uses dramatically less copper wire than connecting the towns directly!
- In microchip design, all wires must run strictly along horizontal and vertical metal layers (Manhattan grid), making this the **Rectilinear** Steiner problem.

---

## 2. The Deep Architecture/Algorithm Breakdown
Given terminal pins $P = \{p_1, \dots, p_n\} \subset \mathbb{R}^2$, the Rectilinear Steiner Minimum Tree (RSMT) seeks an acyclic graph $T = (V, E)$ such that:
1. $P \subseteq V$ (all terminals are connected).
2. All edges are rectilinear segments parallel to coordinate axes.
3. Total wirelength $L(T) = \sum_{e \in E} \text{length}(e)$ is minimized.
Vertices in $V \setminus P$ are called **Steiner points**.

### 2.1 Hanan's Theorem (1966)
Finding the optimal continuous coordinates for Steiner points appears infinite. **Hanan's Theorem** proved a profound simplification:
$$\text{An optimal RSMT always exists whose Steiner points are drawn from the Hanan Grid } H(P)$$
where $H(P) = \{(x_i, y_j) \mid (x_i, y_i) \in P, (x_j, y_j) \in P\}$. For $n$ pins, the Hanan grid has at most $n^2$ candidate points!

---

## 3. The Hardcore Mathematics

### 3.1 Steiner Ratio Bound
Let $L_{\text{RSMT}}(P)$ and $L_{\text{RMST}}(P)$ be the wirelengths of the optimal Steiner tree and optimal Minimum Spanning Tree respectively.
Hwang (1976) proved the legendary Steiner ratio bound:
$$\frac{L_{\text{RMST}}(P)}{L_{\text{RSMT}}(P)} \le \frac{3}{2}$$
This guarantees that an ordinary rectilinear MST is already a $1.5$-approximation to the NP-hard Steiner minimum tree! Adding Hanan Steiner points closes this remaining gap.

---

## 4. The Algorithm Step-by-Step
1. **Compute Base RMST:** Run Prim's or Kruskal's algorithm on pins using $L_1$ metric.
2. **Extract Hanan Grid:** Construct all intersection points $(x_i, y_j)$ from terminal coordinates.
3. **Iterative Greedy Steiner Insertion:**
   - For each candidate $s \in H(P)$:
     - Compute RMST length on $P \cup \{s\}$.
   - If the best candidate reduces total wirelength, add it to terminal set and repeat.
4. **Prune:** Remove degree-1 or degree-2 Steiner points that do not introduce rectilinear branching.

---

## 5. Limitations / When It Breaks
- **NP-Hard Complexity:** Exact RSMT is strongly NP-hard for arbitrary $n$. Industrial routers use the FLUTE (Fast Lookup Table for Efficient Steiner Trees) algorithm for $n \le 9$ and batched heuristics (GeoSteiner) for large nets.
- **Obstacle Avoidance:** Physical blockages (macro blocks, power rails) force Steiner wires to detour around obstacles, requiring Obstacle-Avoiding RSMT (OARSMT).

---

## 6. Where This Fits
RSMT is the global interconnect estimator used in every modern chip placement tool:
- **Wirelength Estimation (W03, W04):** Used to evaluate routing feasibility during floorplanning before detailed routing.
- **Static Timing Analysis (W08):** RC delay models scale quadratically with Steiner wirelength ($R \propto L, C \propto L \implies \tau \propto L^2$).

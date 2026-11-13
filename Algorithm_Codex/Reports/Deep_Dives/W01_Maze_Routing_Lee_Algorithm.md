# Deep Dive: W01 — Maze Routing (Lee's Algorithm)

## 1. The Beginner's Analogy: A Flood on a Circuit Board
Imagine a flooded PCB where two metal pins must be connected by a copper wire without crossing existing surface components:
- You drop a colored dye at Pin A. The dye ripples outward in all directions like a circular wave, filling every free square millimeter of the board.
- The instant the wave crest touches Pin B, you stop the flood: you know for a mathematical certainty that the wave took the shortest possible path!
- To lay down the wire, you simply follow the downhill gradient of the flood timer numbers backward from Pin B to Pin A.

---

## 2. The Deep Architecture/Algorithm Breakdown
Lee's algorithm (1961) is the foundational grid-based routing algorithm in Electronic Design Automation (EDA):
1. **Grid Representation:** The routing area is discretized into a uniform 2D (or multilayer 3D) grid of cells, each labeled as free, obstacle, or pin.
2. **Phase 1: Wave Expansion (BFS):** Starting from source $S$, cells at distance $d=1, 2, 3, \dots$ are sequentially labeled.
3. **Phase 2: Backtracking:** Starting from target $T$, trace back to any adjacent cell with label $d-1$ until reaching $S$.
4. **Phase 3: Cleanup:** Reset wave markers, keeping routed copper cells permanently marked as obstacles for subsequent nets.

---

## 3. The Hardcore Mathematics

### 3.1 Wave Expansion Invariant
On a uniform grid graph $G = (V, E)$ with unit edge weights, BFS maintains distance invariant:
$$d(v) = \min_{u \in N(v)} d(u) + 1$$
Because edges have uniform weight, FIFO queue exploration guarantees that when target $T$ is first dequeued:
$$d(T) = \text{dist}_G(S, T)$$

### 3.2 Manhattan Distance Bound
In obstacle-free continuous space, minimum wirelength between $(x_1, y_1)$ and $(x_2, y_2)$ is bounded by the $L_1$ norm:
$$L_1(S, T) = |x_1 - x_2| + |y_1 - y_2| \le \text{length}(\text{path}_{\text{Lee}})$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Set wavefront queue $Q = [S]$, distance array $\text{dist}[S] = 0$, all other cells $-1$.
2. **Expand:** While $Q$ is not empty:
   - Pop $(r, c)$. If $(r, c) == T$, break.
   - For 4-connected neighbors $(nr, nc)$:
     - If in bounds, unvisited, and unblocked: $\text{dist}[nr, nc] = \text{dist}[r, c] + 1$, push to $Q$.
3. **Backtrack:** From $T$, iteratively move to neighbor with label $\text{dist}-1$ until reaching $S$. Reverse path.

---

## 5. Limitations / When It Breaks
- **Memory & Time Explosion:** For an $N \times N$ grid, Lee's algorithm visits $\mathcal{O}(N^2)$ cells in 2D (and $\mathcal{O}(N^3)$ in multilayer chips), filling entire memory.
- **Modern Fixes:** Hadlock's detour algorithm and Soukup's directional router use A* heuristic estimates ($L_1$ to target) to expand only toward the goal.
- **Net Ordering Dependence:** Routing net 1 may block net 2. Rip-up and reroute algorithms (e.g. Pathfinder) dynamically negotiate congestion.

---

## 6. Where This Fits
Cross-reference with `Algorithm_Codex/L01_Uninformed_Search.md`. Lee's algorithm is literally Breadth-First Search applied to physical electronic design. It remains the gold-standard final-stage maze solver in industrial PCB routers (Altium, KiCad) and VLSI detailed routers.

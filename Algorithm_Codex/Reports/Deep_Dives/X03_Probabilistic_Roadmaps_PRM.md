# Deep Dive: X03 — Probabilistic Roadmaps (PRM)

## 1. The Beginner's Analogy: Building a City Subway Network
Imagine navigating an unfamiliar city:
- **RRT (X01):** Every time you want to walk somewhere new, you hire a pioneer who hacks a completely fresh trail through the jungle with a machete. It works once, but repeating it for every trip is exhausting.
- **PRM (Probabilistic Roadmaps):** Spends time upfront building a permanent underground subway network across the entire city:
  1. *Construction Phase:* Drill subway stations at random safe locations and tunnel tracks between nearby stations.
  2. *Query Phase:* When you need to travel from Point A to Point B, you walk to the nearest subway entrance, ride the rails across town, and exit near your destination in milliseconds!

---

## 2. The Deep Architecture/Algorithm Breakdown
Kavraki et al. (1996) introduced PRM for multi-query path planning in static environments:
1. **Two-Phase Architecture:**
   - **Learning Phase (Offline):** Samples $N$ points in free configuration space $\mathcal{C}_{\text{free}}$, connects each point to its $k$-nearest neighbors if line-of-sight is collision-free, building an undirected roadmap graph $G = (V, E)$.
   - **Query Phase (Online):** Connects start $q_{\text{start}}$ and goal $q_{\text{goal}}$ to the roadmap, then executes Dijkstra or A* graph search in $\mathcal{O}(|E| + |V| \log |V|)$ time.
2. **Contrast with RRT:**
   | Dimension | RRT / RRT* | PRM |
   | :--- | :--- | :--- |
   | Paradigm | Tree growth | Roadmap graph |
   | Query Type | Single-query | Multi-query |
   | Environment | Dynamic / changing | Static factory / room |
   | Offline Time | None | High (graph construction) |
   | Online Time | High (full tree search) | Microseconds (A* on graph) |

---

## 3. The Hardcore Mathematics

### 3.1 Connection Condition
For vertex $v \in V$, candidates $u \in k\text{-NN}(v)$ are connected by edge $e = (u, v)$ if:
$$(1 - \alpha) u + \alpha v \in \mathcal{C}_{\text{free}} \quad \forall \alpha \in [0, 1]$$

### 3.2 Graph Shortest Path
$$d(v) = \min_{u \in N(v)} \left( d(u) + \|u - v\|_2 \right)$$

---

## 4. The Algorithm Step-by-Step
1. **Sample:** Uniformly sample points until $N$ collision-free nodes are collected.
2. **K-NN Connect:** For each node $u$, find $k$ nearest neighbors. Perform collision checks along edges; add valid edges to adjacency list.
3. **Query:** Given $(q_{\text{start}}, q_{\text{goal}})$, connect both to roadmap. Run Dijkstra to return path.

---

## 5. Limitations / When It Breaks
- **Dynamic Obstacles:** If an obstacle moves, large sections of the precomputed roadmap are invalidated, requiring dynamic edge invalidation (e.g. Lazy PRM).

---

## 6. Where This Fits
- **Industrial Robotic Arms (e.g. welding, pick-and-place):** Factory robots perform millions of motions in the same unchanging cage; PRM precomputes the roadmap once, serving instant motion plans forever.

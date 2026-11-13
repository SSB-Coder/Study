# Deep Dive: X01 — Rapidly-Exploring Random Trees (RRT)

## 1. The Beginner's Analogy: Tree Roots Seeking Water Around Rocks
Imagine planting a tree root at Point A that needs to reach groundwater at Point B around buried granite boulders:
- A classical grid search (like A*) places a stiff rectangular metal grid over the entire ground and checks thousands of tiny squares. In large 3D spaces, this grid consumes all available memory.
- **RRT** acts like an aggressive plant root: it throws a random dart into empty underground space, finds the root tip closest to that dart, and sprouts a single fresh wooden branch toward it.
- Because large open voids have the largest volume, random darts fall into empty spaces far more often than cramped areas, causing the tree to rapidly reach across empty space and navigate around granite obstacles without ever needing a grid.

---

## 2. The Deep Architecture/Algorithm Breakdown
Introduced by Steven LaValle (1998), Rapidly-Exploring Random Trees (RRT) solved high-dimensional robotic motion planning:
1. **Continuous Configuration Space (C-Space):** Robots operate in continuous state spaces $\mathcal{C} = \mathcal{C}_{\text{free}} \cup \mathcal{C}_{\text{obs}}$.
2. **Voronoi Bias Invariant:** The probability that a tree node is selected for extension is proportional to the volume of its **Voronoi cell**. Peripheral nodes facing unexplored empty voids have massive Voronoi regions, driving rapid exploration of the state space.
3. **Core Operations:**
   - `Sample`: Draw $q_{\text{rand}} \sim \mathcal{U}(\mathcal{C})$. With probability $p_{\text{goal}}$, set $q_{\text{rand}} = q_{\text{goal}}$.
   - `Nearest`: Find $q_{\text{near}} = \arg\min_{v \in V} \|v - q_{\text{rand}}\|$.
   - `Steer`: Advance distance $\Delta q$ from $q_{\text{near}}$ toward $q_{\text{rand}}$.
   - `CollisionCheck`: Verify segment $[q_{\text{near}}, q_{\text{new}}] \subset \mathcal{C}_{\text{free}}$.

---

## 3. The Hardcore Mathematics

### 3.1 Steer Formula
$$q_{\text{new}} = q_{\text{near}} + \min\left(1, \frac{\Delta q}{\|q_{\text{rand}} - q_{\text{near}}\|}\right) (q_{\text{rand}} - q_{\text{near}})$$

### 3.2 Probabilistic Completeness
RRT is **probabilistically complete**: if a collision-free path of clearance $\delta > 0$ exists:
$$\lim_{k \to \infty} P(\text{Path Found at step } k) = 1$$
However, standard RRT is **not asymptotically optimal**: as $k \to \infty$, the path cost almost surely does not converge to the shortest path, motivating RRT* (X02).

---

## 4. The Algorithm Step-by-Step
1. **Init:** Tree $T = \{q_{\text{start}}\}$.
2. **Loop $k = 1, \dots, K$:**
   - Sample $q_{\text{rand}}$ with goal-bias.
   - Find nearest $q_{\text{near}} \in T$.
   - Steer $q_{\text{new}} = \text{steer}(q_{\text{near}}, q_{\text{rand}}, \Delta q)$.
   - If collision-free, add $q_{\text{new}}$ to $T$ with parent $q_{\text{near}}$.
   - If $\|q_{\text{new}} - q_{\text{goal}}\| < \Delta q$, connect to goal and terminate.

---

## 5. Limitations / When It Breaks
- **Narrow Passages (Bug Traps):** If a robot must slip through a needle-thin doorway, the Voronoi cell of the doorway is microscopically small, causing RRT to spend millions of iterations banging fruitlessly against the surrounding walls.

---

## 6. Where This Fits
- **Cross-Reference A* (L02):** A* operates on discrete graph grids; RRT operates directly on continuous continuous configuration manifolds.

# Deep Dive: X02 — RRT* (Asymptotically Optimal Motion Planning)

## 1. The Beginner's Analogy: Trimming the Jagged Forest Path
Imagine hiking through a dense forest along a path laid down by standard RRT:
- Standard RRT found a route, but it zig-zags frantically: it walks 10 yards north, 8 yards east, 7 yards south, like a drunken explorer.
- **RRT\*** introduces two disciplined forest rangers:
  1. **Choose Best Parent:** Before hammering in a new tent spike, the ranger looks at all nearby established tents within 50 feet and connects to the one with the lowest total travel time from home camp.
  2. **The Rewiring Pass:** Whenever a new tent is pitched, the ranger inspects every existing nearby tent. If routing through this brand-new tent cuts 20 minutes off their journey, their old paths are chopped down and rewired through the new tent!
- Over time, all jagged zig-zags are pulled straight into smooth, taut, optimal highways.

---

## 2. The Deep Architecture/Algorithm Breakdown
Karaman and Frazzoli (2011) proved that standard RRT converges to sub-optimal paths with probability 1, and introduced RRT*:
1. **Asymptotic Optimality:** The cost of the returned path converges almost surely to the global optimum as the number of samples $n \to \infty$:
   $$P\left( \lim_{n \to \infty} c(T_n) = c^* \right) = 1$$
2. **Choose Parent:** Evaluates candidate parents within radius $r(n) = \gamma \left( \frac{\log n}{n} \right)^{1/d}$ to minimize cumulative cost:
   $$c(q_{\text{new}}) = \min_{q \in X_{\text{near}}} \left( c(q) + \text{dist}(q, q_{\text{new}}) \right)$$
3. **Rewire:** For all $q_{\text{near}} \in X_{\text{near}}$, if $c(q_{\text{new}}) + \text{dist}(q_{\text{new}}, q_{\text{near}}) < c(q_{\text{near}})$, reassign $q_{\text{near}}$'s parent to $q_{\text{new}}$.

---

## 3. The Hardcore Mathematics

### 3.1 Near Radius Scaling Law
To preserve asymptotic optimality while keeping computational complexity bounded to $\mathcal{O}(n \log n)$, the neighbor search ball radius must scale as:
$$r_n = \gamma_{\text{RRT}^*} \left( \frac{\log n}{n} \right)^{1/d}, \quad \gamma_{\text{RRT}^*} > 2 \left( 1 + \frac{1}{d} \right)^{1/d} \left( \frac{\mu(\mathcal{C}_{\text{free}})}{\zeta_d} \right)^{1/d}$$
where $d$ is dimension and $\zeta_d$ is the volume of the unit sphere.

---

## 4. The Algorithm Step-by-Step
1. Sample $q_{\text{rand}}$ and steer to $q_{\text{new}}$.
2. Find $X_{\text{near}}$ within radius $r_n$.
3. Choose $q_{\min} = \arg\min_{q \in X_{\text{near}}} (c(q) + \text{dist}(q, q_{\text{new}}))$.
4. Insert $q_{\text{new}}$ with parent $q_{\min}$.
5. For each $q_{\text{near}} \in X_{\text{near}}$: if $c(q_{\text{new}}) + \text{dist}(q_{\text{new}}, q_{\text{near}}) < c(q_{\text{near}})$, rewire parent.

---

## 5. Limitations / When It Breaks
- **CPU Overhead:** Rewiring and nearest-neighbor radius queries require $\mathcal{O}(n \log n)$ KD-tree lookups, making RRT* substantially slower per iteration than plain RRT on low-compute microcontrollers.

---

## 6. Where This Fits
- **Self-Driving Cars & Quadrotors:** RRT* is the core path planner used in production autonomous navigation stacks (ROS 2 Nav2).

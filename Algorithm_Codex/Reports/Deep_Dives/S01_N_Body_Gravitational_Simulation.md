# Deep Dive: S01 — N-Body Gravitational Simulation (Barnes-Hut)

## 1. The Beginner's Analogy: Looking at the Milky Way from Earth
Imagine looking up at the night sky at the Andromeda galaxy containing 1 trillion stars:
- To calculate the gravitational pull on Earth, a brute-force computer would calculate the pull from every single star individually ($10^{12}$ calculations).
- But because Andromeda is 2.5 million light years away, all 1 trillion stars cluster into a tiny glowing dot on the horizon!
- **The Barnes-Hut Algorithm** notices this exact physical truth: if a group of stars is far enough away, you can pretend the entire cluster is just one single giant star sitting at its center of mass.
- This recursive spatial tree (quadtree/octree) slashes computation from an impossible $\mathcal{O}(N^2)$ to a blazing-fast $\mathcal{O}(N \log N)$.

---

## 2. The Deep Architecture/Algorithm Breakdown
N-Body simulation solves the gravitational equations of motion for $N$ celestial bodies:
1. **Newton's Gravitational Law:**
   $$\mathbf{F}_i = \sum_{j \ne i} \frac{G m_i m_j}{\|\mathbf{r}_{ij}\|^2 + \epsilon^2} \frac{\mathbf{r}_{ij}}{\|\mathbf{r}_{ij}\|}$$
   where $\epsilon$ is the gravitational softening length preventing infinite singularities.
2. **The Barnes-Hut Tree:**
   - Discretizes spatial domain into a hierarchical **Quadtree** (2D) or **Octree** (3D).
   - Each internal tree cell records the total mass $M$ and Center of Mass $\mathbf{R}_{\text{CoM}}$ of all particles within its sub-tree.
3. **Multipole Acceptance Criterion (MAC):**
   A tree node of side length $s$ at distance $d$ from particle $i$ is approximated as a single point mass if:
   $$\theta = \frac{s}{d} < \theta_{\text{crit}} \approx 0.5$$

---

## 3. The Hardcore Mathematics

### 3.1 Center of Mass Assembly
For tree cell containing particles $\{p_1, \dots, p_k\}$:
$$M = \sum_{i=1}^k m_i, \quad \mathbf{R}_{\text{CoM}} = \frac{1}{M} \sum_{i=1}^k m_i \mathbf{r}_i$$

### 3.2 Symplectic Leapfrog / Velocity Verlet Integrator
To ensure bounded energy drift over millions of orbital revolutions:
$$\begin{aligned}
\mathbf{v}\left(t + \frac{\Delta t}{2}\right) &= \mathbf{v}(t) + \frac{1}{2} \mathbf{a}(t) \Delta t \\
\mathbf{r}(t + \Delta t) &= \mathbf{r}(t) + \mathbf{v}\left(t + \frac{\Delta t}{2}\right) \Delta t \\
\mathbf{v}(t + \Delta t) &= \mathbf{v}\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \mathbf{a}(t + \Delta t) \Delta t
\end{aligned}$$
Because Velocity Verlet is **symplectic** (phase-space volume preserving), the simulation tracks a shadow Hamiltonian with bounded, non-secular energy oscillations.

---

## 4. The Algorithm Step-by-Step
1. **Build Tree:** Insert all $N$ bodies into spatial quadtree, updating mass and center of mass recursively.
2. **Compute Forces:** For each particle, traverse tree. If cell satisfies $s/d < \theta_{\text{crit}}$, compute single force from $(M, \mathbf{R}_{\text{CoM}})$; otherwise recurse into children.
3. **Integrate:** Advance positions and velocities via symplectic Leapfrog.

---

## 5. Limitations / When It Breaks
- **Close Binary Encounters:** When two stars enter tight binary orbits, softening length $\epsilon$ introduces artificial damping, requiring adaptive time-stepping (Hermite integrators).

---

## 6. Where This Fits
- **Cross-Reference Space-Partitioning Trees (M03):** Barnes-Hut reuses the exact same quadtree/octree data structures implemented in M03 for k-d trees, applied to gravitational physics.

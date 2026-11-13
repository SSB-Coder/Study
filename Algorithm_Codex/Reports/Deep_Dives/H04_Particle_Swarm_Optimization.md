# H05: Particle Swarm Optimization (PSO)

## 1. Intuition & The "Grand Analogy"
Imagine a flock of birds flying over a murky wetland searching for scattered seeds. No single bird knows where the largest cache of seeds is hidden. However, every bird remembers the richest spot it personally found so far (cognitive memory), and each bird can chirp loudly to communicate the location of the best food spot discovered by any bird in the entire flock (social communication). Each bird adjusts its flight trajectory by balancing three impulses: momentum to keep flying forward, a gentle tug back toward its personal best, and an urgent pull toward the flock's global best.

**Particle Swarm Optimization (PSO)** (Kennedy & Eberhart, 1995) models this collective intelligence, enabling swarms of particles to navigate complex multidimensional search spaces.

---

## 2. Architectural & Mathematical Breakdown
A swarm consists of $N$ particles with position $\mathbf{x}_i \in \mathbb{R}^d$ and velocity $\mathbf{v}_i \in \mathbb{R}^d$.
- $\mathbf{p}_i$: Personal best position achieved by particle $i$.
- $\mathbf{g}$: Global best position achieved by the entire swarm.

### Velocity & Position Update Equations
$$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 \odot (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 \odot (\mathbf{g} - \mathbf{x}_i^{(t)})$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
where:
- $w \in [0.4, 0.9]$: Inertia weight (preserves previous trajectory).
- $c_1$: Cognitive acceleration coefficient (self-confidence).
- $c_2$: Social acceleration coefficient (group confidence).
- $r_1, r_2 \sim \text{Uniform}(0, 1)$: Independent stochastic coordinate vectors.

---

## 3. Hardcore Mathematical Derivations

### Swarm Stability & Convergence Analysis (Clerc & Kennedy, 2002)
Treating the 1D deterministic difference equation without stochasticity:
$$v_{t+1} - (w - \phi) v_t + w v_{t-1} = 0, \quad \text{where } \phi = c_1 + c_2$$
The characteristic roots of the recurrence relation are:
$$\lambda_{1, 2} = \frac{1 + w - \phi \pm \sqrt{(1 + w - \phi)^2 - 4w}}{2}$$
For the swarm to converge stably without oscillating into infinity, the eigenvalues must lie strictly within the complex unit circle $|\lambda| < 1$:
$$w < 1, \quad \phi > 0, \quad 2w - \phi + 2 > 0$$
- **Constriction Factor $\chi$:** Setting $w = \chi$ and $c_i = \chi \phi_i$ with $\phi = \phi_1 + \phi_2 > 4$:
  $$\chi = \frac{2}{\left| 2 - \phi - \sqrt{\phi^2 - 4\phi} \right|}, \quad \text{for } \phi = 4.1 \implies \chi \approx 0.7298$$
  guarantees convergence without manual velocity clamping.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Canonical Particle Swarm Optimization
Inputs: Swarm size N, inertia w, cognitive c1, social c2, bounds [x_min, x_max]
1. For each particle i = 1 to N:
     Initialize position x_i uniformly within bounds
     Initialize velocity v_i = 0
     p_i = x_i
2. g = argmin_{p_i} f(p_i)
3. For iteration = 1 to MaxIter:
     For each particle i:
       r1, r2 = uniform(0, 1)
       v_i = w * v_i + c1 * r1 * (p_i - x_i) + c2 * r2 * (g - x_i)
       v_i = clip(v_i, -v_max, v_max)
       x_i = x_i + v_i
       x_i = clip(x_i, x_min, x_max)
       
       If f(x_i) < f(p_i):
          p_i = x_i
          If f(p_i) < f(g):
             g = p_i
4. Return global best position g and value f(g)
```

---

## 5. Failure Modes & When It Breaks
1. **Velocity Explosion:** If inertia $w \ge 1$ or acceleration coefficients are too large, velocities accelerate exponentially, blowing particles past bounds.
2. **Topological Clumping:** A star topology (global best broadcast to all particles) causes swift collapse when the global best lands in an attractive local trap. Ring topologies (local neighborhoods) provide superior multi-modal exploration.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Swarm Intelligence for Multi-Agent Systems:** Coordinating teams of autonomous LLM subagents, where individual agent drafts (cognitive) are blended with consensus shared memory (social).
- **Neural Network Pruning:** Optimizing binary mask vectors across transformer attention heads.

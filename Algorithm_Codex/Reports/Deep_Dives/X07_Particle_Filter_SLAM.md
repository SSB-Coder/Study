# Deep Dive: X07 — Particle Filter SLAM (FastSLAM)

## 1. The Beginner's Analogy: Multiple Parallel Universes
Imagine wandering an unknown maze:
- Instead of keeping one single uncertain estimate of your position, you clone yourself into 30 parallel universes (particles).
- In Universe 1, you slipped slightly to the left. In Universe 2, you took a long stride to the right.
- Crucially, **inside each universe, the robot assumes its footsteps were 100% true!**
- Because each universe treats its own footsteps as gospel truth, every landmark it spots can be tracked with a tiny, private $2 \times 2$ mini-filter completely independent of all other landmarks!
- When a universe observes a landmark exactly where its private map predicted, that universe is awarded high probability and cloned; universes that hallucinated nonsense die off.

---

## 2. The Deep Architecture/Algorithm Breakdown
Montemerlo et al. (2002) formulated FastSLAM using **Rao-Blackwellized Particle Filtering (RBPF)**:
1. **Factorization Property:** The joint SLAM posterior factors analytically:
   $$p(x_{1:t}, m_{1:M} \mid z_{1:t}, u_{1:t}) = p(x_{1:t} \mid z_{1:t}, u_{1:t}) \prod_{i=1}^M p(m_i \mid x_{1:t}, z_{1:t})$$
   - Conditioned on the true robot path $x_{1:t}$, **all landmark positions are mutually independent**!
2. **Computational Leap:**
   - EKF-SLAM: $\mathcal{O}(M^2)$ operations on an intractable joint $(3+2M) \times (3+2M)$ covariance.
   - FastSLAM: $K$ particles, each holding $M$ independent $2 \times 2$ Kalman filters. Total time scales as $\mathcal{O}(K \log M)$ with KD-trees, scaling to $>100,000$ landmarks!

---

## 3. The Hardcore Mathematics

### 3.1 Importance Weight Factor
When landmark $i$ is observed, particle $k$ has importance weight:
$$w_t^{[k]} \propto w_{t-1}^{[k]} \cdot p(z_t \mid x_{1:t}^{[k]}, z_{1:t-1}) = w_{t-1}^{[k]} \cdot \frac{1}{\sqrt{|2\pi S_{i, t}^{[k]}|}} \exp\left( -\frac{1}{2} (z_t - \hat{z}_i^{[k]})^T (S_{i, t}^{[k]})^{-1} (z_t - \hat{z}_i^{[k]}) \right)$$

### 3.2 Effective Sample Size ($N_{\text{eff}}$)
To monitor particle degeneracy:
$$N_{\text{eff}} = \frac{1}{\sum_{k=1}^K (w^{[k]})^2}$$
Resample whenever $N_{\text{eff}} < K / 2$.

---

## 4. The Algorithm Step-by-Step
1. **Motion Sample:** For each particle, draw $x_t^{[k]} \sim p(x_t \mid x_{t-1}^{[k]}, u_t)$.
2. **Landmark EKF Update:** For observed landmark, update the particle's private $2 \times 2$ EKF.
3. **Weight:** Multiply particle weight by measurement likelihood.
4. **Resample:** Systematic low-variance resampling duplicates high-weight particles and discards weak ones.

---

## 5. Limitations / When It Breaks
- **Particle Depletion on Long Loops:** In massive closed loops, resampling discards ancestral history, causing particle diversity to collapse to a single trajectory before loop closure is detected.

---

## 6. Where This Fits
- **Cross-Reference Particle Filters (E06):** Directly applies the SIR particle filter to the simultaneous localization and mapping domain.

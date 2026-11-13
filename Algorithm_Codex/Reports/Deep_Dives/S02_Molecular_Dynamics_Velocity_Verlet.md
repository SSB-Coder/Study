# Deep Dive: S02 — Molecular Dynamics (Velocity Verlet & Lennard-Jones)

## 1. The Beginner's Analogy: Billiard Balls with Magnetic Skins
Imagine watching liquid Argon atoms sloshing inside a microscopic glass container:
- From far away, atoms feel a gentle, welcoming gravitational/van der Waals attraction pulling them together.
- But if two atoms get too close and their electron clouds touch, they violently push away with intense repulsive force.
- **The Lennard-Jones Potential** mathematically balances these two forces: a gentle $1/r^6$ attraction from afar, and an impassable $1/r^{12}$ brick-wall repulsion up close.
- **Velocity Verlet** marches time forward in microscopic femtosecond steps, guaranteeing that the total thermal energy of the flask remains strictly conserved.

---

## 2. The Deep Architecture/Algorithm Breakdown
Molecular Dynamics (MD) simulates the physical movements of atoms and molecules:
1. **The Lennard-Jones (12-6) Potential:**
   $$V_{\text{LJ}}(r) = 4\epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^6 \right]$$
   - $\epsilon$: Depth of the potential well (bond strength).
   - $\sigma$: Distance at which inter-particle potential is zero (atomic diameter).
   - Equilibrium separation occurs at $r_{\min} = 2^{1/6} \sigma \approx 1.122 \sigma$.
2. **The Microcanonical (NVE) Ensemble:** Number of particles $N$, Volume $V$, and Total Energy $E$ are strictly constant.
3. **Cross-Reference Langevin Dynamics (P02):**
   - S02's Molecular Dynamics is a **purely deterministic, energy-conserving Hamiltonian integrator**.
   - Module P02's Langevin Dynamics adds friction $-\gamma \mathbf{v}$ and random thermal kicks $\sqrt{2 k_B T \gamma} \mathbf{R}(t)$ to sample the **canonical (NVT) ensemble** at constant temperature. Langevin is the stochastic, thermostatted cousin of Velocity Verlet!

---

## 3. The Hardcore Mathematics

### 3.1 Pairwise Inter-Atomic Force
$$\mathbf{F}_{ij} = -\nabla V(r_{ij}) = 24\epsilon \left[ 2 \left( \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^6 \right] \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}^2}$$

### 3.2 Symplectic Phase-Space Preservation
Velocity Verlet is time-reversible:
$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t) \Delta t + \frac{1}{2} \mathbf{a}(t) \Delta t^2$$
$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2} [\mathbf{a}(t) + \mathbf{a}(t + \Delta t)] \Delta t$$
The Jacobian of the $(r, v) \to (r', v')$ mapping has unit determinant:
$$\det \frac{\partial (r', v')}{\partial (r, v)} = 1$$
satisfying Liouville's theorem of phase-space volume conservation.

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Generate lattice or random positions. Initialize Maxwell-Boltzmann velocities. Remove net momentum $\sum m_i \mathbf{v}_i = 0$.
2. **Half-Step:** Update positions using current velocities and forces. Apply periodic boundary wrap.
3. **Forces:** Evaluate pairwise Lennard-Jones forces with minimum image convention.
4. **Final Step:** Update velocities with average of old and new forces. Compute kinetic and potential energies.

---

## 5. Limitations / When It Breaks
- **Femtosecond Time Steps:** Because atomic vibrations occur at $10^{14}$ Hz, time step $\Delta t$ must be $\le 1-2$ femtoseconds. Simulating 1 second of real protein folding requires $10^{15}$ steps!

---

## 6. Where This Fits
- **Drug Discovery & AlphaFold:** Force-field molecular dynamics validates and refines protein structures predicted by machine learning models.

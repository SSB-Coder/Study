# Deep Dive: W03 — Force-Directed & Simulated-Annealing Component Placement

## 1. The Beginner's Analogy: Untangling Marionette Strings
Imagine eight puppets on a flat board connected to each other by rubber bands:
- If you scatter them randomly across the room, the rubber bands are stretched taut and criss-cross each other in a catastrophic tangle.
- **Force-Directed Relaxation** lets the rubber bands pull connected puppets toward each other, while tiny repelling magnets push puppets apart so they don't pile up on top of one another.
- **Simulated Annealing** shakes the board vigorously at high heat, gradually cooling it to a gentle vibration, allowing puppets to jump out of awkward corners and settle into the tightest, lowest-wirelength arrangement possible.

---

## 2. The Deep Architecture/Algorithm Breakdown
Macro and standard-cell placement is a primary stage in Physical Design Automation:
1. **Objective Metric:** Half-Perimeter Wirelength (HPWL) serves as the ubiquitous, ultra-fast proxy for rectilinear Steiner wirelength.
2. **Classical Methods:**
   - *Force-Directed / Analytical Placement:* Formulates placement as a quadratic programming problem ($L \propto \sum w_{ij} \|x_i - x_j\|^2$), solved via linear systems (V02) and density smoothing.
   - *Simulated Annealing:* Probabilistic hill-climbing that avoids getting trapped in local minima.
3. **The RL Frontier (AlphaChip):** In 2020-2021, Google published AlphaChip (Nature 2021), framing floorplanning and macro placement as a Reinforcement Learning game. AlphaChip places macros sequentially using a graph neural network policy trained via PPO (G08), beating human engineers and classical placers in wirelength, timing, and power on production TPU chips.

---

## 3. The Hardcore Mathematics

### 3.1 Half-Perimeter Wirelength (HPWL)
For a net $e$ containing pins $\{p_i = (x_i, y_i) \mid i \in e\}$:
$$\text{HPWL}(e) = \left( \max_{i \in e} x_i - \min_{i \in e} x_i \right) + \left( \max_{i \in e} y_i - \min_{i \in e} y_i \right)$$
Total placement wirelength:
$$\Phi_{\text{HPWL}} = \sum_{e \in \mathcal{E}} w_e \text{HPWL}(e)$$

### 3.2 Spring Force Equations
Hooke's law attractive force on component $i$ from net peer $j$:
$$\mathbf{F}_{\text{attr}}(i, j) = k_{ij} (\mathbf{x}_j - \mathbf{x}_i)$$
Coulomb repulsion from overlapping component $k$:
$$\mathbf{F}_{\text{rep}}(i, k) = \frac{c}{\|\mathbf{x}_i - \mathbf{x}_k\|^2} \frac{\mathbf{x}_i - \mathbf{x}_k}{\|\mathbf{x}_i - \mathbf{x}_k\|}$$

---

## 4. The Algorithm Step-by-Step
1. **Global Force Relaxation:** Solve $\sum \mathbf{F}_{\text{attr}} + \mathbf{F}_{\text{rep}} = 0$ iteratively.
2. **Legalization:** Remove remaining cell overlaps by snapping components to discrete chip row sites.
3. **Detailed Placement Refinement:** Run local simulated annealing swaps between adjacent components to minimize final HPWL.

---

## 5. Limitations / When It Breaks
- **Scalability to Billions of Gates:** Modern microprocessors have $>50$ billion transistors. Flat simulated annealing cannot scale beyond $\sim 10^4$ blocks. Hierarchical multi-level clustering (e.g. hMETIS) and electrostatics-based analytical placement (e.g. ePlace, DREAMPlace) are required.

---

## 6. Where This Fits
- **Cross-reference H06 (Simulated Annealing):** W03 is the canonical industrial physical application of classical simulated annealing.
- **Cross-reference AlphaChip & G08 (PPO):** Be clear that classical force-directed/SA placement is the 40-year foundation that RL-based placers were built to outperform.

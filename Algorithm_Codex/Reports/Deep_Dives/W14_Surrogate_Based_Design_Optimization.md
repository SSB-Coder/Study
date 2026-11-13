# Deep Dive: W14 — Surrogate-Based Design Optimization

## 1. The Beginner's Analogy: Guessing the Shape of an Underwater Reef
Imagine mapping an invisible underwater shipwreck using an expensive sonar probe that costs $10,000 per ping:
- You cannot ping every square meter (a brute-force grid would bankrupt you).
- **The Surrogate Model (Gaussian Process)** sketches a smooth mathematical blanket through the few sonar pings you have taken, accompanied by a translucent "cloud of uncertainty" that bulges tall in areas you haven't pinged yet.
- **Expected Improvement (EI)** balances two competing instincts:
  1. *Exploitation:* Ping deeper into the shallow spot that already looks like the treasure chest.
  2. *Exploration:* Ping the giant uncertainty cloud because an even higher reef might be hiding there.
- This strategy discovers the absolute peak in the fewest possible expensive pings.

---

## 2. The Deep Architecture/Algorithm Breakdown
In modern engineering design (aerodynamic shape optimization, internal combustion engine calibration, PCB antenna tuning), numerical simulations (3D CFD, FEA, full-wave electromagnetic solvers) take hours or days per run:
1. **The Surrogate Concept:** Replace the intractable black-box simulator $f(\mathbf{x})$ with a computationally cheap emulator $\hat{f}(\mathbf{x})$ (Gaussian Process / Kriging or Radial Basis Functions).
2. **Initial Design of Experiments (DoE):** Space-filling Latin Hypercube Sampling (LHS) evaluates a small initial set of design points.
3. **Active Learning Loop:**
   - Fit surrogate model to all historical evaluations.
   - Maximize an **Acquisition Function** (Expected Improvement, Upper Confidence Bound) that navigates the exploration-exploitation trade-off.
   - Execute the real simulator only at the single point suggested by the acquisition function.

---

## 3. The Hardcore Mathematics

### 3.1 Gaussian Process Posterior
Given observations $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, prior $f(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))$:
$$\mu(\mathbf{x}) = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$$
$$\sigma^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

### 3.2 Expected Improvement (EI) Closed Form
For minimization where $y^* = \min_i y_i$ is current best value:
$$I(\mathbf{x}) = \max(0, y^* - f(\mathbf{x}))$$
$$\text{EI}(\mathbf{x}) = \mathbb{E}[I(\mathbf{x})] = (y^* - \mu(\mathbf{x})) \Phi\left( \frac{y^* - \mu(\mathbf{x})}{\sigma(\mathbf{x})} \right) + \sigma(\mathbf{x}) \phi\left( \frac{y^* - \mu(\mathbf{x})}{\sigma(\mathbf{x})} \right)$$
where $\Phi(\cdot)$ is the standard normal cumulative distribution function and $\phi(\cdot)$ is the probability density function.

---

## 4. The Algorithm Step-by-Step
1. **Design of Experiments:** Sample initial $N_0$ points across bounded space using Latin Hypercube Sampling.
2. **Evaluate Simulator:** Compute high-fidelity scores $y_i = f(\mathbf{x}_i)$.
3. **Fit Surrogate:** Train Gaussian Process hyperparameters via Maximum Likelihood Estimation.
4. **Optimize Acquisition:** Maximize $\text{EI}(\mathbf{x})$ across parameter domain.
5. **Update:** Query real simulator at $\mathbf{x}^* = \arg\max \text{EI}(\mathbf{x})$. Append to dataset. Repeat until compute budget is exhausted.

---

## 5. Limitations / When It Breaks
- **Dimensionality Bottleneck:** Classical Gaussian Processes struggle past $d > 20$ dimensions because covariance matrix inversion scales as $\mathcal{O}(N^3)$ and Euclidean distance degrades in high dimensions (curse of dimensionality). High-dimensional Bayesian optimization uses Tree-structured Parzen Estimators (TPE) or neural network surrogates.

---

## 6. Where This Fits
- **Cross-Reference Bayesian Optimization (E09) & NSGA-II (H09):** W14 is the direct practical application of E09 and H09 to expensive engineering hardware simulations.

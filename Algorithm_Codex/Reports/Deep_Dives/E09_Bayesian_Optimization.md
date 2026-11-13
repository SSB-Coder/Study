# Deep Dive: Bayesian Optimization (Gaussian Process Surrogates & Acquisition Functions)

## 1. The Beginner's Analogy: Drilling for Oil with Limited Dynamite
Imagine drilling for oil across a vast desert:
- You have only enough funding to drill 10 exploratory wells.
- Each drilling operation takes 3 weeks and costs $1,000,000 (an expensive black-box objective).
- You cannot afford brute-force grid search.
- **Bayesian Optimization** acts as your chief geologist:
  - **Surrogate Model (Gaussian Process):** Based on the first 3 drilled wells, the GP constructs a smooth map predicting oil concentration across the entire desert, complete with exact uncertainty bars.
  - **Acquisition Function:** Recommends the exact coordinates for the next drilling operation by balancing **Exploitation** (drilling near a known gusher) and **Exploration** (drilling in an unmapped canyon where high uncertainty might reveal an even bigger oil reserve).

---

## 2. The Deep Architecture/Algorithm Breakdown
Bayesian Optimization solves global optimization of expensive black-box functions without gradient access:
$$\max_{x \in \mathcal{X}} f(x)$$
1. **Surrogate Model:** Uses a Gaussian Process $\mathcal{GP}(m(x), k(x, x'))$ to model unknown function $f(x)$ with posterior mean $\mu(x)$ and variance $\sigma^2(x)$.
2. **Acquisition Function $\alpha(x)$:** Fast, cheap auxiliary proxy function guiding where to sample next.
3. **Acquisition Functions:**
   - **Expected Improvement (EI):** Quantifies expected improvement over best observed value so far $f^+ = \max y_i$.
   - **Upper Confidence Bound (UCB):** $\alpha_{\text{UCB}}(x) = \mu(x) + \kappa \sigma(x)$ (optimism in the face of uncertainty).
   - **Probability of Improvement (PI):** $\alpha_{\text{PI}}(x) = P(f(x) \ge f^+ + \xi)$.

---

## 3. The Hardcore Mathematics

### 3.1 Closed-Form Derivation of Expected Improvement (EI)
Let current best value be $f^+ = \max_{i} y_i$.
Improvement is $I(x) = \max(0, f(x) - f^+)$.
Under the GP posterior $f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$:
$$\mathbb{E}[I(x)] = \int_{f^+}^\infty (f - f^+) \frac{1}{\sqrt{2\pi}\sigma(x)} \exp\left( -\frac{(f - \mu(x))^2}{2\sigma^2(x)} \right) df$$
Standardizing $z = \frac{\mu(x) - f^+}{\sigma(x)}$:
$$\alpha_{\text{EI}}(x) = \begin{cases} (\mu(x) - f^+) \Phi(z) + \sigma(x) \phi(z) & \text{if } \sigma(x) > 0 \\ 0 & \text{if } \sigma(x) = 0 \end{cases}$$
Where $\Phi(z)$ is the standard normal cumulative distribution function (CDF) and $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ is the standard normal PDF.
Notice the balance:
- $(\mu(x) - f^+) \Phi(z)$ drives **Exploitation** (high mean).
- $\sigma(x) \phi(z)$ drives **Exploration** (high uncertainty).

### 3.2 Gaussian Process Upper Confidence Bound (GP-UCB)
$$\alpha_{\text{UCB}}(x) = \mu(x) + \kappa \cdot \sigma(x)$$
Where hyperparameter $\kappa$ directly controls the exploration-exploitation tradeoff.

---

## 4. The Algorithm Step-by-Step
1. **Initial Exploration:** Sample small random seed set (e.g. $n_0 = 4$) across search domain; evaluate expensive function $f(x)$.
2. **Sequential Optimization Loop:** For iteration $t = 1, \dots, T$:
   - Fit Gaussian Process regressor on all evaluated pairs $(X, y)$.
   - Evaluate acquisition function $\alpha(x)$ over dense candidate grid.
   - Select next query point $x_{next} = \arg\max_x \alpha(x)$.
   - Evaluate true expensive black-box function $y_{next} = f(x_{next})$.
   - Append $(x_{next}, y_{next})$ to observed dataset.
3. **Return:** Optimal parameters $x^* = \arg\max_i y_i$.

---

## 5. Limitations / When It Breaks
- **High Dimensionality ($d > 20$):** Standard GP surrogates struggle in high dimensions because the volume of the space grows exponentially, making the distance between samples uniformly large.
- **Historical Fix:** TuRBO (Trust Region Bayesian Optimization) and BOHB (Bayesian Optimization with HyperBand).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md` and `Transformer/Reports/Deep_Dives/35_Benchmarking_Efficiency.md`. Bayesian optimization is the primary algorithm used to tune neural architecture hyperparameters, LoRA rank configurations, and reinforcement learning reward weights.

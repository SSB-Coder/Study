# Deep Dive: V04 — Numerical Integration (Trapezoidal, Simpson's, Monte Carlo)

## 1. The Beginner's Analogy: Measuring an Irregular Lake
Imagine calculating the surface area of an irregularly curved lake:
- **The Trapezoidal Rule** drives wooden stakes along the shore and stretches straight strings between them, approximating the lake as a series of trapezoids. It is simple, but it clips convex curves.
- **Simpson's Rule** replaces straight strings with flexible wooden slats that bend into parabolas through every triplet of stakes. Because curves fit natural shorelines far better than straight cuts, Simpson's rule achieves dramatic accuracy with very few stakes.
- **Monte Carlo Integration** takes an airplane over the bounding forest and drops 100,000 paintballs at random. The fraction of paintballs that splash in the water multiplied by the bounding area yields the lake's area. While inefficient in 1D, Monte Carlo is the *only* method that survives the curse of dimensionality in 100-dimensional physics.

---

## 2. The Deep Architecture/Algorithm Breakdown

### 2.1 Quadrature Error Orders
For interval $[a, b]$ subdivided into $N$ intervals of width $h = (b - a) / N$:
- **Composite Trapezoidal:** Error $E_T = -\frac{(b-a) h^2}{12} f''(\xi) = \mathcal{O}(h^2)$.
- **Composite Simpson's 1/3:** Error $E_S = -\frac{(b-a) h^4}{180} f^{(4)}(\xi) = \mathcal{O}(h^4)$.
Remarkably, because odd-degree error terms cancel symmetrically, Simpson's rule is exact for all cubic polynomials even though it only fits quadratic interpolants!

### 2.2 Monte Carlo Integration
In $d$ dimensions over volume $\Omega$, classical quadrature requires $N^d$ grid points. Monte Carlo evaluates $f$ at $N$ uniformly sampled points $\mathbf{x}_i \sim \mathcal{U}(\Omega)$:
$$I = \int_\Omega f(\mathbf{x}) d\mathbf{x} \approx \frac{\text{Vol}(\Omega)}{N} \sum_{i=1}^N f(\mathbf{x}_i)$$
By the Central Limit Theorem, error scales as $\mathcal{O}(1 / \sqrt{N})$ completely **independent of dimension $d$**.

---

## 3. The Hardcore Mathematics

### 3.1 Composite Simpson's Rule Derivation
On sub-interval $[x_{2k}, x_{2k+2}]$, fit quadratic polynomial $P_2(x)$ through points $(x_{2k}, y_{2k}), (x_{2k+1}, y_{2k+1}), (x_{2k+2}, y_{2k+2})$:
$$\int_{x_{2k}}^{x_{2k+2}} P_2(x) dx = \frac{h}{3} \left( y_{2k} + 4 y_{2k+1} + y_{2k+2} \right)$$
Summing over all $N/2$ double-intervals:
$$I \approx \frac{h}{3} \left[ f(x_0) + 4 \sum_{k=1}^{N/2} f(x_{2k-1}) + 2 \sum_{k=1}^{N/2-1} f(x_{2k}) + f(x_N) \right]$$

### 3.2 Monte Carlo Variance & Confidence Interval
Variance of the estimator:
$$\text{Var}(\hat{I}) = \frac{\text{Vol}(\Omega)^2}{N} \sigma^2_f$$
Standard error:
$$\text{SE}(\hat{I}) = \text{Vol}(\Omega) \frac{s_f}{\sqrt{N}}, \quad s_f^2 = \frac{1}{N-1} \sum_{i=1}^N (f(\mathbf{x}_i) - \bar{f})^2$$

---

## 4. The Algorithm Step-by-Step
1. **Choose Method:** If $d \le 3$ and $f$ is smooth, use Simpson's rule. If $d > 4$ or domain is irregular, use Monte Carlo.
2. **Simpson Execution:** Ensure $N$ is even. Compute grid points $x_i = a + i h$. Sum endpoints with weight 1, odd indices with weight 4, even with weight 2. Multiply by $h/3$.
3. **Monte Carlo Execution:** Sample $N$ independent uniform vectors in bounding box. Evaluate $f(\mathbf{x}_i)$. Compute sample mean and standard error.

---

## 5. Limitations / When It Breaks
- **Singularities & High Frequencies:** If $f(x)$ has infinite derivatives (e.g. $1/\sqrt{x}$ at $x=0$) or oscillates faster than $1/h$ (Nyquist limit), polynomial quadrature fails; adaptive mesh refinement or Gauss-Kronrod rules are required.
- **Monte Carlo Slow Convergence:** Reducing error by 10x requires 100x more samples. Variance reduction (importance sampling, stratified sampling, quasi-Monte Carlo Sobol sequences) is mandatory in production.

---

## 6. Where This Fits
1. **Thermodynamic Engine Cycles (W13):** Computing $P-V$ indicator work $W = \oint P dV$ via Simpson's rule.
2. **Bayesian Evidence / Marginal Likelihood (E04):** Computing high-dimensional normalizer $p(D) = \int p(D|\theta) p(\theta) d\theta$ via Monte Carlo.
3. **Physics Particle Simulation (S01, S02):** Integrating force fields across spatial volumes.

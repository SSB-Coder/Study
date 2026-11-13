# Deep Dive: Generalized Linear Models (GLM) & Iteratively Reweighted Least Squares

## 1. The Beginner's Analogy: The Flexible Ruler for Different Materials
Imagine measuring lengths using a standard rigid wooden ruler:
- If you're measuring a wooden plank, a linear ruler works great.
- But what if you are measuring the count of lightning strikes during a thunderstorm? Counts cannot be negative (you can't have $-3$ strikes), and when storms get severe, the count explodes multiplicatively, not additively.
- **A Generalized Linear Model (GLM)** replaces the rigid linear assumption with a custom **link function**. For count data (Poisson), it measures on a logarithmic scale; for probabilities, it measures on a logit scale; for waiting times (Gamma), it measures on an inverse scale.
- **Iteratively Reweighted Least Squares (IRLS)** repeatedly adjusts the ruler's tension and local curve until it fits the physical laws of the data distribution.

---

## 2. The Deep Architecture/Algorithm Breakdown
GLMs unify linear regression, logistic regression, Poisson regression, and Gamma regression under the Exponential Dispersion Family:
1. **Random Component:** Target $y$ follows an exponential dispersion family distribution with conditional mean $\mu = \mathbb{E}[y \mid x]$ and variance $\text{Var}(y \mid x) = \phi V(\mu)$.
2. **Systematic Component:** Linear predictor $\eta = x^T \beta$.
3. **Link Function $g(\cdot)$:** Connects the linear predictor to the expected value: $\eta = g(\mu) \iff \mu = g^{-1}(\eta)$.
4. **Canonical Links:**
   - Gaussian: $g(\mu) = \mu$ (Identity link).
   - Bernoulli: $g(\mu) = \ln\frac{\mu}{1-\mu}$ (Logit link).
   - Poisson: $g(\mu) = \ln(\mu)$ (Log link for positive counts).
   - Gamma: $g(\mu) = 1/\mu$ (Reciprocal link for positive continuous durations).
5. **Iteratively Reweighted Least Squares (IRLS):** Solves maximum likelihood parameters via Fisher scoring: at each iteration, updates $\beta$ by solving a weighted least-squares problem on working response $z$.

---

## 3. The Hardcore Mathematics

### 3.1 Exponential Dispersion Family
Density formulation:
$$f(y; \theta, \phi) = \exp\left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right)$$
Properties:
$$\mathbb{E}[y] = \mu = b'(\theta), \quad \text{Var}(y) = b''(\theta) a(\phi) = V(\mu) a(\phi)$$

### 3.2 IRLS / Fisher Scoring Derivation
Log-likelihood for sample $i$:
$$\ell_i(\beta) = \frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)$$
By chain rule:
$$\frac{\partial \ell_i}{\partial \beta_j} = \frac{\partial \ell_i}{\partial \theta_i} \frac{\partial \theta_i}{\partial \mu_i} \frac{\partial \mu_i}{\partial \eta_i} \frac{\partial \eta_i}{\partial \beta_j} = \frac{y_i - \mu_i}{\phi V(\mu_i)} \cdot \frac{1}{g'(\mu_i)} \cdot x_{ij}$$
The Fisher Information Matrix (Hessian expectation):
$$F = X^T W X, \quad \text{where } W = \text{diag}(w_1, \dots, w_m), \; w_i = \frac{1}{\phi V(\mu_i) [g'(\mu_i)]^2}$$
Define working response variable vector $z \in \mathbb{R}^m$:
$$z_i = \eta_i + (y_i - \mu_i) g'(\mu_i)$$
The Fisher scoring Newton step simplifies to a weighted normal equation:
$$\beta^{(t+1)} = (X^T W X)^{-1} X^T W z$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Set initial mean $\mu^{(0)} = y$ (with zero values shifted by small $\epsilon$) and linear predictor $\eta^{(0)} = g(\mu^{(0)})$.
2. **Iterate until convergence:**
   - Compute mean $\mu = g^{-1}(\eta)$.
   - Evaluate variance function $V(\mu)$ and derivative $g'(\mu)$.
   - Compute diagonal weights $w_i = \frac{1}{V(\mu_i) [g'(\mu_i)]^2}$.
   - Compute working responses $z_i = \eta_i + (y_i - \mu_i) g'(\mu_i)$.
   - Solve weighted linear regression: $\beta \leftarrow (X^T W X)^{-1} X^T W z$.
   - Update $\eta = X\beta$.
3. **Stop:** When $\|\beta^{(t+1)} - \beta^{(t)}\| < 10^{-6}$.

---

## 5. Limitations / When It Breaks
- **Overdispersion:** Poisson models assume mean equals variance ($\mathbb{E}[y] = \text{Var}(y)$). Real count data often exhibits variance much larger than the mean (overdispersion).
- **Zero Inflation:** Excess zeros (e.g. insurance claims) violate standard Poisson/Gamma density.
- **Historical Fix:** Negative Binomial regression handles overdispersion; Zero-Inflated Poisson (ZIP) models handle excess zeros.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/00_The_Complete_Mathematics_of_Neural_Networks.md`. Feedforward neural networks with various output activations and losses are non-linear deep extensions of GLMs.

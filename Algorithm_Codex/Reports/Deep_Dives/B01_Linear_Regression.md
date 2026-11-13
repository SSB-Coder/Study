# Deep Dive: Linear Regression & Regularization (Ridge, Lasso, Elastic Net)

## 1. The Beginner's Analogy: Fitting a Straight Rigid Wire Through a Data Cloud
Imagine hovering above a collection of data points suspended in 3D space like a swarm of fireflies. Your objective is to pass a perfectly straight, stiff steel wire through the center of the swarm.
- **Ordinary Least Squares (OLS)** positions the wire so that the sum of the squared vertical distances from every firefly to the wire is minimized. If a single rogue firefly sits 10 miles away, the stiff wire tilts violently toward it.
- **Ridge Regression ($L_2$)** attaches a rubber band from the wire to the origin, penalizing steep slopes and preventing extreme, unstable swings.
- **Lasso Regression ($L_1$)** attaches a magnetic snap to coordinate axes; if a feature is unhelpful, Lasso pulls its weight all the way to absolute zero, acting as an automatic feature eliminator.
- **Elastic Net** blends both rubber bands, handling clusters of correlated fireflies simultaneously.

---

## 2. The Deep Architecture/Algorithm Breakdown
Linear regression models the relationship between continuous dependent variable $y \in \mathbb{R}$ and feature vector $x \in \mathbb{R}^d$ via a linear combination $y = x^T \beta + \beta_0 + \epsilon$:
1. **Design Matrix $X \in \mathbb{R}^{m \times d}$:** Augmented with a column of ones for intercept $\beta_0$.
2. **Residual Sum of Squares (RSS):** The convex quadratic objective $\mathcal{L}_{OLS}(\beta) = \frac{1}{2m} \|y - X\beta\|_2^2$.
3. **Normal Equation:** The exact closed-form analytical solution $\hat{\beta} = (X^T X)^{-1} X^T y$.
4. **Gradient Descent:** Iterative first-order optimization navigating the quadratic loss bowl $\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla \mathcal{L}(\beta)$.
5. **Regularization Penalties:**
   - **Ridge ($L_2$):** Adds $\frac{\lambda}{2} \|\beta\|_2^2$, guaranteeing invertibility of $(X^T X + \lambda I)$.
   - **Lasso ($L_1$):** Adds $\lambda \|\beta\|_1$, creating diamond-shaped constraint boundaries that induce exact parameter sparsity via soft-thresholding.
   - **Elastic Net:** Combines $L_1$ and $L_2$ penalties parameterized by $\alpha$ and $l_1$-ratio.

---

## 3. The Hardcore Mathematics

### 3.1 Normal Equation Derivation
Let objective $\mathcal{L}(\beta) = \frac{1}{2} (y - X\beta)^T (y - X\beta)$.
Expanding:
$$\mathcal{L}(\beta) = \frac{1}{2} \left( y^T y - 2\beta^T X^T y + \beta^T X^T X \beta \right)$$
Taking the vector derivative with respect to $\beta$:
$$\nabla_\beta \mathcal{L}(\beta) = -X^T y + X^T X \beta = 0 \implies X^T X \beta = X^T y$$
$$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$

### 3.2 Gradient Descent
For batch optimization over $m$ samples:
$$\nabla_\beta \mathcal{L}(\beta) = \frac{1}{m} X^T (X\beta - y)$$
Update rule:
$$\beta \leftarrow \beta - \alpha \cdot \frac{1}{m} X^T (X\beta - y)$$

### 3.3 Ridge Regression Closed-Form
Objective: $\mathcal{L}_{\text{Ridge}}(\beta) = \frac{1}{2m} \|y - X\beta\|_2^2 + \frac{\lambda}{2} \|\beta\|_2^2$.
Differentiating:
$$\nabla_\beta \mathcal{L}_{\text{Ridge}}(\beta) = \frac{1}{m} X^T (X\beta - y) + \lambda \beta = 0$$
$$\hat{\beta}_{\text{Ridge}} = (X^T X + m\lambda I)^{-1} X^T y$$

### 3.4 Lasso Coordinate Descent & Soft-Thresholding
Objective: $\mathcal{L}_{\text{Lasso}}(\beta) = \frac{1}{2m} \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^d |\beta_j|$.
Because $|\beta_j|$ is non-differentiable at $\beta_j = 0$, we use subgradient coordinate descent. Isolating coordinate $j$:
Let $\rho_j = X_{:,j}^T (y - X_{:,-j}\beta_{-j})$ and $z_j = X_{:,j}^T X_{:,j}$.
The coordinate-wise minimizer is given by the **Soft-Thresholding Operator** $\mathcal{S}_{\tau}(u) = \text{sign}(u)(|u| - \tau)_+$:
$$\hat{\beta}_j = \frac{\mathcal{S}_{m\lambda}(\rho_j)}{z_j} = \begin{cases} \frac{\rho_j - m\lambda}{z_j} & \text{if } \rho_j > m\lambda \\ 0 & \text{if } |\rho_j| \le m\lambda \\ \frac{\rho_j + m\lambda}{z_j} & \text{if } \rho_j < -m\lambda \end{cases}$$

---

## 4. The Algorithm Step-by-Step
1. **Feature Centering / Standardization:** Standardize $X$ so features have zero mean and unit variance.
2. **Add Bias Column:** Concatenate a column of ones if bias is not estimated separately.
3. **Solver Selection:**
   - If $d \le 2000$ and $X^T X$ is well-conditioned: compute Normal Equation directly.
   - If $d \gg m$ or multicollinear: solve via Ridge $(X^T X + \lambda I)^{-1} X^T y$.
   - If sparsity required: iterate coordinate descent over $j \in \{1, \dots, d\}$ updating $\beta_j$ via soft-thresholding until convergence $\|\beta^{(t+1)} - \beta^{(t)}\| < \epsilon$.
   - If large-scale streaming data: execute batch or mini-batch gradient descent.

---

## 5. Limitations / When It Breaks
- **Multicollinearity:** When features are linearly dependent, $X^T X$ is singular (determinant zero), causing Normal Equation inversion to explode.
- **Outlier Sensitivity:** Squaring errors heavily penalizes large residuals, tilting the hyperplane away from true trends.
- **Historical Fix:** Ridge regularization fixes matrix singularity; Huber / LAD regression mitigates outlier sensitivity; Polynomial / Spline expansion captures non-linear manifolds.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/01_MLP_Feedforward.md`. Linear regression is the single-layer perceptron with identity activation. In Transformer systems, Low-Rank Adaptation (LoRA) and linear attention mechanisms build directly on linear algebra projection primitives.

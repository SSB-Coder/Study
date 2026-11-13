# Deep Dive: Linear and Quadratic Discriminant Analysis (LDA and QDA)

## 1. The Beginner's Analogy: Casting the Sharpest Shadow
Imagine holding several multi-colored clusters of balloons in a 3D room:
- You hold a flashlight and project the shadows of all the balloons onto a flat 2D wall.
- If you shine the light from an arbitrary angle, the shadows of the red balloons and blue balloons overlap into a confused, messy silhouette.
- **Linear Discriminant Analysis (LDA)** finds the exact optical angle where the centers of the red and blue shadows are pushed as far apart as possible, while the shadows of balloons within each color are packed as tightly as possible.
- **Quadratic Discriminant Analysis (QDA)** allows the balloons of each color to have different shapes (e.g. red balloons form long thin cigars, blue balloons form flat pancakes), producing curved quadratic separation boundaries.

*(Note: Linear Discriminant Analysis is distinct from Latent Dirichlet Allocation in Module E08, which also shares the acronym "LDA".)*

---

## 2. The Deep Architecture/Algorithm Breakdown
LDA and QDA are generative classifiers modeling the class-conditional density $P(X \mid Y=k)$ as multivariate Gaussian distributions:
1. **Multivariate Gaussian Assumption:** $X \mid Y=k \sim \mathcal{N}(\mu_k, \Sigma_k)$.
2. **Prior Probabilities:** $\pi_k = P(Y=k) = \frac{N_k}{N}$.
3. **Linear Discriminant Analysis (LDA):** Assumes all classes share an **identical, pooled covariance matrix** $\Sigma_k = \Sigma$. The quadratic terms in the log-odds cancel out, resulting in a strictly linear decision boundary.
4. **Quadratic Discriminant Analysis (QDA):** Permits each class to possess its own distinct covariance matrix $\Sigma_k$, producing curved conic section decision boundaries (ellipses, parabolas, hyperbolas).
5. **Fisher's Linear Discriminant:** An alternative derivation maximizing Rayleigh's quotient: between-class scatter $S_B$ over within-class scatter $S_W$.

---

## 3. The Hardcore Mathematics

### 3.1 Bayes Optimal Discriminant Function
By Bayes' rule:
$$P(Y=k \mid X=x) \propto P(X=x \mid Y=k) P(Y=k)$$
Taking the natural logarithm of the joint density:
$$\delta_k(x) = \ln P(X=x \mid Y=k) + \ln \pi_k = -\frac{d}{2} \ln(2\pi) - \frac{1}{2} \ln |\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln \pi_k$$

### 3.2 LDA Formulation (Shared Covariance $\Sigma$)
When $\Sigma_k = \Sigma$ for all $k$, the quadratic term $x^T \Sigma^{-1} x$ is common across all classes and can be discarded:
$$\delta_k^{\text{LDA}}(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \ln \pi_k$$
This is strictly linear in $x$: $w_k^T x + b_k$, where $w_k = \Sigma^{-1} \mu_k$ and $b_k = -\frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \ln \pi_k$.
The pooled covariance estimator:
$$\hat{\Sigma} = \frac{1}{N - K} \sum_{k=1}^K \sum_{i: y_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

### 3.3 QDA Formulation (Separate Covariances $\Sigma_k$)
Retains class-specific covariance matrices $\Sigma_k$:
$$\delta_k^{\text{QDA}}(x) = -\frac{1}{2} \ln |\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln \pi_k$$
The sample covariance for class $k$:
$$\hat{\Sigma}_k = \frac{1}{N_k - 1} \sum_{i: y_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

---

## 4. The Algorithm Step-by-Step
1. **Estimate Parameters:**
   - Compute class sample counts $N_k$ and priors $\pi_k = N_k / N$.
   - Compute class centroid vectors $\hat{\mu}_k = \frac{1}{N_k} \sum_{i: y_i=k} x_i$.
   - For LDA: compute pooled covariance $\hat{\Sigma}$ and invert via pseudo-inverse.
   - For QDA: compute individual covariances $\hat{\Sigma}_k$, determinants $|\hat{\Sigma}_k|$, and inverses $\hat{\Sigma}_k^{-1}$.
2. **Evaluate Discriminants:** For test vector $x$, compute $\delta_k(x)$ across all $k \in \{1, \dots, K\}$.
3. **Classification:** Predict $\hat{y} = \arg\max_k \delta_k(x)$.

---

## 5. Limitations / When It Breaks
- **Singular Covariance Matrix ($d > N$):** When feature count exceeds observations, $\Sigma$ is low-rank and cannot be inverted without shrinkage regularization (Ledoit-Wolf covariance).
- **Non-Gaussian Manifolds:** If data clusters are multi-modal or horseshoe-shaped, Gaussian class assumptions degrade.
- **Historical Fix:** Regularized Discriminant Analysis (RDA) and Kernel Fisher Discriminant Analysis.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/33_Mechanistic_Interpretability.md`. In mechanistic interpretability, extracting linear concept directions from hidden activations relies on linear probe projections equivalent to Fisher's discriminant.

# O03: Information Geometry & Natural Gradients

## 1. The Beginner's Analogy
Imagine you are hiking through a mountain range on a map.
If the map is drawn in inches, taking a "1-inch step" north might move you 100 feet across a flat plain, but that same "1-inch step" on a steep cliff might send you falling 5,000 feet into a canyon! 
Standard Gradient Descent treats all parameters as flat Euclidean numbers ($d\theta$). If you change the variance of a distribution from $0.01$ to $0.02$, you double its spread (a massive catastrophic change in distribution space), whereas changing it from $10.01$ to $10.02$ changes almost nothing.
**Information Geometry** recognizes that probability distributions do not live in flat Euclidean space; they live on a curved **Riemannian manifold**:
- The **Fisher Information Matrix (FIM)** acts as the local "gravity / curvature gauge" of probability distributions.
- **Natural Gradient Descent (Amari)** takes steps that are measured in constant units of **Kullback-Leibler (KL) divergence** rather than parameter coordinates.
- Regardless of how you choose to write down the parameters (log-variance vs variance vs standard deviation), the natural gradient path through the space of distributions is identical and invariant!

## 2. The Deep Architecture & Mechanics
Information geometry establishes a rigorous bridge between differential geometry, statistics, and optimization:
1. **The Statistical Manifold $\mathcal{M}$:**
   A parametric family $\mathcal{S} = \{p_\theta(x) : \theta \in \Theta\}$ forms a smooth differentiable manifold where each point is an entire probability distribution.
2. **The Fisher Information Metric:**
   The unique Riemannian metric tensor invariant under sufficient statistics (Chentsov's Theorem):
   $$g_{ij}(\theta) = F_{ij}(\theta) = \mathbb{E}_{x \sim p_\theta}\left[ \frac{\partial \ln p_\theta(x)}{\partial \theta_i} \frac{\partial \ln p_\theta(x)}{\partial \theta_j} \right]$$
3. **KL Divergence as Manifold Distance:**
   Locally, the symmetric KL divergence between two infinitesimally close distributions is quadratic:
   $$D_{\text{KL}}(p_\theta \parallel p_{\theta + d\theta}) = \frac{1}{2} d\theta^T F(\theta) d\theta + \mathcal{O}(\|d\theta\|^3)$$
4. **Natural Gradient Update:**
   To minimize loss $\mathcal{L}(\theta)$ subject to a fixed distribution change constraint $D_{\text{KL}}(p_\theta \parallel p_{\theta + d\theta}) \le \epsilon$:
   $$\tilde{\nabla}_\theta \mathcal{L} = F(\theta)^{-1} \nabla_\theta \mathcal{L}$$
   $$\theta^{(t+1)} = \theta^{(t)} - \eta F(\theta)^{-1} \nabla_\theta \mathcal{L}$$

## 3. The Hardcore Mathematics

### Derivation of Natural Gradient via Constrained Optimization
Consider the steepest descent direction on the statistical manifold constrained to a trust region of size $\epsilon$:
$$\min_{d\theta} \mathcal{L}(\theta + d\theta) \quad \text{s.t.} \quad D_{\text{KL}}(p_\theta \parallel p_{\theta + d\theta}) \le \epsilon$$
First-order Taylor expansion of loss: $\mathcal{L}(\theta + d\theta) \approx \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}^T d\theta$.
Second-order expansion of KL divergence around $d\theta = 0$:
$$D_{\text{KL}}(p_\theta \parallel p_{\theta + d\theta}) = -\int p_\theta(x) \ln \frac{p_{\theta + d\theta}(x)}{p_\theta(x)} dx$$
Since $\nabla_\theta \ln p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}$:
$$\nabla_{d\theta} D_{\text{KL}} \Big|_{d\theta=0} = -\int p_\theta(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} dx = -\nabla_\theta \int p_\theta(x) dx = -\nabla_\theta 1 = 0$$
The Hessian is the Fisher Information Matrix:
$$\nabla_{d\theta}^2 D_{\text{KL}} \Big|_{d\theta=0} = \mathbb{E}\left[ \nabla \ln p_\theta (\nabla \ln p_\theta)^T \right] = F(\theta)$$
Setting the Lagrangian:
$$\min_{d\theta} \left( \nabla_\theta \mathcal{L}^T d\theta + \frac{\lambda}{2} d\theta^T F(\theta) d\theta \right)$$
Differentiating with respect to $d\theta$ and setting to zero:
$$\nabla_\theta \mathcal{L} + \lambda F(\theta) d\theta = 0 \implies d\theta = -\frac{1}{\lambda} F(\theta)^{-1} \nabla_\theta \mathcal{L}$$
Yielding the Natural Gradient step with learning rate $\eta = 1/\lambda$.

### Analytical Fisher Matrix for 1D Gaussian Distribution
Let $p(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$.
Log-likelihood:
$$\ln p(x; \mu, \sigma) = -\ln \sigma - \frac{1}{2}\ln(2\pi) - \frac{(x - \mu)^2}{2\sigma^2}$$
Partial derivatives (scores):
$$\frac{\partial \ln p}{\partial \mu} = \frac{x - \mu}{\sigma^2}, \qquad \frac{\partial \ln p}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(x - \mu)^2}{\sigma^3}$$
Expected outer products:
$$\mathbb{E}\left[ \left(\frac{\partial \ln p}{\partial \mu}\right)^2 \right] = \frac{\mathbb{E}[(x - \mu)^2]}{\sigma^4} = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$$
$$\mathbb{E}\left[ \frac{\partial \ln p}{\partial \mu} \frac{\partial \ln p}{\partial \sigma} \right] = \frac{\mathbb{E}[(x - \mu)^3]}{\sigma^5} - \frac{\mathbb{E}[x - \mu]}{\sigma^3} = 0$$
$$\mathbb{E}\left[ \left(\frac{\partial \ln p}{\partial \sigma}\right)^2 \right] = \frac{1}{\sigma^2} - \frac{2\mathbb{E}[(x - \mu)^2]}{\sigma^4} + \frac{\mathbb{E}[(x - \mu)^4]}{\sigma^6} = \frac{1}{\sigma^2} - \frac{2}{\sigma^2} + \frac{3\sigma^4}{\sigma^6} = \frac{2}{\sigma^2}$$
Thus, the Gaussian Fisher Information Matrix is diagonal:
$$F(\mu, \sigma) = \begin{bmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{bmatrix}, \qquad F(\mu, \sigma)^{-1} = \begin{bmatrix} \sigma^2 & 0 \\ 0 & \frac{\sigma^2}{2} \end{bmatrix}$$

## 4. The Algorithm Step-by-Step
1. Compute the analytical or empirical gradient $\nabla_\theta \mathcal{L}$ on current batch.
2. Estimate the Fisher Information Matrix $F(\theta) = \frac{1}{B} \sum_{b=1}^B \nabla_\theta \ln p_\theta(x_b) \nabla_\theta \ln p_\theta(x_b)^T$.
3. Add damping $\delta I$ for numerical conditioning: $F_{\text{damped}} = F(\theta) + \delta I$.
4. Solve linear system $F_{\text{damped}} \Delta \theta = \nabla_\theta \mathcal{L}$ via Cholesky decomposition or conjugate gradient.
5. Update parameters: $\theta \leftarrow \theta - \eta \Delta \theta$.

## 5. Limitations & Failure Modes
- **Inversion Scalability:** Inverting an $M \times M$ Fisher matrix scales cubically $\mathcal{O}(M^3)$, which is intractable for neural networks with millions of parameters (solved via Kronecker-factored Approximate Curvature, K-FAC, or TRPO conjugate gradient).
- **Singular Fisher Matrices:** For over-parameterized models, $F(\theta)$ is rank-deficient and singular, necessitating Tikhonov damping.

## 6. Where This Fits
Cross-reference: Natural gradients provide the exact mathematical foundation for Trust Region Policy Optimization (TRPO, G09) and Natural Actor-Critic in reinforcement learning, as well as second-order curvature approximations (I02 BFGS).

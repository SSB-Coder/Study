# P02: Langevin Dynamics & Metropolis-Adjusted Langevin Algorithm (MALA)

## 1. The Beginner's Analogy
Imagine a pollen grain suspended in a drop of water (Brownian motion). 
Water molecules randomly bombard the pollen grain from all sides, causing it to jitter randomly. 
Now place that drop of water inside an electric field that gently pulls the pollen toward the center of the dish.
The pollen experiences two competing forces:
1. **Deterministic Drift:** Pushing it along the gradient toward high-probability regions (low potential energy).
2. **Thermal Diffusion:** Random thermal jitter kicking it in all directions, preventing it from getting stuck in any single local pit.

This physical balance is **Langevin Dynamics**. 
Unlike standard gradient descent (which greedily collapses to a single point), Langevin dynamics explores the entire probability distribution, spending more time in high-density regions while continuously diffusing through lower-density regions.

## 2. The Deep Architecture & Mechanics
Langevin dynamics forms the mathematical backbone connecting continuous physics, Bayesian sampling, and modern generative diffusion models:
1. **Continuous Overdamped Langevin SDE:**
   $$dx_t = \frac{1}{2} \nabla \ln p(x_t) dt + dW_t$$
   where $W_t$ is standard Brownian motion with covariance $\mathbb{E}[dW_t dW_t^T] = I dt$.
2. **The Fokker-Planck Stationary Invariant:**
   The probability density $\rho(x, t)$ of particles evolving under this SDE satisfies the Fokker-Planck PDE. As $t \to \infty$, the unique stationary distribution is provably $\rho_\infty(x) = p(x)$.
3. **Unadjusted Langevin Algorithm (ULA):**
   Discretizing the SDE via the Euler-Maruyama scheme:
   $$x_{k+1} = x_k + \frac{\varepsilon^2}{2} \nabla \ln p(x_k) + \varepsilon \xi_k, \quad \xi_k \sim \mathcal{N}(0, I)$$
4. **Metropolis Correction (MALA):**
   Finite step sizes $\varepsilon > 0$ introduce an asymptotic discretization bias $O(\varepsilon^2)$. MALA accepts or rejects each step using a Metropolis-Hastings filter, ensuring exact convergence to the true posterior.

## 3. The Hardcore Mathematics

### Fokker-Planck Equilibrium Derivation
The Fokker-Planck equation for drift vector $\mu(x) = \frac{1}{2} \nabla \ln p(x)$ and diffusion tensor $D = \frac{1}{2} I$ is:
$$\frac{\partial \rho(x, t)}{\partial t} = -\nabla \cdot (\mu(x) \rho(x, t)) + \frac{1}{2} \nabla^2 \rho(x, t)$$
Substituting $\mu(x) = \frac{1}{2} \frac{\nabla p(x)}{p(x)}$ into the probability flux $J(x, t)$:
$$J(x, t) = \frac{1}{2} \nabla \ln p(x) \rho(x, t) - \frac{1}{2} \nabla \rho(x, t)$$
At equilibrium $\rho(x) = p(x)$:
$$J(x) = \frac{1}{2} \frac{\nabla p(x)}{p(x)} p(x) - \frac{1}{2} \nabla p(x) = \frac{1}{2}\nabla p(x) - \frac{1}{2}\nabla p(x) = 0$$
Because the probability current identically vanishes everywhere, detailed balance holds, and $p(x)$ is the stationary equilibrium distribution.

### MALA Proposal Transition Probability
The Euler-Maruyama discretization defines a Gaussian proposal density $q(x^* | x) = \mathcal{N}(x^*; \mu(x), \varepsilon^2 I)$ where:
$$\mu(x) = x + \frac{\varepsilon^2}{2} \nabla \ln p(x)$$
Log-probability of the proposal:
$$\ln q(x^* | x) = -\frac{d}{2}\ln(2\pi\varepsilon^2) - \frac{1}{2\varepsilon^2} \left\| x^* - x - \frac{\varepsilon^2}{2}\nabla \ln p(x) \right\|_2^2$$
The Metropolis acceptance probability is:
$$\alpha(x, x^*) = \min\left(1, \; \frac{p(x^*) q(x | x^*)}{p(x) q(x^* | x)}\right)$$
$$\ln \alpha(x, x^*) = \min\left(0, \; \ln p(x^*) - \ln p(x) + \ln q(x | x^*) - \ln q(x^* | x)\right)$$

## 4. The Algorithm Step-by-Step
1. Initialize position $x_0$ and step size $\varepsilon$.
2. For iteration $k = 0 \dots K-1$:
   - Compute drift $\mu(x_k) = x_k + \frac{\varepsilon^2}{2} \nabla \ln p(x_k)$.
   - Draw random Gaussian kick $\xi \sim \mathcal{N}(0, I_D)$.
   - Propose $x^* = \mu(x_k) + \varepsilon \xi$.
   - Compute forward proposal log-density $\ln q(x^* | x_k)$ and reverse proposal $\ln q(x_k | x^*)$.
   - Evaluate Metropolis ratio $\ln \alpha$.
   - Accept $x_{k+1} = x^*$ if $\ln \text{Uniform}(0, 1) < \ln \alpha$; else reject $x_{k+1} = x_k$.

## 5. Limitations & Failure Modes
- **Step-Size Instability:** If $\varepsilon$ is too large, the drift term $\frac{\varepsilon^2}{2}\nabla \ln p(x)$ overshoots into regions of low probability, causing proposals to explode or be continuously rejected.
- **Multimodal Trapping:** Like all local MCMC methods, Langevin particles can take exponential time to cross deep energy barriers between disconnected modes.

## 6. Where This Fits
Cross-reference: Langevin dynamics is the generative mechanism used in reverse-time diffusion models (DDPMs, SGM) to generate images from noise, and provides the gradient-guided alternative to Gibbs sampling (E06) and HMC (P01).

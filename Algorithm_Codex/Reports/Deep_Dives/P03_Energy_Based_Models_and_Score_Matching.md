# P03: Energy-Based Models & Denoising Score Matching

## 1. The Beginner's Analogy
In standard probability modeling, if you want to assign probabilities to pictures of cats, you face an impossible mathematical hurdle: every probability in the universe must sum to exactly $1.0$. 
Calculating that total sum—integrating over every possible arrangement of pixels in existence (the partition function $Z$)—is mathematically intractable ($Z = \int e^{-E(x)} dx$).
**Score Matching and Energy-Based Models** bypass this barrier with an ingenious mathematical insight:
- Instead of asking "What is the exact percentage probability of this point?", we ask: "If I drop a marble here, which direction is uphill toward the nearest cat?" (the gradient of log-density, $\nabla_x \ln p(x)$, known as the **Score**).
- Because $Z$ is a constant with respect to $x$, its derivative $\nabla_x Z$ is identically **zero**!
- To learn the score field without knowing the true density, **Denoising Score Matching** adds a tiny bit of Gaussian noise to data points and trains a model to point directly back to the clean data point. 
Following these learned score arrows with Langevin dynamics (P02) generates pristine data out of pure static noise—the foundational mechanism of all modern **Diffusion Models**!

## 2. The Deep Architecture & Mechanics
1. **Energy-Based Models (EBMs):**
   Model probability through a scalar energy function $E_\theta(x) \in \mathbb{R}$:
   $$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad Z_\theta = \int_{\mathbb{R}^D} \exp(-E_\theta(x)) dx$$
2. **The Score Function:**
   Defined as the gradient of the log-probability density with respect to the input:
   $$s_\theta(x) \triangleq \nabla_x \ln p_\theta(x) = -\nabla_x E_\theta(x)$$
   Notice that the intractable normalizer $Z_\theta$ is completely eliminated.
3. **Denoising Score Matching (Vincent 2011):**
   Perturbs data $x$ with noise $\tilde{x} = x + \sigma \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$.
   The conditional distribution $q(\tilde{x} | x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$ has an exact analytical score:
   $$\nabla_{\tilde{x}} \ln q(\tilde{x} | x) = -\frac{\tilde{x} - x}{\sigma^2}$$
   Vincent proved that minimizing the Fisher divergence against this noisy kernel estimates the true data score:
   $$\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}, \tilde{x} \sim q(\tilde{x}|x)}\left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \ln q(\tilde{x}|x) \right\|_2^2 \right]$$

## 3. The Hardcore Mathematics

### Vincent's Denoising Score Matching Equivalence Theorem
Let $q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x}|x) p(x) dx$ be the smoothed data distribution.
The explicit score matching objective on the smoothed distribution is:
$$J_{\text{ESM}}(\theta) = \frac{1}{2} \int q_\sigma(\tilde{x}) \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \ln q_\sigma(\tilde{x}) \right\|_2^2 d\tilde{x}$$
Expanding the square:
$$J_{\text{ESM}}(\theta) = \int q_\sigma(\tilde{x}) \left[ \frac{1}{2} \|s_\theta(\tilde{x})\|^2 - s_\theta(\tilde{x}) \cdot \nabla_{\tilde{x}} \ln q_\sigma(\tilde{x}) \right] d\tilde{x} + C$$
Substituting $\nabla_{\tilde{x}} \ln q_\sigma(\tilde{x}) = \frac{\nabla_{\tilde{x}} q_\sigma(\tilde{x})}{q_\sigma(\tilde{x})}$ and swapping the integral with the joint density $q_\sigma(\tilde{x}, x) = q_\sigma(\tilde{x}|x) p(x)$:
$$\int q_\sigma(\tilde{x}) s_\theta(\tilde{x}) \cdot \frac{\nabla_{\tilde{x}} q_\sigma(\tilde{x})}{q_\sigma(\tilde{x})} d\tilde{x} = \int \int p(x) q_\sigma(\tilde{x}|x) s_\theta(\tilde{x}) \cdot \nabla_{\tilde{x}} \ln q_\sigma(\tilde{x}|x) d\tilde{x} dx$$
Reassembling into a complete square:
$$\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p(x) q_\sigma(\tilde{x}|x)}\left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \ln q_\sigma(\tilde{x}|x) \right\|_2^2 \right]$$
Remarkably, the intractable marginal score $\nabla \ln q_\sigma(\tilde{x})$ is replaced by the simple closed-form conditional score $-\frac{\tilde{x} - x}{\sigma^2}$!

### Score-Based Generative Sampling (Langevin Sampling)
Once the score network $s_\theta(x) \approx \nabla_x \ln p(x)$ is trained, new samples are generated starting from pure Gaussian noise $x_0 \sim \mathcal{N}(0, I)$ by running Langevin dynamics:
$$x_{k+1} = x_k + \frac{\alpha}{2} s_\theta(x_k) + \sqrt{\alpha} \xi_k, \quad \xi_k \sim \mathcal{N}(0, I)$$
As $\alpha \to 0$ and $k \to \infty$, the trajectory samples exactly from the data distribution.

## 4. The Algorithm Step-by-Step
1. **Training Pass (Denoising Score Matching):**
   - Sample batch of clean training points $x \sim \mathcal{D}$.
   - Sample noise vector $\varepsilon \sim \mathcal{N}(0, I)$ and scale $\sigma$.
   - Form corrupted points $\tilde{x} = x + \sigma \varepsilon$.
   - Target ground truth score: $y = -\varepsilon / \sigma$.
   - Compute model score prediction $s_\theta(\tilde{x})$.
   - Loss: $\mathcal{L} = \frac{1}{2B} \sum_{i=1}^B \|s_\theta(\tilde{x}_i) - y_i\|_2^2$.
   - Update model parameters via gradient descent.
2. **Sampling Pass (Reverse Diffusion):**
   - Initialize particles $x_0 \sim \mathcal{N}(0, \sigma_{\max}^2 I)$.
   - Simulate annealed Langevin diffusion down to $\sigma_{\min}$.
   - Return clean generated samples $x_K$.

## 5. Limitations & Failure Modes
- **Low-Density Manifold Disconnection:** In empty space between data modes, no training samples exist. The model cannot learn meaningful score arrows in empty regions unless perturbed with very large noise $\sigma$ (the motivation for multi-scale annealed diffusion).
- **Curse of Dimensionality in Energy Normalization:** Evaluating the true likelihood $\ln p_\theta(x)$ remains impossible without bridge sampling or thermodynamic integration.

## 6. Where This Fits
Cross-reference: Score Matching connects classical energy-based Hopfield networks (K01) and Boltzmann machines to contemporary Score-Based Generative Models (Song & Ermon) and Denoising Diffusion Probabilistic Models (DDPMs).

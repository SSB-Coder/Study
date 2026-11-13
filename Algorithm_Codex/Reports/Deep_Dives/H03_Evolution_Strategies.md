# H03: Evolution Strategies (ES) & CMA-ES

## 1. Intuition & The "Grand Analogy"
Imagine blindfolded hikers trying to locate the highest point on an uncharted mountain range in a fog. They do not know the gradient of the slope. Instead, the team leader sends out a ring of scouting drones in all directions from their current position according to a 3D Gaussian cloud. The scouts report their altitudes. The leader computes a weighted average of the scouts who found higher elevations, moves the team's center of mass toward that direction, and rotates/stretches the exploration cloud to align with the mountain's ridgeline.

This is **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** (Hansen & Ostermeier, 2001). It is universally regarded as the gold standard in continuous black-box, derivative-free optimization.

---

## 2. Architectural & Mathematical Breakdown
At each generation $g$, sample $\lambda$ candidate vectors from a multivariate Gaussian distribution:
$$\mathbf{x}_k \sim \mathcal{N}\left( \mathbf{m}^{(g)}, \, (\sigma^{(g)})^2 \mathbf{C}^{(g)} \right), \quad k = 1, \dots, \lambda$$
where:
- $\mathbf{m} \in \mathbb{R}^d$: Distribution mean.
- $\sigma > 0$: Step-size (overall scale).
- $\mathbf{C} \in \mathbb{R}^{d \times d}$: Covariance matrix (geometric shape and orientation).

### Mean Update (Recombination of Top $\mu$ Elites)
Sort samples by fitness $f(\mathbf{x}_{1:\lambda}) \le f(\mathbf{x}_{2:\lambda}) \le \dots \le f(\mathbf{x}_{\lambda:\lambda})$.
$$\mathbf{m}^{(g+1)} = \sum_{i=1}^\mu w_i \mathbf{x}_{i:\lambda}, \quad \sum_{i=1}^\mu w_i = 1, \; w_1 \ge \dots \ge w_\mu > 0$$

### Covariance Matrix Adaptation (Rank-$\mu$ Update)
$$\mathbf{C}^{(g+1)} = (1 - c_1 - c_\mu) \mathbf{C}^{(g)} + c_1 \mathbf{p}_c \mathbf{p}_c^\top + c_\mu \sum_{i=1}^\mu w_i \mathbf{y}_{i:\lambda} \mathbf{y}_{i:\lambda}^\top$$
where $\mathbf{y}_i = (\mathbf{x}_{i:\lambda} - \mathbf{m}^{(g)}) / \sigma^{(g)}$.

---

## 3. Hardcore Mathematical Derivations

### Natural Evolution Strategies (NES) Formulation
Let $p_\theta(x) = \mathcal{N}(x; m, \Sigma)$ be a parameterized search distribution. The objective is to maximize expected fitness:
$$J(\theta) = \mathbb{E}_{x \sim p_\theta}[f(x)] = \int f(x) p_\theta(x) dx$$
The search gradient is:
$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim p_\theta} [ f(x) \nabla_\theta \ln p_\theta(x) ]$$
Natural Evolution Strategies (NES) applies the **Fisher Information Matrix** $F_\theta$ to take invariant natural gradient steps:
$$\tilde{\nabla}_\theta J = F_\theta^{-1} \nabla_\theta J(\theta)$$
CMA-ES can be rigorously derived as a second-order natural gradient descent algorithm that estimates the inverse Hessian $\mathbf{H}^{-1}$ of the underlying objective function without computing any derivatives!

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: CMA-ES (Simplified Canonical)
Inputs: Initial mean m_0, step size sigma_0, population size lambda, elite count mu
1. Initialize C = I, evolution paths p_sigma = 0, p_c = 0
2. For generation g = 0 to G_max:
     a. Sample lambda candidate vectors: x_k = m + sigma * (B * D * z_k), z_k ~ N(0, I)
     b. Evaluate fitness f(x_k) for all k
     c. Sort samples by fitness ascending
     d. Update mean m = sum_{i=1}^mu w_i * x_{i:lambda}
     e. Update evolution path p_c and step-size path p_sigma
     f. Update covariance matrix C using rank-1 and rank-mu updates
     g. Decompose C = B * D^2 * B^T via eigendecomposition
3. Return best discovered solution
```

---

## 5. Failure Modes & When It Breaks
1. **High Dimensionality:** Inverting or eigendecomposing $\mathbf{C} \in \mathbb{R}^{d \times d}$ costs $\mathcal{O}(d^3)$ per generation. For $d > 1,000$ dimensions, diagonal CMA-ES (Sep-CMA-ES) is required.
2. **Deceptive Multi-Modality:** If the step size $\sigma$ contracts prematurely around a local valley, CMA-ES collapses its covariance into a needle and gets trapped.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **OpenAI Evolution Strategies for RL:** Salimans et al. (2017) demonstrated that natural evolution strategies can train deep transformer and neural policies completely without backpropagation.
- **Black-Box Prompt Tuning:** When API access to LLM weights or log-probabilities is restricted, CMA-ES optimizes continuous soft-prompt embedding vectors directly from binary success outputs.

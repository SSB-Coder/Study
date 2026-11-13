# O01: Earth Mover's Distance & Wasserstein Metrics (Monge-Kantorovich Formulation)

## 1. The Beginner's Analogy
Imagine you have several piles of dirt located at various positions across a construction site (distribution $p$). You need to fill several holes located elsewhere (distribution $q$). 
Moving a pound of dirt over a distance of one meter requires physical work.
- Traditional metrics like Euclidean distance or Mean Squared Error ($L_2$) compare points at the **exact same coordinate index**. If one pile moves just one millimeter to the left, $L_2$ can explode or become completely uninformative if the supports don't overlap.
- **Kullback-Leibler (KL) Divergence** divides by the target probability. If $p(x) > 0$ where $q(x) = 0$, KL diverges to infinity!
- **The Earth Mover's Distance (Wasserstein Distance):** Calculates the **absolute minimum total physical work** (mass $\times$ distance) required to shovel and transport the entire pile of dirt into the holes. 
Even if two probability distributions are miles apart with zero overlap, the Wasserstein distance provides a smooth, continuous, meaningful gradient reflecting their physical separation distance.

## 2. The Deep Architecture & Mechanics
Optimal Transport (OT) addresses the geometric comparison of probability measures:
1. **Monge Problem (Deterministic Transport Map):**
   Sought a bijective mapping $T: \mathcal{X} \to \mathcal{Y}$ minimizing $\int c(x, T(x)) d\mu(x)$. Frequently has no solution when mass must split (e.g., one pile divided into two holes).
2. **Kantorovich Relaxation (Joint Coupling Plan):**
   Relaxes deterministic maps to a joint probability distribution (coupling) $\gamma \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})$ with marginals constrained to $p$ and $q$:
   $$\Pi(p, q) = \left\{ P \in \mathbb{R}_+^{n \times m} \;\middle|\; P \mathbf{1}_m = p, \; P^T \mathbf{1}_n = q \right\}$$
3. **1D Closed-Form Quantile Equivalence:**
   In one dimension, optimal transport has an exact closed-form solution: sorting both empirical samples and computing the $L_p$ distance between their sorted orders (quantile functions).

## 3. The Hardcore Mathematics

### Discrete Kantorovich Linear Program
Given discrete probability vectors $p \in \Delta^n, q \in \Delta^m$ and ground metric cost matrix $C \in \mathbb{R}_+^{n \times m}$ where $C_{ij} = d(x_i, y_j)$:
$$W_1(p, q) = \min_{P \in \Pi(p, q)} \langle C, P \rangle_F = \min_{P \in \Pi(p, q)} \sum_{i=1}^n \sum_{j=1}^m C_{ij} P_{ij}$$
subject to:
$$\sum_{j=1}^m P_{ij} = p_i \quad \forall i, \qquad \sum_{i=1}^n P_{ij} = q_j \quad \forall j, \qquad P_{ij} \ge 0$$

### Kantorovich-Rubinstein Duality
By linear programming duality, the primal transportation problem is equivalent to maximizing the difference between dual potentials $\alpha \in \mathbb{R}^n$ and $\beta \in \mathbb{R}^m$:
$$W_1(p, q) = \max_{\alpha, \beta} \left( \alpha^T p + \beta^T q \right) \quad \text{s.t.} \quad \alpha_i + \beta_j \le C_{ij}$$
When cost metric $C_{ij} = \|x_i - y_j\|$, this reduces to taking the supremum over 1-Lipschitz continuous functions $f \in \text{Lip}_1$:
$$W_1(p, q) = \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)] \right)$$
This exact dual theorem serves as the foundational objective of the **Wasserstein GAN (WGAN)**, where discriminator $f$ is constrained via gradient penalty or weight clipping to remain 1-Lipschitz.

### 1D Empirical Quantile Theorem
Let $X = \{x_1, \dots, x_N\}$ and $Y = \{y_1, \dots, y_N\}$ be 1D samples sorted ascendingly $x_{(1)} \le x_{(2)} \le \dots \le x_{(N)}$ and $y_{(1)} \le y_{(2)} \le \dots \le y_{(N)}$.
The $p$-Wasserstein distance is exact:
$$W_p(X, Y) = \left( \frac{1}{N} \sum_{i=1}^N \left| x_{(i)} - y_{(i)} \right|^p \right)^{1/p}$$

## 4. The Algorithm Step-by-Step
1. **1D Wasserstein:**
   - Sort input vectors $X$ and $Y$ in $O(N \log N)$ time.
   - Return mean absolute difference between sorted coordinate arrays.
2. **Multi-Dimensional Discrete Optimal Transport:**
   - Compute pairwise ground distance matrix $C_{ij} = \|x_i - y_j\|_2$.
   - Solve the Kantorovich linear program via dual coordinate ascent or transportation linear programming.
   - Extract optimal transport coupling matrix $P^*$ and compute total work $\sum_{ij} P^*_{ij} C_{ij}$.

## 5. Limitations & Failure Modes
- **Computational Complexity:** Solving exact linear programs scales cubically $O(N^3 \log N)$, making exact Kantorovich OT intractable for datasets with $N > 10,000$ points (motivating the Sinkhorn entropic regularization in O02).
- **Curse of Dimensionality in Sample Complexity:** Estimating continuous Wasserstein distances from finite samples requires $O(N^{-1/D})$ empirical samples, which degrades exponentially in high dimensions.

## 6. Where This Fits
Cross-reference: Wasserstein distance provides geometrically meaningful loss gradients when distributions do not share overlapping support, resolving mode collapse and vanishing gradients in GANs and modern continuous Flow Matching.

# O02: Entropic Regularized Optimal Transport (Sinkhorn-Knopp Algorithm)

## 1. The Beginner's Analogy
Imagine the transportation problem from O01, but now you have a strict deadline. Solving the exact linear program takes days ($O(N^3)$) because you have to make hard, all-or-nothing binary decisions about which specific shovelful of dirt goes into which specific hole.
**Entropic Regularization (The Sinkhorn Algorithm)** injects a puff of random "thermal fog" (entropy $\varepsilon$) into the problem:
- Instead of forcing strict deterministic assignments, it allows the dirt to spread out smoothly like warm gas.
- Suddenly, the brutal combinatorial linear program transforms into an incredibly elegant problem of **rescaling rows and columns of a matrix** to match the desired totals!
- In just a few alternating vectorized matrix multiplications ($O(N^2)$), the algorithm converges to the optimal transport plan millions of times faster than standard linear programming.

## 2. The Deep Architecture & Mechanics
Marco Cuturi's 2013 breakthrough introduced entropic regularization to computational optimal transport:
1. **Regularized Primal Objective:**
   Adds an entropy penalty $- \varepsilon H(P)$ to the Kantorovich objective:
   $$\min_{P \in \Pi(p, q)} \sum_{i,j} P_{ij} C_{ij} + \varepsilon \sum_{i,j} P_{ij} \left( \ln P_{ij} - 1 \right)$$
2. **Gibbs Kernel Decomposition:**
   The regularized problem is strictly convex. By the Karush-Kuhn-Tucker (KKT) conditions, the unique optimal coupling has the diagonal scaling form:
   $$P^* = \text{diag}(u) K \text{diag}(v), \quad \text{where } K_{ij} = \exp\left(-\frac{C_{ij}}{\varepsilon}\right)$$
3. **Sinkhorn-Knopp Iterations:**
   Satisfying the marginal constraints $P \mathbf{1} = p$ and $P^T \mathbf{1} = q$ simplifies to alternating updates:
   $$u^{(t+1)} = \frac{p}{K v^{(t)}}, \qquad v^{(t+1)} = \frac{q}{K^T u^{(t+1)}}$$
   These are pure matrix-vector multiplications, perfectly suited for vectorized GPU execution.

## 3. The Hardcore Mathematics

### Lagrange Multipliers & Gibbs Formulation
Form the Lagrangian with dual variables $\alpha \in \mathbb{R}^n, \beta \in \mathbb{R}^m$:
$$\mathcal{L}(P, \alpha, \beta) = \sum_{i,j} P_{ij} C_{ij} + \varepsilon P_{ij} (\ln P_{ij} - 1) - \alpha^T (P \mathbf{1} - p) - \beta^T (P^T \mathbf{1} - q)$$
Setting the stationary condition $\frac{\partial \mathcal{L}}{\partial P_{ij}} = 0$:
$$C_{ij} + \varepsilon \ln P_{ij} - \alpha_i - \beta_j = 0$$
Solving for $P_{ij}$:
$$P_{ij} = \exp\left( \frac{\alpha_i}{\varepsilon} \right) \exp\left( -\frac{C_{ij}}{\varepsilon} \right) \exp\left( \frac{\beta_j}{\varepsilon} \right)$$
Defining scaling vectors $u_i = \exp(\alpha_i / \varepsilon)$, $v_j = \exp(\beta_j / \varepsilon)$, and Gibbs kernel $K_{ij} = \exp(-C_{ij} / \varepsilon)$:
$$P_{ij} = u_i K_{ij} v_j$$

### Marginal Satisfaction Equations
Imposing marginal constraints:
1. Row constraint: $\sum_j P_{ij} = u_i \sum_j K_{ij} v_j = p_i \implies u_i = \frac{p_i}{(K v)_i}$
2. Column constraint: $\sum_i P_{ij} = v_j \sum_i K_{ij} u_i = q_j \implies v_j = \frac{q_j}{(K^T u)_j}$

### Log-Domain Stabilization (Log-Sum-Exp)
For small $\varepsilon \to 0$, $K_{ij} = \exp(-C_{ij} / \varepsilon)$ underflows to numerical zero in float64. 
Stabilization performs updates in the dual log-domain:
$$\alpha_i^{(t+1)} = -\varepsilon \ln \left( \sum_j \exp\left( \frac{\beta_j^{(t)} - C_{ij}}{\varepsilon} \right) \right) + \varepsilon \ln(p_i)$$
$$\beta_j^{(t+1)} = -\varepsilon \ln \left( \sum_i \exp\left( \frac{\alpha_i^{(t+1)} - C_{ij}}{\varepsilon} \right) \right) + \varepsilon \ln(q_j)$$
using the identity $\ln \sum \exp(z) = z_{\max} + \ln \sum \exp(z - z_{\max})$.

## 4. The Algorithm Step-by-Step
1. Compute pairwise distance cost matrix $C \in \mathbb{R}_+^{n \times m}$.
2. Form Gibbs kernel $K = \exp(-C / \varepsilon)$.
3. Initialize scaling vector $v = \mathbf{1}_m / m$.
4. Repeat until convergence $\|u \odot (K v) - p\|_1 < \text{tol}$:
   - $u \leftarrow p / (K v)$
   - $v \leftarrow q / (K^T u)$
5. Construct optimal coupling $P = \text{diag}(u) K \text{diag}(v)$ and return transport cost $\langle P, C \rangle_F$.

## 5. Limitations & Failure Modes
- **Entropic Blur:** Large values of $\varepsilon$ diffuse mass across all targets rather than concentrating on sparse, sharp optimal matches.
- **Numerical Instability at Low Epsilon:** As $\varepsilon \to 0$ to approach exact unregularized OT, standard scaling matrices underflow without log-sum-exp stabilization.

## 6. Where This Fits
Cross-reference: Sinkhorn entropic OT is the computational engine behind Wasserstein loss layers in neural networks, point-cloud registration, single-cell biological trajectory inference, and modern continuous-time Flow Matching models.

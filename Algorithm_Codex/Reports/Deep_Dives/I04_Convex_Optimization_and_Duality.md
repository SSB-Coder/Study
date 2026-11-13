# Deep Dive: Convex Optimization & Projected Gradient Descent (PGD)

## 1. The Beginner's Analogy: Playing Basketball Inside a Fence
Imagine bouncing a basketball:
- You want to bounce the ball down to the lowest floor.
- But your playing court is enclosed inside a boundary fence (the convex set $\mathcal{C}$).
- When you throw the ball toward the lowest point outside the fence, it hits the fence and bounces back.
- **Projected Gradient Descent (PGD)** takes a standard downhill gradient step. If the new position lands outside the fence in forbidden territory, it instantly snaps (projects) the ball back to the closest legal point on the boundary of the court.
- Because the court is **convex** (no inward dents or cave-ins), the closest point is always unique and easy to find!

---

## 2. The Deep Architecture/Algorithm Breakdown
Convex optimization studies problems where the objective $f(x)$ is a convex function and the feasible domain $\mathcal{C}$ is a convex set:
1. **Convexity Invariant:** Any local minimum is guaranteed to be a global minimum.
2. **The Projection Operator $\Pi_\mathcal{C}(v)$:** Orthogonal projection onto convex set $\mathcal{C}$:
   $$\Pi_\mathcal{C}(v) = \arg\min_{x \in \mathcal{C}} \frac{1}{2} \|x - v\|_2^2$$
3. **Projected Gradient Descent:** Iterative algorithm alternating between unconstrained gradient descent and set projection:
   $$x_{t+1} = \Pi_\mathcal{C}\left( x_t - \eta \nabla f(x_t) \right)$$
4. **Common Convex Constraint Sets:**
   - $\ell_2$ Ball: $\mathcal{C} = \{x \mid \|x\|_2 \le R\} \implies \Pi(v) = \frac{v}{\max(1, \|v\|_2 / R)}$.
   - Non-Negative Orthant: $\mathcal{C} = \{x \mid x \ge 0\} \implies \Pi(v) = \max(0, v)$.
   - Probability Simplex: $\mathcal{C} = \{x \mid \sum x_i = 1, x_i \ge 0\}$ (Duchi et al., 2008).

---

## 3. The Hardcore Mathematics

### 3.1 Convergence of PGD
If $f$ is convex with $L$-Lipschitz continuous gradient ($\nabla f$ is $L$-Lipschitz), for step size $\eta \le 1/L$:
$$f(x_T) - f(x^*) \le \frac{\|x_0 - x^*\|_2^2}{2 \eta T} = O\left( \frac{1}{T} \right)$$

### 3.2 Projection onto the Probability Simplex
To project vector $v \in \mathbb{R}^d$ onto the simplex $\Delta^d = \{x \ge 0 \mid \mathbf{1}^T x = 1\}$:
$$\Pi_{\Delta}(v) = \max(0, v - \theta)$$
Where threshold $\theta$ is the unique scalar satisfying:
$$\sum_{j=1}^d \max(0, v_j - \theta) = 1$$
Found in $O(d \log d)$ time by sorting components of $v$ descending and finding the pivot threshold index $\rho$.

---

## 4. The Algorithm Step-by-Step
1. **Gradient Step:** $y_{t+1} = x_t - \eta \nabla f(x_t)$.
2. **Snap to Convex Domain:** $x_{t+1} = \Pi_\mathcal{C}(y_{t+1})$.
3. **Convergence:** Check if $\|x_{t+1} - x_t\| < \epsilon$.

---

## 5. Limitations / When It Breaks
- **Complex Constraint Geometries:** If the constraint set has no closed-form projection operator, evaluating $\Pi_\mathcal{C}$ requires an inner numerical optimization loop.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/32_Distributed_Training_and_ZeRO.md`. PGD on $\ell_\infty$ and $\ell_2$ balls is the standard algorithm used to generate adversarial prompt injections and evaluate LLM robustness against jailbreak attacks.

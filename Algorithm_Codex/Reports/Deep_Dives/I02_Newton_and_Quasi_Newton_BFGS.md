# Deep Dive: Newton's Method & Quasi-Newton Optimization (BFGS / L-BFGS)

## 1. The Beginner's Analogy: Driving with Accelerometer vs. Looking Through the Windshield
Imagine parking a heavy car:
- Gradient descent looks only at your current speed (first derivative). If you are moving fast toward a wall, it doesn't know whether the terrain is leveling out or plunging down a cliff.
- **Newton's Method** measures both speed *and* acceleration (the curvature / Hessian matrix). It approximates the road ahead as a smooth quadratic parabola and leaps directly to the bottom of the bowl in a single step if the landscape is quadratic.
- **The Problem:** Computing and inverting the full $d \times d$ curvature matrix (Hessian) takes $O(d^3)$ time and explodes in memory.
- **BFGS (Quasi-Newton):** Instead of calculating the real curvature from scratch at every step, BFGS looks at how the slope changed between your last step and your current step, building an increasingly accurate approximation of the inverse Hessian matrix using simple rank-2 vector outer products!

---

## 2. The Deep Architecture/Algorithm Breakdown
Second-order optimization models the objective function $\mathcal{L}(x)$ via its second-order Taylor expansion:
$$\mathcal{L}(x + \Delta x) \approx \mathcal{L}(x) + g^T \Delta x + \frac{1}{2} \Delta x^T H \Delta x$$
Where $g = \nabla \mathcal{L}(x)$ is the gradient and $H = \nabla^2 \mathcal{L}(x)$ is the Hessian matrix.
1. **Newton-Raphson Step:** Setting the gradient of the quadratic approximation to zero yields the classic Newton step:
   $$\Delta x = -H^{-1} g$$
2. **Quadratic Convergence:** Achieves quadratic local convergence rate near strict local minima: $\|x_{t+1} - x^*\| \le M \|x_t - x^*\|^2$.
3. **The Secant Equation:** Approximates the Hessian action using step difference $s_k = x_{k+1} - x_k$ and gradient difference $y_k = g_{k+1} - g_k$:
   $$B_{k+1} s_k = y_k \iff H_{k+1} y_k = s_k$$
4. **BFGS Rank-2 Inverse Hessian Update:** Directly maintains and updates the inverse Hessian estimate $H_k \approx B_k^{-1}$ without ever solving a matrix inversion.

---

## 3. The Hardcore Mathematics

### 3.1 The BFGS Update Formula
Let $s_k = x_{k+1} - x_k$ and $y_k = g_{k+1} - g_k$.
Define curvature scalar $\rho_k = \frac{1}{y_k^T s_k}$.
The Sherman-Morrison-Woodbury theorem gives the rank-2 update for the inverse Hessian approximation $H_{k+1}$:
$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$
Properties:
- If $y_k^T s_k > 0$ (the curvature condition, guaranteed by Wolfe line search), $H_{k+1}$ remains strictly positive-definite if $H_k$ is positive-definite.
- Computation per iteration drops from $O(d^3)$ to $O(d^2)$ matrix-vector multiplications.

### 3.2 Armijo-Goldstein Backtracking Line Search
To guarantee global convergence from distant starting points, step length $\alpha_k$ is adjusted to satisfy the sufficient decrease condition:
$$\mathcal{L}(x_k + \alpha_k p_k) \le \mathcal{L}(x_k) + c_1 \alpha_k g_k^T p_k$$
Where $p_k = -H_k g_k$ is the search direction and $c_1 \in (0, 1)$ (typically $10^{-4}$).

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Choose starting point $x_0$, initial inverse Hessian $H_0 = I$. Compute gradient $g_0 = \nabla \mathcal{L}(x_0)$.
2. **Iterate $k = 0, 1, \dots$:**
   - Compute search direction $p_k = -H_k g_k$.
   - Perform backtracking line search to find step size $\alpha_k$ satisfying Armijo condition.
   - Update point: $x_{k+1} = x_k + \alpha_k p_k$.
   - Compute step vector $s_k = x_{k+1} - x_k$ and new gradient $g_{k+1}$.
   - Compute gradient change $y_k = g_{k+1} - g_k$.
   - If $y_k^T s_k > 10^{-10}$, update $H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$.
3. **Stop:** When $\|g_k\| < \text{tol}$.

---

## 5. Limitations / When It Breaks
- **Non-Convex Saddle Points:** True Newton's method can leap directly toward local maxima or saddle points where $H$ is negative-definite or indefinite.
- **Memory Scaling:** Maintaining $H_k \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ RAM, which is impossible for models with 7B+ parameters.
- **Historical Fix:** L-BFGS (Limited-memory BFGS) stores only the last $m \approx 10$ vectors $(s_k, y_k)$ and applies two-loop recursion in $O(m \cdot d)$ space.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/32_Distributed_Training_and_ZeRO.md`. Second-order curvature estimation underlies modern specialized deep learning optimizers like K-FAC, Sophia, and Muon.

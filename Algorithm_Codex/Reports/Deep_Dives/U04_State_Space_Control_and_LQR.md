# Deep Dive: U04 — State-Space Control & the Linear-Quadratic Regulator (LQR)

## 1. The Beginner's Analogy: Balancing a Broom on Your Fingertip
Imagine balancing a vertical broomstick on the palm of your open hand:
- The broom has four coupled variables at once: hand position, hand velocity, broom tilt angle, and broom angular speed.
- A single PID loop cannot coordinate all four variables without tripping over itself.
- **State-Space Control** treats all four variables as a single vector $\mathbf{x} = [p, v, \theta, \omega]^T$.
- **The Linear-Quadratic Regulator (LQR)** is a mathematical referee that balances two competing desires:
  1. "How badly do I want the broom perfectly upright?" (Matrix $Q$).
  2. "How much battery/motor power am I willing to spend shaking my hand?" (Matrix $R$).
- By solving the Riccati equation, LQR produces the mathematically optimal gain matrix $K$ that stabilizes the broom with the absolute minimum control energy.

---

## 2. The Deep Architecture/Algorithm Breakdown
Linear state-space control models multi-input multi-output (MIMO) dynamic systems:
1. **Discrete Dynamics:**
   $$\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k$$
2. **Infinite-Horizon Quadratic Cost:**
   $$J = \sum_{k=0}^\infty \left( \mathbf{x}_k^T Q \mathbf{x}_k + \mathbf{u}_k^T R \mathbf{u}_k \right)$$
   where $Q \succeq 0$ penalizes state deviations and $R \succ 0$ penalizes control effort.
3. **The Control/Estimation Duality:**
   LQR control and the Kalman Filter (E06, X04) are exact mathematical **duals**:
   | Control (LQR) | Estimation (Kalman Filter) |
   | :--- | :--- |
   | State transition $A$ | State transition $A^T$ |
   | Input matrix $B$ | Measurement matrix $C^T$ |
   | Control weight $R$ | Measurement noise covariance $R_v$ |
   | State weight $Q$ | Process noise covariance $Q_w$ |
   | Control gain $K$ | Kalman gain $L^T$ |
   Both solve the identical **Algebraic Riccati Equation**! Together, they form the complete Linear-Quadratic-Gaussian (LQG) regulator.

---

## 3. The Hardcore Mathematics

### 3.1 Discrete Algebraic Riccati Equation (DARE)
Using dynamic programming and Bellman's principle of optimality, cost-to-go is $V(\mathbf{x}) = \mathbf{x}^T P \mathbf{x}$. $P$ satisfies:
$$P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A$$

### 3.2 Optimal State-Feedback Law
$$\mathbf{u}_k^* = -K \mathbf{x}_k, \quad K = (R + B^T P B)^{-1} B^T P A$$
The resulting autonomous closed-loop system is:
$$\mathbf{x}_{k+1} = (A - B K) \mathbf{x}_k$$
All eigenvalues of $(A - BK)$ are guaranteed to lie strictly inside the complex unit circle ($|\lambda_i| < 1$).

---

## 4. The Algorithm Step-by-Step
1. **Model Linearization:** Form matrices $A = \partial f / \partial \mathbf{x}$ and $B = \partial f / \partial \mathbf{u}$.
2. **Check Controllability:** Verify controllability matrix $\mathcal{C} = [B, AB, \dots, A^{n-1} B]$ has full rank $n$.
3. **Solve DARE:** Iterate Riccati equation until $\|P_{k+1} - P_k\| < \text{tol}$.
4. **Compute Gain:** $K = (R + B^T P B)^{-1} B^T P A$.
5. **Execute:** At every time step, measure state $\mathbf{x}_k$ and apply $\mathbf{u}_k = -K \mathbf{x}_k$.

---

## 5. Limitations / When It Breaks
- **Unmeasured States:** Requires measuring full state vector $\mathbf{x}$. In practice, sensors measure only outputs $\mathbf{y} = C \mathbf{x}$, requiring a Luenberger observer or Kalman Filter.
- **Constraints:** LQR cannot handle hard state/input constraints (e.g. max motor voltage, wall obstacles). Model Predictive Control (MPC) solves constrained LQR over a receding horizon.

---

## 6. Where This Fits
- **Cross-Reference Kalman Filter (E06, X04):** Deepens the foundational duality of modern control theory.

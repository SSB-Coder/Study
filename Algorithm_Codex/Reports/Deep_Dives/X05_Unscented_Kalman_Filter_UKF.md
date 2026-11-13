# Deep Dive: X05 — Unscented Kalman Filter (UKF)

## 1. The Beginner's Analogy: Five Scouts on the Ridge
Imagine pushing a Gaussian cloud of fog through a violently swirling canyon:
- **EKF (X04)** computes the slope right at the exact center of the fog bank and pretends the canyon floor is a flat plywood ramp. If the canyon bends into a sharp funnel, this flat-plane guess flies completely off track.
- **The Unscented Kalman Filter (UKF)** says: *"It is easier to approximate a probability distribution than it is to approximate an arbitrary nonlinear function."*
- Instead of computing derivatives, UKF plants five scouts (Sigma Points) across the fog bank: one in the center, and two along each main axis.
- It marches each scout through the true canyon physics, observes where they land, and fits a fresh Gaussian ellipse to the scouts. It captures 3rd-order nonlinear curvature with zero Jacobian calculus!

---

## 2. The Deep Architecture/Algorithm Breakdown
Julier and Uhlmann (1997) formulated the Unscented Transform (UT):
1. **Sigma Point Sampling:** For state dimension $n$, construct $2n + 1$ deterministic sigma points $\chi_i$:
   $$\chi_0 = \bar{\mathbf{x}}$$
   $$\chi_i = \bar{\mathbf{x}} + \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \dots, n$$
   $$\chi_{n+i} = \bar{\mathbf{x}} - \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \dots, n$$
   where matrix square root is computed via Cholesky decomposition.
2. **Nonlinear Propagation:** Each sigma point is evaluated through the exact functions: $\mathcal{Y}_i = f(\chi_i)$.
3. **Statistical Accuracy:** Captures mean and covariance up to the 3rd order of Taylor series for Gaussian distributions (compared to only 1st-order in EKF).

---

## 3. The Hardcore Mathematics

### 3.1 Scaling & Weight Parameters
$$\lambda = \alpha^2 (n + \kappa) - n$$
$$W_0^{(m)} = \frac{\lambda}{n + \lambda}, \quad W_0^{(c)} = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)$$
$$W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad i = 1, \dots, 2n$$

### 3.2 Mean and Covariance Reconstruction
$$\hat{\mathbf{x}} = \sum_{i=0}^{2n} W_i^{(m)} \mathcal{Y}_i$$
$$P = \sum_{i=0}^{2n} W_i^{(c)} (\mathcal{Y}_i - \hat{\mathbf{x}}) (\mathcal{Y}_i - \hat{\mathbf{x}})^T + Q$$

---

## 4. The Algorithm Step-by-Step
1. **Generate Sigmas:** Compute Cholesky factor $L = \text{chol}(P)$. Form $2n+1$ points.
2. **Predict:** Propagate sigmas through $f(\chi_i, u)$. Reconstruct $\hat{\mathbf{x}}$ and $P$.
3. **Observe:** Propagate updated sigmas through $h(\chi_i)$. Reconstruct predicted measurement $\hat{\mathbf{z}}$, innovation covariance $S$, and cross-covariance $P_{xz}$.
4. **Kalman Gain:** $K = P_{xz} S^{-1}$. Update state and covariance.

---

## 5. Limitations / When It Breaks
- **Cholesky Breakdown:** Numerical roundoff can cause $P$ to lose positive-definiteness, crashing `np.linalg.cholesky`. Square-root UKF (SR-UKF) maintains Cholesky factor directly via QR updates (V01).

---

## 6. Where This Fits
- **Autonomous Drones & Spacecraft:** UKF is the preferred estimator for quaternion attitude dynamics and highly nonlinear re-entry trajectories.

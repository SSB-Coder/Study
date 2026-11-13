# Deep Dive: State-Space Estimation (Kalman Filters & Particle Filters)

## 1. The Beginner's Analogy: Tracking a Submarine in Murky Waters
Imagine tracking a submarine navigating underwater in dense fog:
- The submarine has a speedometer and rudder, so physics equations predict where it *should* move (the Motion Model). But ocean currents introduce unpredictable drift.
- A sonar ping bounces back from the hull, but thermal layers in the ocean add heavy acoustic static (the Measurement Model).
- **The Kalman Filter** maintains a Gaussian bell-curve of the submarine's location. When the sonar pings, it calculates a **Kalman Gain**—a weighted compromise between the physics prediction and the noisy ping—sharpening the uncertainty ellipse into a tight, accurate estimate.
- **The Particle Filter** comes to the rescue when the ocean has jagged coral reefs and the submarine can suddenly bounce off walls (non-linear, multi-modal physics). Instead of a single bell curve, it sprinkles 1,000 tiny virtual toy submarines across the map. Toy submarines that match the sonar ping survive and duplicate; toy submarines that clash with the ping die off.

---

## 2. The Deep Architecture/Algorithm Breakdown
State-space estimation tracks hidden continuous state $x_t \in \mathbb{R}^d$ from noisy observations $y_t \in \mathbb{R}^p$:
1. **Linear Gaussian State-Space (Kalman Filter):**
   - State Dynamics: $x_t = F_t x_{t-1} + B_t u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)$
   - Measurement: $y_t = H_t x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)$
   - Optimality: Under linear Gaussian dynamics, the Kalman filter is the provably minimum mean square error (MMSE) estimator.
2. **Predict-Update Cycle:**
   - **Predict:** Projects state mean $\hat{x}_{t|t-1}$ and covariance $P_{t|t-1}$ forward in time.
   - **Update:** Computes Kalman Gain $K_t$ to assimilate observation innovation $y_t - H \hat{x}_{t|t-1}$.
3. **Sequential Importance Resampling (SIR Particle Filter):**
   - Non-parametric recursive Bayes filter representing arbitrary posterior densities using $P$ discrete weighted particles $\{(x_t^{(i)}, w_t^{(i)})\}_{i=1}^P$.
   - **Importance Weighting:** $w_t^{(i)} \propto w_{t-1}^{(i)} p(y_t \mid x_t^{(i)})$.
   - **Resampling:** Solves the particle degeneracy problem by replacing depleted particles with clones of high-weight particles.

---

## 3. The Hardcore Mathematics

### 3.1 The Discrete Kalman Filter Equations
**Time Update (Predict):**
$$\hat{x}_{t|t-1} = F_t \hat{x}_{t-1|t-1} + B_t u_t$$
$$P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q_t$$

**Measurement Update (Correct):**
Innovation covariance:
$$S_t = H_t P_{t|t-1} H_t^T + R_t$$
Optimal Kalman Gain:
$$K_t = P_{t|t-1} H_t^T S_t^{-1}$$
Updated state estimate:
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - H_t \hat{x}_{t|t-1})$$
Updated error covariance:
$$P_{t|t} = (I - K_t H_t) P_{t|t-1}$$

### 3.2 SIR Particle Filter Equations
Given particle set $x_{t-1}^{(i)} \sim p(x_{t-1} \mid y_{1:t-1})$:
1. **Propagate:** $x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})$.
2. **Update Weights:** $w_t^{(i)} = p(y_t \mid x_t^{(i)})$, normalize $\tilde{w}_t^{(i)} = \frac{w_t^{(i)}}{\sum_j w_t^{(j)}}$.
3. **Effective Sample Size:** $N_{\text{eff}} = \frac{1}{\sum_{i=1}^P (\tilde{w}_t^{(i)})^2}$. If $N_{\text{eff}} < P/2$, resample particles with replacement proportional to $\tilde{w}_t^{(i)}$.

---

## 4. The Algorithm Step-by-Step
1. **Kalman Loop:**
   - Compute prior state and covariance via transition matrix $F$.
   - Compute Kalman gain $K = P H^T (H P H^T + R)^{-1}$.
   - Assimilate new observation $y$ to obtain posterior mean and covariance.
2. **Particle Filter Loop:**
   - Shift each particle through stochastic process noise.
   - Compute likelihood of observation given each particle position.
   - Resample particles using systematic universal sampling.

---

## 5. Limitations / When It Breaks
- **Non-Linear Dynamics in Kalman:** Linear Kalman filters diverge on non-linear systems; requires Extended Kalman Filter (EKF via Jacobian Taylor linearization) or Unscented Kalman Filter (UKF via sigma-points).
- **Particle Deprivation:** In particle filters, if all particles land in low-likelihood regions, the filter collapses.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/10_LNN_Liquid.md`. Continuous-time Kalman filtering provides the foundational state-tracking formulation underlying Liquid Neural Networks (LNNs) and neural ODE state transitions.

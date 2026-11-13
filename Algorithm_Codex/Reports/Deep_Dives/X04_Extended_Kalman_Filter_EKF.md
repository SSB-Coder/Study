# Deep Dive: X04 — Extended Kalman Filter (EKF)

## 1. The Beginner's Analogy: The Foggy Ship Captain's Chart
Imagine piloting a ship through dense fog with a compass, engine RPM log, and an occasional radar flash off a distant lighthouse:
- The classical **Linear Kalman Filter** assumes the ship moves along a flat runway where everything is a straight line.
- But real ships turn in circles ($\sin \theta, \cos \theta$) and radar measures curved distance and angle ($\sqrt{\Delta x^2 + \Delta y^2}, 	ext{atan2}$).
- **The Extended Kalman Filter (EKF)** takes your best guess of where the ship is right now, draws a flat magnifying glass over that immediate spot (first-order Taylor series Jacobian), and applies the Kalman math locally. Even though the global world is curved, every tiny step under the magnifying glass is nearly linear!

---

## 2. The Deep Architecture/Algorithm Breakdown
The Extended Kalman Filter generalizes Kalman filtering to nonlinear dynamic systems:
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, Q)$$
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(0, R)$$
where $f$ and $h$ are non-linear vector functions.

### 2.1 First-Order Linearization
Instead of computing the exact non-Gaussian probability distribution, EKF linearizes $f$ and $h$ around current state estimates using Jacobian matrices:
$$F_k = \left. \frac{\partial f}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k}, \quad H_k = \left. \frac{\partial h}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_{k|k-1}}$$

---

## 3. The Hardcore Mathematics

### 3.1 Predict Step
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q$$

### 3.2 Update Step
$$\tilde{\mathbf{y}}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$S_k = H_k P_{k|k-1} H_k^T + R$$
$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k \tilde{\mathbf{y}}_k$$
$$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R K_k^T \quad \text{(Joseph form)}$$

---

## 4. The Algorithm Step-by-Step
1. **Predict:** Propagate state through nonlinear $f(\mathbf{x}, \mathbf{u})$. Compute Jacobian $F$. Propagate covariance $P = FPF^T + Q$.
2. **Measurement:** Predict measurement $h(\hat{\mathbf{x}})$. Compute innovation $\tilde{\mathbf{y}} = \mathbf{z} - h(\hat{\mathbf{x}})$.
3. **Linearize Measurement:** Compute Jacobian $H = \partial h / \partial \mathbf{x}$.
4. **Gain & Fuse:** Compute Kalman gain $K = P H^T (H P H^T + R)^{-1}$. Update $\mathbf{x}$ and $P$.

---

## 5. Limitations / When It Breaks
- **Severe Non-Linearities:** First-order Taylor truncation fails if the uncertainty cloud is large compared to the curvature of $f$ or $h$, causing divergence. The Unscented Kalman Filter (X05) eliminates Jacobians entirely.

---

## 6. Where This Fits
- **Cross-Reference Kalman Filters (E06):** Direct nonlinear extension of linear KF.
- **EKF-SLAM (X06):** EKF extended to estimate both robot pose and environmental map landmarks.

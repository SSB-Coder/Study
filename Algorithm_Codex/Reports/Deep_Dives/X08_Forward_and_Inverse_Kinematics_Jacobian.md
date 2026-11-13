# Deep Dive: X08 — Forward & Inverse Kinematics (Jacobian IK)

## 1. The Beginner's Analogy: Reaching for a Coffee Cup
Imagine moving your hand to pick up a coffee cup on your desk:
- **Forward Kinematics (FK)** is what happens naturally: if your brain knows your shoulder angle ($30^\circ$), elbow angle ($45^\circ$), and wrist angle ($10^\circ$), basic geometry easily computes exactly where your fingertip lands.
- **Inverse Kinematics (IK)** is the real problem your nervous system solves every second: you see the coffee mug at $(x, y, z)$, and you must calculate the exact set of muscle and joint angles to put your hand on the handle.
- **The Jacobian** measures how much your hand moves in $(x, y)$ when you nudge each individual joint. By inverting this sensitivity matrix, the arm takes small steps straight toward the cup.

---

## 2. The Deep Architecture/Algorithm Breakdown
Kinematics models robotic manipulators without considering dynamic forces:
1. **Forward Kinematics (FK):**
   $$\mathbf{x} = f(\boldsymbol{\theta})$$
   Maps joint configuration space $\mathcal{Q} \subset \mathbb{R}^n$ to task workspace $\mathcal{W} \subset \mathbb{R}^m$.
2. **Manipulator Jacobian:**
   $$J(\boldsymbol{\theta}) = \frac{\partial f}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{m \times n}$$
   Relates joint velocities to end-effector Cartesian velocities: $\dot{\mathbf{x}} = J(\boldsymbol{\theta}) \dot{\boldsymbol{\theta}}$.
3. **Inverse Kinematics (IK):** Inverting the nonlinear map $\boldsymbol{\theta} = f^{-1}(\mathbf{x}^*)$. Because closed-form analytic solutions exist only for simple geometric arms (Pieper's criterion), general manipulators use numerical iterative Newton-Raphson solvers.

---

## 3. The Hardcore Mathematics

### 3.1 Kinematic Jacobian Formulation
For a 2D planar arm with joint origins $\mathbf{p}_i$ and end-effector $\mathbf{p}_{ee}$:
$$J_i = \mathbf{z}_i \times (\mathbf{p}_{ee} - \mathbf{p}_i) = \begin{bmatrix} -(y_{ee} - y_i) \\ (x_{ee} - x_i) \end{bmatrix}$$

### 3.2 Damped Least Squares (Levenberg-Marquardt IK)
Near kinematic singularities (e.g. arm fully outstretched where $\det(J J^T) \to 0$), the Moore-Penrose pseudoinverse $J^\dagger = J^T (J J^T)^{-1}$ blows up to infinite velocities.
**Damped Least Squares (DLS)** introduces Tikhonov regularization $\lambda$:
$$\Delta \boldsymbol{\theta} = J^T \left( J J^T + \lambda^2 I \right)^{-1} \Delta \mathbf{x}$$
This guarantees bounded joint velocities across all singular configurations.

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Start with current joint configuration $\boldsymbol{\theta}_0$.
2. **Loop:**
   - Compute current end-effector position $\mathbf{x} = \text{FK}(\boldsymbol{\theta})$.
   - Compute tracking error $\mathbf{e} = \mathbf{x}^* - \mathbf{x}$. If $\|\mathbf{e}\| < \epsilon$, return.
   - Assemble Jacobian $J(\boldsymbol{\theta})$.
   - Compute joint increment $\Delta \boldsymbol{\theta} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}$.
   - Update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \Delta \boldsymbol{\theta}$.

---

## 5. Limitations / When It Breaks
- **Redundancy & Local Minima:** When $n > m$ (redundant arm), infinitely many joint configurations reach the same $(x, y)$. The solver can get trapped in local minimum configurations that collide with the robot's own body or joint limits. Nullspace projection ($P_{\text{null}} = I - J^\dagger J$) resolves redundancy.

---

## 6. Where This Fits
- **Cross-Reference Newton-Raphson (I02, V05):** Jacobian IK is Newton's method applied to geometric kinematic chains.

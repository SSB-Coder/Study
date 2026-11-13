# K05: Physics-Informed Neural Networks (PINNs)

## 1. The Beginner's Analogy
Suppose you are teaching a student to predict planetary orbits. If you show them telescope photos, a naive neural network might learn to fit the dots, but predict that Mars will suddenly reverse direction next Tuesday because it has no concept of gravity or momentum.
A **Physics-Informed Neural Network (PINN)** does not just fit data. It embeds physical differential equations (like Newton's laws or Navier-Stokes fluid mechanics) directly into the neural network's loss function. If the network proposes an impossible orbit that violates conservation of energy, the physics loss penalizes it severely.

## 2. The Deep Architecture & Mechanics
PINNs bridge purely empirical deep learning with mathematical physics:
1. **Neural Representation:** A neural network $\hat{u}_\theta(t, x)$ acts as a universal function approximator for the continuous physical field.
2. **Automatic/Analytic Differentiation:** Derivatives of the network output with respect to spatio-temporal inputs (e.g. $\frac{\partial u}{\partial t}$, $\frac{\partial^2 u}{\partial x^2}$) are computed exactly via the chain rule.
3. **Collocation Points:** The model evaluates physical residual equations at thousands of un-annotated spatio-temporal coordinates $\{(t_i, x_i)\}$ across the domain without requiring experimental ground truth measurements at those points.

## 3. The Hardcore Mathematics

### General Non-Linear PDE Formulation
Let a physical system be governed by:
$$f\left(x, t, u, \frac{\partial u}{\partial t}, \nabla u, \nabla^2 u\right) = 0, \quad x \in \Omega, \; t \in [0, T]$$
With boundary conditions $\mathcal{B}(u) = 0$ on $\partial \Omega$ and initial conditions $u(x, 0) = g(x)$.

### The Composite PINN Loss
$$\mathcal{L}(\theta) = w_{\text{data}} \mathcal{L}_{\text{data}}(\theta) + w_{\text{BC}} \mathcal{L}_{\text{BC}}(\theta) + w_{\text{PDE}} \mathcal{L}_{\text{PDE}}(\theta)$$
Where:
- Boundary Loss:
  $$\mathcal{L}_{\text{BC}}(\theta) = \frac{1}{N_{\text{BC}}} \sum_{i=1}^{N_{\text{BC}}} \left| \mathcal{B}(\hat{u}_\theta(x_{\text{BC}}^{(i)}, t_{\text{BC}}^{(i)})) \right|^2$$
- Collocation PDE Residual Loss:
  $$\mathcal{L}_{\text{PDE}}(\theta) = \frac{1}{N_f} \sum_{j=1}^{N_f} \left| f\left(x_f^{(j)}, t_f^{(j)}, \hat{u}_\theta, \nabla \hat{u}_\theta, \dots\right) \right|^2$$

## 4. The Algorithm Step-by-Step
1. **Sample Coordinates:** Sample initial/boundary points $\{x_{\text{BC}}\}$ and interior collocation points $\{x_f\}$.
2. **Forward Evaluation:** Pass coordinates through neural network $\hat{u}_\theta$.
3. **Compute Differential Operators:** Differentiate neural outputs with respect to inputs using exact analytical chain rule.
4. **Compute Residual:** Plug derivatives into the governing differential equation.
5. **Backpropagation:** Update neural weights $\theta$ via gradient descent to minimize both data mismatch and physics residual.

## 5. Limitations & Failure Modes
- **Multi-Scale Stiffness:** PDEs with high Reynolds numbers or extreme stiffness exhibit steep gradient landscapes that get trapped in high-frequency local minima.
- **Gradient Pathologies:** Imbalanced gradient magnitudes between data loss and PDE loss require adaptive weight balancing algorithms.

## 6. Where This Fits
Cross-reference: PINNs form the foundation of AI for Science, Neural Operators (FNO), and physics-grounded world models for autonomous physical control.

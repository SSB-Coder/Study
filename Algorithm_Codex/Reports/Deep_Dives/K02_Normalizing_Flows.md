# K02: Normalizing Flows (RealNVP)

## 1. The Beginner's Analogy
Imagine molding clay. You start with a simple, perfectly uniform sphere of clay (a standard Gaussian distribution). To create a detailed, intricate sculpture of a dragon (a complex data distribution), you push, pull, stretch, and bend the clay. Crucially, you never tear the clay or fold it onto itself—every single point on the sphere maps uniquely to a point on the dragon.
A **Normalizing Flow** is this sequence of invertible mathematical stretchings. Because the transformation is bijective and differentiable, you can map simple noise to realistic data (sampling) AND map data back to noise to compute exact probabilities (density estimation).

## 2. The Deep Architecture & Mechanics
Normalizing Flows optimize exact likelihoods by chaining invertible transformations $f = f_K \circ \dots \circ f_1$:
1. **Base Distribution $p_Z(z)$:** A tractable density, typically a standard isotropic Gaussian $\mathcal{N}(0, I)$.
2. **Affine Coupling Layers (RealNVP):** Partitions input $x$ into two halves $(x_1, x_2)$. One half remains unchanged, while the second half undergoes an elementwise affine scaling and translation parameterized by arbitrary neural networks:
   $$y_1 = x_1, \quad y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$$
3. **Triangular Jacobian:** The Jacobian matrix of an affine coupling layer is lower-triangular, making its determinant simply the product of its diagonal elements $\prod_j \exp(s_j(x_1))$.

## 3. The Hardcore Mathematics

### Change-of-Variables Theorem
For a bijective, differentiable mapping $z = f(x)$ with inverse $x = f^{-1}(z)$:
$$p_X(x) = p_Z(f(x)) \cdot |\det J_f(x)|$$
Taking logarithms:
$$\ln p_X(x) = \ln p_Z(f(x)) + \ln |\det J_f(x)|$$
For a composition of $K$ layers $z_K = f_K(f_{K-1}(\dots f_1(x)))$:
$$\ln p_X(x) = \ln p_Z(z_K) + \sum_{k=1}^K \ln |\det J_{f_k}(z_{k-1})|$$

### RealNVP Jacobian Determinant
The Jacobian of the forward transformation has block structure:
$$J = \begin{bmatrix} I & 0 \\ \frac{\partial y_2}{\partial x_1} & \text{diag}(\exp(s(x_1))) \end{bmatrix}$$
Determinant:
$$\det J = \det(I) \cdot \det(\text{diag}(\exp(s(x_1)))) = \prod_{j=1}^{d_2} \exp(s_j(x_1))$$
Log-determinant:
$$\ln |\det J| = \sum_{j=1}^{d_2} s_j(x_1)$$
Notice that the scaling and translation networks $s(\cdot)$ and $t(\cdot)$ do NOT need to be invertible themselves! They can be arbitrary deep neural networks.

## 4. The Algorithm Step-by-Step
1. **Forward Mapping (Sampling):**
   - Sample base noise $z \sim \mathcal{N}(0, I)$.
   - Pass $z$ through inverse layers $x = f_1^{-1}(f_2^{-1}(\dots f_K^{-1}(z)))$ to generate sample $x$.
2. **Inference (Exact Density):**
   - Pass sample $x$ through forward layers $z = f(x)$.
   - Accumulate scale outputs $\sum \ln |\det J|$.
   - Compute exact log-likelihood $\ln p_X(x) = \ln \mathcal{N}(z; 0, I) + \sum \ln |\det J|$.
3. **Training:** Maximize exact log-likelihood over training dataset via gradient ascent.

## 5. Limitations & Failure Modes
- **Topological Invariance:** Because the mapping is a homeomorphism, normalizing flows cannot change the topology of the base space (e.g. mapping connected Gaussian to disconnected disconnected clusters requires extreme stretching).
- **Dimension Bottleneck:** The latent space must have the exact same dimensionality as the data space.

## 6. Where This Fits
Cross-reference: Continuous Normalizing Flows (CNFs) directly bridge into Neural ODEs, Flow Matching (the modern foundation of Stable Diffusion 3 and Flux), and continuous score-based generative modeling.

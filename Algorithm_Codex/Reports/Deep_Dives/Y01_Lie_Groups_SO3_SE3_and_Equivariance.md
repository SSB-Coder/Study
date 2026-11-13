# Y01: Lie Groups SO(3), SE(3) and Geometric Equivariance

## 1. Mathematical Formulation & Architecture
Spatial orientation and rigid-body mechanics live on smooth differential manifolds possessing continuous group symmetries (**Lie Groups**).

### The Special Orthogonal Group $\text{SO}(3)$
The rotation group consists of all orthogonal $3 \times 3$ matrices with unit determinant:
$$\text{SO}(3) = \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \; \det(R) = 1 \}$$
Its tangent space at the identity is the **Lie Algebra** $\mathfrak{so}(3)$, consisting of $3 \times 3$ skew-symmetric matrices. The isomorphism $\wedge: \mathbb{R}^3 \to \mathfrak{so}(3)$ (**hat operator**) maps angular velocity vectors $\vec{\omega} = (\omega_1, \omega_2, \omega_3)^T$:
$$[\vec{\omega}]_\times = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}$$

The exponential map $\exp: \mathfrak{so}(3) \to \text{SO}(3)$ wraps the tangent vector onto the manifold via **Rodrigues' Rotation Formula**:
$$\theta = \lVert\vec{\omega}\rVert, \quad \exp([\vec{\omega}]_\times) = I + \frac{\sin\theta}{\theta}[\vec{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2}[\vec{\omega}]_\times^2$$

The inverse **logarithmic map** $\log: \text{SO}(3) \to \mathfrak{so}(3)$ extracts axis-angle coordinates:
$$\theta = \arccos\left(\frac{\text{Tr}(R) - 1}{2}\right), \quad [\vec{\omega}]_\times = \frac{\theta}{2\sin\theta}(R - R^T)$$

### The Special Euclidean Group $\text{SE}(3)$
Rigid transformations combine rotations and translations:
$$\text{SE}(3) = \text{SO}(3) \ltimes \mathbb{R}^3 = \left\{ T = \begin{pmatrix} R & \mathbf{p} \\ \mathbf{0}^T & 1 \end{pmatrix} \;\middle|\; R \in \text{SO}(3), \; \mathbf{p} \in \mathbb{R}^3 \right\}$$
Its Lie algebra $\mathfrak{se}(3)$ represents spatial twists $\xi = (\mathbf{v}, \vec{\omega}) \in \mathbb{R}^6$:
$$[\xi]_\wedge = \begin{pmatrix} [\vec{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{pmatrix}, \quad \exp([\xi]_\wedge) = \begin{pmatrix} \exp([\vec{\omega}]_\times) & V \mathbf{v} \\ \mathbf{0}^T & 1 \end{pmatrix}$$
where $V = I + \frac{1-\cos\theta}{\theta^2}[\vec{\omega}]_\times + \frac{\theta-\sin\theta}{\theta^3}[\vec{\omega}]_\times^2$.

### Geometric Equivariance
A neural network or operator $f: \mathcal{X} \to \mathcal{Y}$ is $G$-equivariant if group actions commute with evaluation:
$$f(g \cdot x) = g \cdot f(x), \quad \forall g \in G$$
For point clouds in 3D (e.g. molecular structures in AlphaFold or SE(3)-Transformers), equivariance ensures predictions rotate and translate identically when the coordinate frame is transformed.

## 2. Implementation Details
The module `Y01_Lie_Groups_SO3_SE3_and_Equivariance.py`:
- Implements exact vectorized `hat` and `vee` mappings.
- Implements numerically stabilized Rodrigues exponential and logarithmic maps with Taylor expansions for $\theta \to 0$.
- Computes the $6 \times 6$ Adjoint representation matrix $\text{Ad}_T = \begin{pmatrix} R & [\mathbf{p}]_\times R \\ 0 & R \end{pmatrix}$.
- Verifies exact $G$-equivariance on 3D point cloud manifolds.

## 3. Asymptotic & Numerical Analysis
- **Time Complexity**: $\mathcal{O}(1)$ for group operations (fixed $3 \times 3$ and $4 \times 4$ linear algebra).
- **Space Complexity**: $\mathcal{O}(1)$ storage.

## 4. Edge Cases & Numerical Stability
- **Small Angle Singularities**: When $\theta < 10^{-12}$, direct division by $\theta$ causes numerical NaN. Stable implementations employ series expansions:
  $$\frac{\sin\theta}{\theta} \approx 1 - \frac{\theta^2}{6}, \quad \frac{1-\cos\theta}{\theta^2} \approx \frac{1}{2} - \frac{\theta^2}{24}$$
- **Trigonometric Domain Clamping**: Numerical rounding can yield $\frac{\text{Tr}(R)-1}{2} = 1.0000000000000002$, triggering domain errors in $\arccos$. Values are clamped strictly to $[-1, 1]$.

## 5. Standalone Verification
The test suite validates:
- Orthogonality and determinant preservation: $R^T R = I, \det(R) = 1.0$.
- Round-trip bijective exactness: $\lVert\log(\exp(\xi)) - \xi\rVert < 10^{-10}$.
- Adjoint group conjugation: $T [\xi]_\wedge T^{-1} = [\text{Ad}_T \xi]_\wedge$.
- Strict $\text{SE}(3)$ coordinate equivariance ($< 10^{-12}$ error).

## 6. Real-World Applications & Context
Foundational to modern molecular dynamics (SE(3)-equivariant diffusion models, AlphaFold), robotics manipulator kinematics, SLAM, and computer graphics camera pose interpolation.

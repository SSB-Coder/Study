# Deep Dive: S04 — Finite-Difference PDE Solving

## 1. The Beginner's Analogy: Passing Heat Down a Metal Rod
Imagine heating the center of a long iron rod:
- You divide the rod into 50 tiny segments with chalk marks.
- Every segment has a thermometer measuring its local temperature.
- Newton's Law of Cooling says: if Segment 25 is hotter than its two neighbors (24 and 26), heat flows outward proportional to the temperature curvature (the second derivative).
- **The Finite Difference Method** steps time forward in tiny slices, updating each segment's thermometer by adding a fraction of the difference between its neighbors. If time steps are small enough, this grid calculation reproduces continuous partial differential calculus to machine precision.

---

## 2. The Deep Architecture/Algorithm Breakdown
Finite Difference Methods (FDM) discretize spatial and temporal derivatives onto structured computational grids:
1. **Taylor Series Derivative Discretization:**
   $$\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2 \Delta x}, \quad \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$
2. **The CFL Stability Criterion:** Explicit time-stepping schemes become violently unstable if the numerical domain of dependence does not contain the physical domain of dependence:
   - For Heat Equation: $r = \frac{\alpha \Delta t}{\Delta x^2} \le \frac{1}{2}$.
   - For Wave Equation: $C = \frac{c \Delta t}{\Delta x} \le 1$.
3. **Contrast with PINNs (K05):**
   - **FDM (Classical):** Hand-derives stencil updates directly on fixed spatial grids. Fast, exact stability guarantees, but scales poorly to complex 3D CAD geometries.
   - **Physics-Informed Neural Networks (PINNs, K05):** Mesh-free! Trains a neural network to minimize PDE residuals via autograd, handling irregular geometries and inverse parameter identification.

---

## 3. The Hardcore Mathematics

### 3.1 von Neumann Stability Analysis
Let roundoff error decompose into Fourier modes: $e_i^n = \xi^n e^{j k i \Delta x}$.
Substituting into the explicit heat equation:
$$\xi = 1 - 4 r \sin^2\left( \frac{k \Delta x}{2} \right)$$
For stability, amplification factor $|\xi| \le 1$:
$$-1 \le 1 - 4 r \sin^2\left( \frac{k \Delta x}{2} \right) \le 1 \implies 4 r \le 2 \implies r = \frac{\alpha \Delta t}{\Delta x^2} \le \frac{1}{2}$$

---

## 4. The Algorithm Step-by-Step
1. **Discretize Domain:** Set grid points $x_i = i \Delta x$, time steps $t_n = n \Delta t$.
2. **Apply Initial Conditions:** Set $u_i^0 = f(x_i)$.
3. **Time Marching:** Compute $u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2u_i^n + u_{i-1}^n)$.
4. **Enforce Boundary Conditions:** Set $u_0^{n+1}$ and $u_N^{n+1}$ according to Dirichlet or Neumann conditions.

---

## 5. Limitations / When It Breaks
- **Parabolic Time Step Bottleneck:** Because $\Delta t \le \frac{\Delta x^2}{2 \alpha}$, doubling spatial resolution requires cutting time steps by $4\times$. Implicit methods (Crank-Nicolson) solve a tridiagonal linear system to achieve unconditional stability.

---

## 6. Where This Fits
- **Reaction-Diffusion (T06):** Turing morphogenesis patterns solve coupled non-linear reaction-diffusion PDEs using this exact finite-difference machinery.
- **PINNs (K05):** Direct comparison between numerical and learned PDE solvers.

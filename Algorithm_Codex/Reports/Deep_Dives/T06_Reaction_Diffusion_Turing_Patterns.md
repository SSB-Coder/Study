# Deep Dive: T06 — Reaction-Diffusion & Turing Patterns

## 1. The Beginner's Analogy: How the Leopard Got Its Spots
In 1952, Alan Turing asked a profound biological question: *How does a completely uniform, spherical embryo develop complex spots, stripes, and fingers?*
- Imagine two chemicals diffusing through skin tissue:
  1. **The Activator (Chemical V):** Produces spots of dark melanin pigment. It catalyzes its own creation: wherever a speck of V appears, it rapidly creates more V!
  2. **The Inhibitor (Chemical U):** Suppresses the activator, preventing it from spreading infinitely.
- Crucially, **the inhibitor diffuses much faster than the activator ($D_u > D_v$)!**
- The activator builds a concentrated local spot, while the fast-diffusing inhibitor races outward, forming a sterile moat around the spot. This creates sharp, permanent spotted and striped animal coat patterns from smooth chemistry!

---

## 2. The Deep Architecture/Algorithm Breakdown
Alan Turing's seminal paper *"The Chemical Basis of Morphogenesis"* (1952) introduced reaction-diffusion systems:
1. **Diffusion-Driven Instability:**
   A spatially homogeneous steady state that is **stable** in the absence of diffusion becomes **spatially unstable** when two chemicals diffuse at different rates ($D_u \ne D_v$), spontaneously generating macroscale periodic spatial patterns.
2. **The Gray-Scott Model (1983):**
   A cubic autocatalytic chemical reaction:
   $$U + 2V \to 3V, \quad V \to P$$
   - $F$: Feed rate replenishing chemical $U$.
   - $k$: Kill rate converting activated $V$ into inert product.

---

## 3. The Hardcore Mathematics

### 3.1 Coupled Reaction-Diffusion PDEs
$$\begin{aligned}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u - u v^2 + F (1 - u) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v + u v^2 - (F + k) v
\end{aligned}$$

### 3.2 2D Discrete Laplacian
On an $N \times N$ periodic torus with grid spacing $\Delta x = 1$:
$$\nabla^2 u(r, c) \approx u(r+1, c) + u(r-1, c) + u(r, c+1) + u(r, c-1) - 4 u(r, c)$$

### 3.3 Turing Instability Condition
Linearizing the reaction terms $\mathbf{f}(u, v)$ around homogeneous steady state $(u_0, v_0)$ yields Jacobian $J$:
For Turing instability to occur:
1. $\text{tr}(J) < 0$ and $\det(J) > 0$ (stable without diffusion).
2. $D_v J_{11} + D_u J_{22} > 2 \sqrt{D_u D_v \det(J)}$ (instability driven by differential diffusion).

---

## 4. The Algorithm Step-by-Step
1. **Initialize Grid:** Set $u=1, v=0$. Inject a small perturbation $v>0$ in the center.
2. **Finite-Difference Marching:**
   - Compute discrete Laplacians $\nabla^2 u$ and $\nabla^2 v$ with periodic roll.
   - Compute cubic reaction $u v^2$.
   - Update $u \leftarrow u + \Delta t (D_u \nabla^2 u - u v^2 + F(1-u))$.
   - Update $v \leftarrow v + \Delta t (D_v \nabla^2 v + u v^2 - (F+k)v)$.
3. **Clip & Render:** Clamp concentrations to $[0, 1]$.

---

## 5. Limitations / When It Breaks
- **Numerical Instability:** If diffusion rate $D_u \Delta t / \Delta x^2 > 0.25$, explicit integration blows up due to the 2D Von Neumann CFL limit (S04).

---

## 6. Where This Fits
- **Cross-Reference Finite-Difference Solvers (S04):** Direct non-linear coupled application of spatial Laplacian stencils.

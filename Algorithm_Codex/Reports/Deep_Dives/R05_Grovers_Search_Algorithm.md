# R05: Grover's Search Algorithm

## 1. Mathematical Formulation & Architecture
Grover's algorithm (1996) solves the problem of finding a unique marked item $w \in \{0, 1\}^n$ among an unsorted database of $N = 2^n$ elements with quadratic speedup: $\mathcal{O}(\sqrt{N})$ quantum queries versus $\mathcal{O}(N)$ classical queries.

### State Space Decomposition
The Hilbert space decomposes into a 2D subspace spanned by the marked state $\lvert w\rangle$ and the uniform superposition of non-marked states $\lvert w^\perp\rangle$:
$$\lvert s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1} \lvert x\rangle = \sin\left(\frac{\theta}{2}\right) \lvert w\rangle + \cos\left(\frac{\theta}{2}\right) \lvert w^\perp\rangle, \quad \sin\left(\frac{\theta}{2}\right) = \frac{1}{\sqrt{N}}$$

### Grover Iteration Operator
Each iteration applies the composite operator $G = D \cdot O$:
1. **Oracle Reflection** $O_w = I - 2\lvert w\rangle\langle w\rvert$:
   $$O_w \lvert x\rangle = \begin{cases} -\lvert x\rangle & \text{if } x = w \\ \lvert x\rangle & \text{if } x \ne w \end{cases}$$
2. **Diffusion Operator** (Inversion about the mean) $D = 2\lvert s\rangle\langle s\rvert - I$:
   $$D \sum_x a_x \lvert x\rangle = \sum_x (2\bar{a} - a_x)\lvert x\rangle, \quad \bar{a} = \frac{1}{N}\sum_x a_x$$

Geometrically, the product $G = D \cdot O$ represents a 2D rotation by angle $\theta$ towards $\lvert w\rangle$. After $R$ iterations:
$$R \approx \left\lfloor \frac{\pi}{4}\sqrt{N} \right\rfloor$$
the state rotates to within $\mathcal{O}(1/\sqrt{N})$ of the target state $\lvert w\rangle$.

## 2. Implementation Details
The `GroverSearch` class in `R05_Grovers_Search_Algorithm.py`:
- Initializes the state vector to uniform superposition.
- Implements the exact selective phase inversion oracle $O_w$.
- Computes the inversion about the mean in $\mathcal{O}(N)$ time.
- Records amplitude amplification evolution over successive iterations.

## 3. Asymptotic & Numerical Analysis
- **Complexity**: Query complexity is $\mathcal{O}(\sqrt{N})$, which is provably optimal (Bennett, Bernstein, Brassard, Vazirani 1997).
- **Overcooking Phenomenon**: If iterations exceed $R_{\text{opt}}$, the state rotates past $\lvert w\rangle$, causing probability oscillatory decay.

## 4. Edge Cases & Numerical Stability
- Target state at extreme index boundaries ($0$ or $N-1$) processes identically.
- Amplitude arithmetic preserves $L_2$ norm $\sum_x \lvert a_x\rvert^2 = 1.0$ within floating point tolerance $\sim 10^{-16}$.

## 5. Standalone Verification
The test suite validates:
- 3-qubit search ($N=8$): Success probability reaches $94.5\%$ at $R=2$.
- 4-qubit search ($N=16$): Success probability reaches $96.1\%$ at $R=3$.
- Conservation of unitary probability mass across all iterations.

## 6. Real-World Applications & Context
Subroutine for collision finding, minimum finding, quantum pattern matching, and accelerating SAT solving algorithms.

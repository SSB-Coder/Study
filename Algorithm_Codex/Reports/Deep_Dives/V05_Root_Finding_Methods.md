# Deep Dive: V05 — Root-Finding Methods (Bisection, Newton-Raphson, Secant)

## 1. The Beginner's Analogy: Zeroing In on Target
Imagine hunting for where a curved electrical line crosses zero ground:
- **Bisection** is binary search on a continuous line: test both sides of the fence, see where the sign flips, and cut the interval in half every step. It is utterly reliable, never gets lost, but marches at a fixed, unhurried pace.
- **Newton-Raphson** stands at your current guess, measures the local slope underfoot, and follows that tangent downhill straight to the zero intercept. If you are close to the target, Newton's method is breathtakingly fast: the number of correct decimal places doubles on every single step.
- **Secant Method** wants Newton's speed without having to calculate explicit derivatives: it draws a straight ruler across your last two guesses and leaps to where that secant ruler hits zero.

---

## 2. The Deep Architecture/Algorithm Breakdown

### 2.1 Convergence Rates Comparison
Let error $e_k = |x_k - x^*|$. A method has order of convergence $p$ if $\lim_{k \to \infty} \frac{e_{k+1}}{e_k^p} = C$:
| Method | Convergence Order $p$ | Evaluations / Iteration | Requirement | Robustness |
| :--- | :--- | :--- | :--- | :--- |
| **Bisection** | $1$ (Linear, rate $0.5$) | $1$ | Bracket $[a, b]$ with $f(a)f(b) < 0$ | Guaranteed |
| **Secant** | $\approx 1.618$ (Superlinear) | $1$ | Two initial points $x_0, x_1$ | Local |
| **Newton-Raphson** | $2$ (Quadratic) | $2$ ($f$ and $f'$) | Differentiable $f$, initial $x_0$ | Fast local, can diverge |

---

## 3. The Hardcore Mathematics

### 3.1 Newton-Raphson Derivation & Quadratic Proof
Expand $f(x^*)$ around current estimate $x_k$ via Taylor series:
$$0 = f(x^*) = f(x_k) + f'(x_k)(x^* - x_k) + \frac{1}{2} f''(\xi)(x^* - x_k)^2$$
Truncating higher-order terms yields the iteration:
$$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
Subtracting $x^*$ from both sides:
$$e_{k+1} = e_k - \frac{f(x_k)}{f'(x_k)} = \frac{e_k f'(x_k) - f(x_k)}{f'(x_k)}$$
Using Taylor expansion for $f(x_k)$ around $x^*$:
$$e_{k+1} \approx \frac{f''(x^*)}{2 f'(x^*)} e_k^2 \implies p = 2$$

### 3.2 Bisection Interval Halving
After $k$ iterations, the bracket width is:
$$\Delta_k = \frac{b - a}{2^k}$$
To achieve tolerance $\epsilon$, the required number of iterations is strictly:
$$k = \left\lceil \log_2 \left( \frac{b - a}{\epsilon} \right) \right\rceil$$

---

## 4. The Algorithm Step-by-Step
1. **Bisection:**
   - Verify $f(a) f(b) < 0$. Compute midpoint $c = (a + b) / 2$.
   - If $|f(c)| < \text{tol}$ or $(b-a)/2 < \text{tol}$, return $c$.
   - If $\text{sgn}(f(a)) == \text{sgn}(f(c))$, set $a = c$; else set $b = c$.
2. **Newton-Raphson:**
   - Evaluate $f(x_k)$ and $f'(x_k)$. If $|f'(x_k)| < \epsilon$, trigger fallback.
   - Update $x_{k+1} = x_k - f(x_k) / f'(x_k)$.
   - Check convergence: $|x_{k+1} - x_k| < \text{tol}$ or $|f(x_{k+1})| < \text{tol}$.

---

## 5. Limitations / When It Breaks
- **Zero Derivatives & Inflection Points:** If $f'(x_k) = 0$, Newton divides by zero. Near inflection points, Newton can enter infinite cycles (e.g. oscillating between two values) or shoot off to infinity.
- **Multiple Roots:** If $f'(x^*) = 0$ (root has multiplicity $m > 1$), Newton's convergence degrades from quadratic ($p=2$) to linear ($p=1$).

---

## 6. Where This Fits
1. **Orbital Mechanics (S06):** Solving Kepler's transcendental equation $M = E - e \sin(E)$ for eccentric anomaly $E$ via Newton-Raphson.
2. **Support Vector Machines (B06):** Determining intercept thresholds.
3. **Inverse Kinematics (X08):** Multi-dimensional Newton iteration on robot kinematic chains.

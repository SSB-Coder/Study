# Deep Dive: Markov Chain Monte Carlo (Metropolis-Hastings & Gibbs Sampling)

## 1. The Beginner's Analogy: The Blindfolded Explorer on an Island Archipelago
Imagine a blindfolded explorer wandering across an archipelago of islands of varying population:
- You want the explorer to spend time on each island in exact proportion to that island's population.
- The explorer cannot see the whole map. At each step, they propose stepping onto a neighboring island:
  - If the neighbor island has a *larger* population, they always take the step!
  - If the neighbor island has a *smaller* population, they don't immediately refuse; they flip a biased coin whose chance of heads equals the population ratio $\frac{\text{Pop}_{\text{neighbor}}}{\text{Pop}_{\text{current}}}$.
- By following this simple local rule (**Metropolis-Hastings**), the explorer will naturally spend $70\%$ of their lifetime on the largest island, $20\%$ on the medium island, and $10\%$ on the tiny reef, perfectly sampling the unknown population distribution!

---

## 2. The Deep Architecture/Algorithm Breakdown
MCMC draws representative samples from complex, unnormalized target probability densities $p(x) = \frac{\tilde{p}(x)}{Z}$ where normalizing partition constant $Z = \int \tilde{p}(x) dx$ is intractable:
1. **Stationary Distribution:** MCMC constructs an ergodic Markov chain whose unique invariant distribution $\pi(x)$ equals target density $p(x)$.
2. **Detailed Balance Condition:** The reversibility condition guaranteeing stationarity:
   $$p(x) T(x \rightarrow x') = p(x') T(x' \rightarrow x)$$
3. **Metropolis-Hastings (MH):** Proposes transitions via proposal kernel $q(x' \mid x)$ and accepts with probability:
   $$\alpha(x, x') = \min\left(1, \frac{\tilde{p}(x') q(x \mid x')}{\tilde{p}(x) q(x' \mid x)}\right)$$
   *(The intractable denominator $Z$ cancels out completely in the ratio!)*
4. **Gibbs Sampling:** Special case where transitions update one coordinate at a time by sampling from full conditional distributions $P(x_i \mid x_{-i})$. Acceptance probability $\alpha \equiv 1.0$.

---

## 3. The Hardcore Mathematics

### 3.1 Proof of Detailed Balance for Metropolis-Hastings
Let transition kernel be $T(x \rightarrow x') = q(x' \mid x) \alpha(x, x')$.
Assume without loss of generality that $\tilde{p}(x') q(x \mid x') \le \tilde{p}(x) q(x' \mid x)$:
Then $\alpha(x, x') = \frac{\tilde{p}(x') q(x \mid x')}{\tilde{p}(x) q(x' \mid x)}$, and $\alpha(x', x) = 1$.
Now evaluate both sides of detailed balance:
$$p(x) T(x \rightarrow x') = p(x) q(x' \mid x) \left[ \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)} \right] = p(x') q(x \mid x')$$
$$p(x') T(x' \rightarrow x) = p(x') q(x \mid x') \cdot 1 = p(x') q(x \mid x')$$
Both sides match identically! Thus detailed balance holds, proving that the stationary distribution is $p(x)$.

### 3.2 Gibbs Sampling as Component-Wise MH
Updating coordinate $i$: proposal is the exact conditional distribution $q(x_i' \mid x) = p(x_i' \mid x_{-i})$.
The acceptance ratio:
$$\alpha = \frac{p(x_i', x_{-i}) p(x_i \mid x_{-i})}{p(x_i, x_{-i}) p(x_i' \mid x_{-i})} = \frac{p(x_i' \mid x_{-i}) p(x_{-i}) p(x_i \mid x_{-i})}{p(x_i \mid x_{-i}) p(x_{-i}) p(x_i' \mid x_{-i})} = 1.0$$
Every proposed Gibbs step is accepted with probability $100\%$!

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Choose arbitrary initial state $x^{(0)}$. Set burn-in steps $B$ and total samples $S$.
2. **Metropolis-Hastings Loop:** For $t = 1$ to $B + S$:
   - Propose candidate $x' \sim q(x' \mid x^{(t-1)})$.
   - Calculate acceptance probability $\alpha = \min\left(1, \frac{\tilde{p}(x') q(x^{(t-1)} \mid x')}{\tilde{p}(x^{(t-1)}) q(x' \mid x^{(t-1)})}\right)$.
   - Draw $u \sim \text{Uniform}(0, 1)$. If $u \le \alpha$, set $x^{(t)} = x'$; else set $x^{(t)} = x^{(t-1)}$.
3. **Discard Burn-In:** Throw away first $B$ samples; retain remaining $S$ samples for posterior inference.

---

## 5. Limitations / When It Breaks
- **Slow Mixing & Dimensionality:** In high dimensions, random-walk proposals get trapped in local modes or require tiny steps, causing severe sample autocorrelation.
- **Historical Fix:** Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS) utilize gradient vector fields to glide across target surfaces.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/07_Diffusion_Models.md`. Langevin MCMC dynamics and stochastic score-based reverse diffusion generate photorealistic images by taking gradient-guided MCMC steps.

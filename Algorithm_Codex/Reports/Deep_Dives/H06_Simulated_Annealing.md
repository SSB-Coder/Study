# H06: Simulated Annealing (SA) & The Metropolis Criterion

## 1. Intuition & The "Grand Analogy"
Imagine a metallurgist forging an extraordinarily durable steel sword. If they heat the steel to a molten glow and immediately plunge it into freezing water (quenching), the iron and carbon atoms freeze in disordered, brittle crystalline dislocations. But if the metallurgist cools the metal at an excruciatingly slow, controlled rate (**annealing**), the atoms retain sufficient thermal kinetic energy at high temperatures to escape trapped configurations, settling gradually into an immaculate, minimum-energy crystal lattice.

**Simulated Annealing (SA)** (Kirkpatrick, Gelatt, Vecchi, 1983) applies this thermodynamic cooling process to optimization. At high temperatures, the algorithm willingly accepts worse solutions to climb out of local valleys. As temperature decreases, it gradually transitions into pure greedy descent.

---

## 2. Architectural & Mathematical Breakdown
Let $E(\mathbf{x})$ denote the objective function (energy).
At each step, generate a neighboring state $\mathbf{x}_{\text{cand}} \sim \mathcal{N}(\mathbf{x}, \sigma^2)$ and compute energy differential:
$$\Delta E \triangleq E(\mathbf{x}_{\text{cand}}) - E(\mathbf{x})$$

### The Metropolis Acceptance Criterion
$$P(\text{Accept } \mathbf{x}_{\text{cand}}) = \begin{cases} 1 & \text{if } \Delta E < 0 \quad (\text{Downhill move always accepted}) \\ \exp\left( -\frac{\Delta E}{T} \right) & \text{if } \Delta E \ge 0 \quad (\text{Uphill move accepted stochastically}) \end{cases}$$

### Annealing Schedules
- **Geometric Cooling:** $T_{k+1} = \alpha T_k, \quad \alpha \in [0.85, 0.99]$.
- **Logarithmic Cooling (Geman & Geman):** $T_k = \frac{T_0}{\ln(1 + k)}$.

---

## 3. Hardcore Mathematical Derivations

### Asymptotic Convergence to Global Optimum
The state space exploration forms a time-inhomogeneous Markov chain with stationary distribution:
$$\pi_T(\mathbf{x}) = \frac{1}{Z(T)} \exp\left( -\frac{E(\mathbf{x})}{T} \right)$$
where $Z(T) = \sum_{\mathbf{x}'} \exp\left( -\frac{E(\mathbf{x}')}{T} \right)$ is the partition function.

Let $\mathcal{X}^* = \arg\min_{\mathbf{x}} E(\mathbf{x})$ be the set of global minima. As temperature cools asymptotically to absolute zero:
$$\lim_{T \to 0} \pi_T(\mathbf{x}) = \lim_{T \to 0} \frac{e^{-(E(\mathbf{x}) - E^*)/T}}{\sum_{\mathbf{x}'} e^{-(E(\mathbf{x}') - E^*)/T}} = \begin{cases} \frac{1}{|\mathcal{X}^*|} & \text{if } \mathbf{x} \in \mathcal{X}^* \\ 0 & \text{otherwise} \end{cases}$$
**Geman & Geman Theorem (1984):** If the cooling schedule satisfies $T(t) \ge \frac{c}{\ln(1 + t)}$, the probability of residing in a global minimum converges to 1 as $t \to \infty$.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Simulated Annealing
Inputs: Initial temperature T_0, cooling rate alpha, iterations per temp M
1. Initialize state x = x_0, T = T_0
2. While T > T_min:
     For m = 1 to M:
       x_cand = GenerateNeighbor(x)
       delta_E = E(x_cand) - E(x)
       If delta_E < 0 or rand() < exp(- delta_E / T):
          x = x_cand
       If E(x) < E(x_best):
          x_best = x
     T = alpha * T
3. Return x_best
```

---

## 5. Failure Modes & When It Breaks
1. **Quenching (Premature Freeze):** If temperature drops too fast ($\alpha \ll 0.8$), the algorithm turns into greedy hill climbing and freezes in the nearest local minimum.
2. **Neighborhood Sizing:** If candidate jumps are too large, proposal acceptance drops near 0; if too small, diffusion across multi-modal barriers is excessively slow.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Temperature in Softmax Generation:** Autoregressive LLM token sampling uses $P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$, which is mathematically identical to the Boltzmann-Metropolis distribution.
- **Constrained Decoding & Token Swaps:** Exploring combinatorial token edits in sentence paraphrase generation.

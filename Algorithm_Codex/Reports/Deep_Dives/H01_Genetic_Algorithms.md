# H01: Genetic Algorithms (GA)

## 1. Intuition & The "Grand Analogy"
Imagine breeding prize-winning roses over generations. You cannot manually rewrite the genetic DNA of a flower with tweezers. Instead, you select the hardiest, most vibrant blooms in your garden (selection), cross-pollinate them to combine desirable traits (crossover), and allow natural cosmic rays or random copying anomalies to introduce fresh variations (mutation). The best specimens survive and reproduce, while sickly weeds are culled.

**Genetic Algorithms (GAs)** (Holland, 1975; Goldberg, 1989) apply Darwinian natural selection and Mendelian genetics to optimization. By evolving a population of candidate solution strings across generations, GAs explore rugged, non-differentiable, and discontinuous fitness landscapes without requiring gradients.

---

## 2. Architectural & Mathematical Breakdown
A GA operates on a population $\mathcal{P}_t = \{ \mathbf{x}_1, \dots, \mathbf{x}_N \}$ of candidate chromosomes $\mathbf{x}_i \in \{0, 1\}^L$ (or continuous $\mathbb{R}^d$).

### Core Genetic Operators
1. **Fitness Evaluation:** $f_i = f(\mathbf{x}_i)$.
2. **Selection:**
   - *Roulette Wheel (Fitness Proportionate):* $p_i = \frac{f_i}{\sum_j f_j}$
   - *Tournament Selection:* Pick $k$ individuals uniformly at random; the highest fitness wins.
3. **Crossover (Recombination):**
   - *Single-Point Crossover:* Split parents at index $c \sim \text{Uniform}(1, L-1)$ and swap tails.
   - *Uniform Crossover:* Each gene independently inherited from parent 1 or 2 with probability 0.5.
4. **Mutation:** Bit-flip each gene with small mutation rate $p_m \approx 1/L$.
5. **Elitism:** Directly preserve the top $E$ best individuals into the next generation without mutation.

---

## 3. Hardcore Mathematical Derivations

### Holland's Schema Theorem (Fundamental Theorem of Genetic Algorithms)
A schema $H$ is a template string over the alphabet $\{0, 1, *\}$, where $*$ is a wildcard.
- Order $o(H)$: Number of fixed positions (0 or 1) in $H$.
- Defining Length $\delta(H)$: Distance between first and last fixed positions.

Let $m(H, t)$ be the expected number of chromosomes matching schema $H$ at generation $t$, with average fitness $f(H)$ and population mean $\bar{f}$:
$$m(H, t+1) \ge m(H, t) \cdot \frac{f(H)}{\bar{f}} \left[ 1 - p_c \frac{\delta(H)}{L - 1} \right] \left[ (1 - p_m)^{o(H)} \right]$$
Using $(1 - p_m)^{o(H)} \approx 1 - o(H) p_m$ for small $p_m$:
$$m(H, t+1) \ge m(H, t) \frac{f(H)}{\bar{f}} \left[ 1 - p_c \frac{\delta(H)}{L - 1} - o(H) p_m \right]$$
**Conclusion (Building Block Hypothesis):** Short, low-order, above-average schemata ("building blocks") receive exponentially increasing representation over generations.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Canonical Genetic Algorithm
Inputs: Population size N, mutation rate p_m, crossover rate p_c, elite count E
1. Initialize population P_0 with N random chromosomes
2. For generation t = 0 to G_max:
     a. Evaluate fitness f(x) for all x in P_t
     b. Sort population by fitness descending; copy top E into P_{t+1} (Elitism)
     c. While |P_{t+1}| < N:
          Select parent 1 and parent 2 via Tournament Selection
          If random() < p_c:
             child1, child2 = Crossover(parent1, parent2)
          Else:
             child1, child2 = copy(parent1), copy(parent2)
          child1 = Mutate(child1, p_m)
          child2 = Mutate(child2, p_m)
          Add children to P_{t+1}
3. Return chromosome with highest fitness in final population
```

---

## 5. Failure Modes & When It Breaks
1. **Premature Convergence:** If a mediocre individual with slightly above-average fitness dominates early tournaments, the population's genetic diversity collapses to zero, stranding the GA in a local optimum.
2. **Genetic Drift:** In small populations, random sampling errors cause alleles to fixate regardless of their true fitness benefit.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **PromptBreeder & EvoPrompt:** Evolving prompts for LLMs using genetic operators (mutation via rewriting instructions, crossover via blending two prompt styles).
- **Adversarial Red-Teaming (Jailbreaking):** Genetic algorithms mutate token sequences to find discrete adversarial suffix prompts that bypass safety guardrails.

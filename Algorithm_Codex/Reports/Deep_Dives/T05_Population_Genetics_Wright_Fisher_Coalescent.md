# Deep Dive: T05 — Population Genetics (Wright-Fisher & Coalescent)

## 1. The Beginner's Analogy: Drawing Marbles Across Generations
Imagine a jar containing 100 Blue marbles and 100 Red marbles:
- **Wright-Fisher Forward Evolution:** To create the next generation, you blindly draw 200 marbles with replacement. By random luck, you draw 104 Blue and 96 Red. Next round, you draw from that new ratio. Over hundreds of generations, random drift inevitably causes one color to take over 100% (fixation), driving the other extinct.
- **Kingman's Coalescent Backward Evolution:** Look at 5 living people today. Trace their family trees backward in time. As you go backward, family branches merge together ("coalesce") whenever two cousins share a common grandparent. Eventually, all 5 branches merge into one single person: the Most Recent Common Ancestor (MRCA).

---

## 2. The Deep Architecture/Algorithm Breakdown
Population genetics models the mathematical dynamics of genetic variation across time:
1. **Wright-Fisher Model (1930s):**
   - The canonical forward discrete-generation model.
   - A population of $N$ diploid individuals has $2N$ gene copies.
   - Next generation is formed by drawing $2N$ alleles independently from a **Binomial distribution** $B(2N, p')$.
2. **Kingman's Coalescent (1982):**
   - The dual backward-in-time perspective.
   - Instead of tracking all $2N$ individuals forward, track only the $k$ sampled individuals backward.
   - The waiting time until any two of the $k$ lineages coalesce is exponentially distributed with rate $\binom{k}{2} / (2N)$.

---

## 3. The Hardcore Mathematics

### 3.1 Forward Wright-Fisher Binomial Transition
$$P(k_{t+1} = j \mid p_t) = \binom{2N}{j} (p')^j (1 - p')^{2N - j}$$
With selection coefficient $s$:
$$p' = \frac{p (1 + s)}{1 + p s}$$

### 3.2 Kingman's Coalescent Waiting Times
When $k$ ancestral lineages remain, the probability that two lineages pick the same parent in the previous generation is:
$$P(\text{Coalescence in 1 gen}) = \frac{\binom{k}{2}}{2N}$$
In the continuous-time diffusion limit ($N \to \infty$), the waiting time $T_k$ is an exponential random variable:
$$T_k \sim \text{Exp}\left( \frac{k(k - 1)}{4N} \right)$$
Expected time to coalescence:
$$\mathbb{E}[T_k] = \frac{4N}{k(k - 1)}$$
Total expected time to the Most Recent Common Ancestor (TMRCA):
$$\mathbb{E}[T_{\text{MRCA}}] = \sum_{k=2}^n \mathbb{E}[T_k] = 4N \sum_{k=2}^n \left( \frac{1}{k-1} - \frac{1}{k} \right) = 4N \left( 1 - \frac{1}{n} \right)$$

---

## 4. The Algorithm Step-by-Step
1. **Forward Wright-Fisher:** Set initial $p_0$. In each generation, compute selection-weighted $p'$, sample $k \sim \text{Binomial}(2N, p')$, record $p = k / 2N$. Repeat until fixation ($p \in \{0, 1\}$) or max generations.
2. **Backward Coalescent:** Start with $k$ lineages. While $k > 1$, draw exponential waiting time $\Delta t \sim \text{Exp}\left( \binom{k}{2} / 2N \right)$. Randomly merge two lineages, decrement $k \leftarrow k - 1$.

---

## 5. Limitations / When It Breaks
- **Non-Panmictic Structure:** Classical coalescent assumes a single well-mixed population. Geographic isolation, population bottlenecks, and migration require structured island or stepping-stone models.

---

## 6. Where This Fits
- **Cross-Reference Diffusion & Langevin (P02):** The continuous limit of Wright-Fisher is the famous Kimura Genetic Diffusion Equation: $\frac{\partial \phi}{\partial t} = \frac{1}{4N} \frac{\partial^2}{\partial p^2}[p(1-p)\phi] - s \frac{\partial}{\partial p}[p(1-p)\phi]$.

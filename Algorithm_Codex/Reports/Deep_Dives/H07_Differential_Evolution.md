# H04: Differential Evolution (DE)

## 1. Intuition & The "Grand Analogy"
Imagine explorers searching for treasure hidden in an undulating desert of sand dunes. Each explorer holds a GPS tracker. Rather than taking random steps, explorer $i$ looks across the desert and selects three other distinct explorers ($r_1, r_2, r_3$). They measure the exact direction and distance between $r_2$ and $r_3$, multiply this difference vector by a scaling factor $F$, and add it to the position of $r_1$ to create a candidate camp. If the candidate camp is closer to the treasure than explorer $i$'s current tent, they immediately pack up and move.

**Differential Evolution (DE)** (Storn & Price, 1997) is one of the most powerful continuous stochastic optimization algorithms ever devised. Its hallmark feature is **self-adapting step size**: as the population converges around an optimum, the difference vectors naturally shrink, automatically transitioning from wide exploration to microscopic exploitation.

---

## 2. Architectural & Mathematical Breakdown
A population consists of $N$ vectors $\mathbf{x}_i \in \mathbb{R}^d$.

### The 4 Stages of DE/rand/1/bin
1. **Mutation:** For target vector $\mathbf{x}_i$, select mutually exclusive random indices $r_1 \neq r_2 \neq r_3 \neq i$:
   $$\mathbf{v}_i = \mathbf{x}_{r_1} + F \cdot (\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$$
   where $F \in (0, 2]$ is the mutation scale factor (typically $F \approx 0.8$).
2. **Crossover (Binomial):** Blend mutant $\mathbf{v}_i$ with target $\mathbf{x}_i$ to produce trial vector $\mathbf{u}_i$:
   $$u_{i, j} = \begin{cases} v_{i, j} & \text{if } \text{rand}_j \le CR \text{ or } j = j_{\text{rand}} \\ x_{i, j} & \text{otherwise} \end{cases}$$
   where $CR \in [0, 1]$ is the crossover probability, and $j_{\text{rand}} \sim \text{Uniform}(1, d)$ guarantees at least one mutant gene is preserved.
3. **Selection (One-on-One Greed):**
   $$\mathbf{x}_i^{(g+1)} = \begin{cases} \mathbf{u}_i & \text{if } f(\mathbf{u}_i) \le f(\mathbf{x}_i^{(g)}) \\ \mathbf{x}_i^{(g)} & \text{otherwise} \end{cases}$$

---

## 3. Hardcore Mathematical Derivations

### Inherent Scale Invariance & Self-Adaptation
Let $\mathbf{C} = \text{Cov}(\mathcal{P})$ be the covariance matrix of the population.
The variance of the mutant difference vector $\mathbf{d} = \mathbf{x}_{r_2} - \mathbf{x}_{r_3}$ is:
$$\text{Var}(\mathbf{x}_{r_2} - \mathbf{x}_{r_3}) = \text{Var}(\mathbf{x}_{r_2}) + \text{Var}(\mathbf{x}_{r_3}) - 2 \text{Cov}(\mathbf{x}_{r_2}, \mathbf{x}_{r_3}) = 2 \mathbf{C}$$
The mutation step produces candidate perturbations aligned with the principal axes of the population distribution:
$$\text{Cov}(\mathbf{v}_i) \approx \mathbf{C} + 2 F^2 \mathbf{C} = (1 + 2F^2) \mathbf{C}$$
As the swarm concentrates in a narrow valley, $\mathbf{C}$ automatically contracts in width and stretches along the valley contour, providing second-order-like adaptive exploration with zero matrix inversions.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Classical DE/rand/1/bin
Inputs: Population size N >= 4, scaling factor F in (0, 2), crossover CR in [0, 1]
1. Initialize population x_1, ..., x_N randomly within bounds
2. For generation = 1 to G_max:
     For each i = 1 to N:
       Choose r1, r2, r3 from {1, ..., N} \ {i} mutually distinct
       Compute mutant: v = x_r1 + F * (x_r2 - x_r3)
       
       Choose j_rand uniformly from {1, ..., d}
       For j = 1 to d:
         If rand() <= CR or j == j_rand:
            u_j = v_j
         Else:
            u_j = x_{i, j}
            
       If f(u) <= f(x_i):
          x_i_next = u
       Else:
          x_i_next = x_i
3. Return best solution in population
```

---

## 5. Failure Modes & When It Breaks
1. **Parameter Stagnation:** If $F$ is too small ($F < 0.4$), the population rapidly collapses into a subspace of lower dimensionality before finding the optimum.
2. **Boundary Violations:** The difference vector can easily propel trial vectors far outside physical search bounds, requiring clamping or reflection.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Hyperparameter Optimization (HPO):** Tuning learning rate schedules, weight decay, and LoRA rank hyperparameters during LLM pretraining.
- **Continuous Prompt Distillation:** Tuning soft prompt tokens under black-box access constraints.

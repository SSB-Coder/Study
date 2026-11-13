# H09: Multi-Objective Evolutionary Algorithms & NSGA-II

## 1. Intuition & The "Grand Analogy"
Imagine shopping for an electric sports car. You want two things that are fundamentally in tension: maximum acceleration (0-60 in 2 seconds) and maximum range (600 miles per charge). A car with giant heavy batteries will have long range but poor acceleration. A lightweight stripped-down roadster will accelerate like a rocket but run out of battery in 80 miles. There is no single "best" car; rather, there is a curve of elite trade-offs where you cannot improve acceleration without sacrificing range.

This curve is the **Pareto Optimal Frontier**. **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** (Deb et al., 2002) is the canonical algorithm for discovering and maintaining a diverse set of Pareto-optimal solutions across conflicting objectives.

---

## 2. Architectural & Mathematical Breakdown
Let $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \dots, f_m(\mathbf{x}))^\top$ be $m$ objective functions to be minimized.

### Pareto Dominance
A solution $\mathbf{x}_1$ **dominates** $\mathbf{x}_2$ (denoted $\mathbf{x}_1 \prec \mathbf{x}_2$) if and only if:
1. $\forall i \in \{1, \dots, m\}: f_i(\mathbf{x}_1) \le f_i(\mathbf{x}_2)$, and
2. $\exists j \in \{1, \dots, m\}: f_j(\mathbf{x}_1) < f_j(\mathbf{x}_2)$.

### The Fast Non-Dominated Sorting Algorithm
Partitions population $\mathcal{P}$ into hierarchical Pareto fronts $\mathcal{F}_1, \mathcal{F}_2, \dots$:
- $\mathcal{F}_1$: Set of all non-dominated individuals in $\mathcal{P}$.
- $\mathcal{F}_k$: Set of non-dominated individuals after removing $\mathcal{F}_1 \cup \dots \cup \mathcal{F}_{k-1}$.

### Crowding Distance Operator ($I_i$)
To maintain uniform diversity along the Pareto frontier without artificial niche parameters, NSGA-II computes the perimeter of the cuboid formed by its nearest neighbors on front $\mathcal{F}$:
$$I_i = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$
Boundary solutions are assigned $I = \infty$.

---

## 3. Hardcore Mathematical Derivations

### Computational Complexity of Fast Non-Dominated Sorting
Naive non-dominated sorting requires comparing every individual against every other individual for all $M$ objectives, yielding $\mathcal{O}(M N^3)$.
NSGA-II reduces this to $\mathcal{O}(M N^2)$:
- For each individual $p$, maintain domination count $n_p$ (number of solutions dominating $p$) and set $S_p$ (solutions dominated by $p$).
- All individuals with $n_p = 0$ form front $\mathcal{F}_1$.
- For each $p \in \mathcal{F}_1$, iterate through each $q \in S_p$ and decrement $n_q \leftarrow n_q - 1$.
- If $n_q = 0$, $q$ belongs to $\mathcal{F}_2$.
- Each solution is visited at most once per front level, yielding strict $\mathcal{O}(M N^2)$ operations.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: NSGA-II
Inputs: Population size N, objectives f_1, ..., f_m
1. Combine parent P_t and offspring Q_t into R_t of size 2N
2. Perform Fast Non-Dominated Sorting on R_t -> F_1, F_2, ...
3. P_{t+1} = empty list, i = 1
4. While |P_{t+1}| + |F_i| <= N:
     Compute Crowding Distance on F_i
     P_{t+1} = P_{t+1} U F_i
     i = i + 1
5. If |P_{t+1}| < N:
     Compute Crowding Distance on F_i
     Sort F_i descending by Crowding Distance
     Add top (N - |P_{t+1}|) individuals from F_i to P_{t+1}
6. Create offspring Q_{t+1} via Crowded Tournament Selection, crossover, and mutation
```

---

## 5. Failure Modes & When It Breaks
1. **Curse of Many Objectives ($M > 4$):** As the number of objectives increases, almost all solutions in a random population become non-dominated with respect to each other ($\mathcal{F}_1$ contains 99% of individuals), destroying the selection pressure (requires NSGA-III or reference points).
2. **Discontinuous Frontiers:** Sharp disjoint disconnected Pareto sets can cause crowding distance to over-sample boundary gaps.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Multi-Objective Alignment (Helpfulness vs Harmlessness):** RLHF models must balance contradictory objectives (e.g. being maximally helpful vs strictly safe/harmless). NSGA-II maps out the complete Pareto boundary of alignment weights.
- **Model Compression Trade-offs:** Optimizing the trade-off frontier between model size (quantized bit-width), inference latency, and accuracy on benchmarks.

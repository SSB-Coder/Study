# H02: Genetic Programming (GP) & Symbolic Regression

## 1. Intuition & The "Grand Analogy"
Imagine an engineer writing software who does not write code line-by-line. Instead, they cut up mathematical functions, variable tokens ($x$), and numbers into wooden puzzle blocks and throw them into a giant bucket. They assemble hundreds of random mathematical equation trees. Trees that predict real experimental data with low error are allowed to "mate"—swapping entire branches of code (subtree crossover) or mutating individual operator leaves. Over generations, working formulas evolve automatically from raw syntax trees.

**Genetic Programming (GP)** (Koza, 1992) extends genetic algorithms from fixed-length binary strings to flexible, executable computer programs structured as **Abstract Syntax Trees (ASTs)**.

---

## 2. Architectural & Mathematical Breakdown
A GP program is represented as a tree where:
- **Internal Nodes (Functions):** Operators $\mathcal{F} = \{ +, -, \times, \div \}$.
- **Leaf Nodes (Terminals):** Variables and constants $\mathcal{T} = \{ x, c_1, c_2, \dots \}$.

### Symbolic Regression Objective
Given observed data points $\{ (x_i, y_i) \}_{i=1}^M$, find an expression tree $T$ that minimizes Mean Squared Error:
$$\text{MSE}(T) = \frac{1}{M} \sum_{i=1}^M \left( \text{Eval}(T, x_i) - y_i \right)^2$$
$$\text{Fitness}(T) = \frac{1}{1 + \text{MSE}(T)}$$

### Genetic Tree Operators
1. **Subtree Crossover:** Pick a random node $u \in T_1$ and $v \in T_2$; replace subtree at $u$ with subtree at $v$.
2. **Subtree Mutation:** Pick a random node $u \in T$; replace it with a newly generated random subtree.
3. **Bloat Control:** Parsimony pressure penalizes tree depth or node count to prevent runaway tree expansion:
   $$\text{Fitness}_{\text{adj}}(T) = \text{Fitness}(T) - \lambda \cdot \text{Nodes}(T)$$

---

## 3. Hardcore Mathematical Derivations

### Tree Generation via Ramped Half-and-Half (Koza)
To ensure broad structural diversity:
- **Full Method:** Nodes at depth $d < D_{\max}$ are strictly chosen from function set $\mathcal{F}$; nodes at depth $D_{\max}$ are chosen from terminal set $\mathcal{T}$. All leaves appear at maximum depth.
- **Grow Method:** Nodes at depth $d < D_{\max}$ are chosen from $\mathcal{F} \cup \mathcal{T}$; nodes at $D_{\max}$ are from $\mathcal{T}$. Produces trees of irregular, naturalistic shapes.
- **Ramped Half-and-Half:** For depths $d = 2, \dots, D_{\max}$, allocate $50\%$ of the initial population to Full and $50\%$ to Grow.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Tree-Based Genetic Programming
1. Initialize population of AST trees using Ramped Half-and-Half
2. For generation = 1 to MaxGen:
     a. Evaluate fitness 1 / (1 + MSE) on dataset for each tree
     b. Preserve top elite trees
     c. While next generation not full:
          Select parents T1, T2 via tournament
          If random() < p_crossover:
             child = SubtreeCrossover(T1, T2)
          Else if random() < p_mutation:
             child = SubtreeMutation(T1)
          Else:
             child = Clone(T1)
          Add child to next population (cull if depth > MaxDepth)
3. Return tree with lowest MSE
```

---

## 5. Failure Modes & When It Breaks
1. **Code Bloat:** Without strict parsimony penalties, trees grow exponentially in size without improving fitness, consuming all memory and compute.
2. **Division by Zero:** Division operators require protective wrappers (e.g. protected division $x \div 0 = 1$).

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Program Synthesis & Verifier Guided Search:** Generating Python programs directly to answer mathematical questions (e.g. CodeX, AlphaCode).
- **Symbolic Discovery in Science (AI Feynman):** Discovering exact closed-form physical laws from high-dimensional transformer embeddings.

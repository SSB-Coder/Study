# L04: Constraint Satisfaction Problems (CSP) & Arc Consistency (AC-3)

## 1. Intuition & The "Grand Analogy"
Imagine scheduling classes at a university across 50 lecture halls. You have 300 professors, 5,000 students, and strict requirements: Prof. Smith cannot teach on Fridays, Chem 101 needs a lab bench, and no student can be booked in two exams at 10 AM. If you attempt standard trial-and-error search, you will test trillions of invalid schedules. 

**Constraint Satisfaction Problems (CSPs)** solve this by turning the problem inside out: instead of guessing solutions and evaluating them at the end, every assignment immediately ripples through the network of constraints, automatically erasing impossible choices from neighboring variables (**constraint propagation**). If an arc becomes inconsistent, it is pruned before you ever waste time exploring it.

---

## 2. Architectural & Mathematical Breakdown
A CSP is defined by a triple $\langle X, D, C \rangle$:
- $X = \{ X_1, \dots, X_n \}$: Set of variables.
- $D = \{ D_1, \dots, D_n \}$: Set of domains ($D_i$ is the set of allowable values for $X_i$).
- $C = \{ C_1, \dots, C_m \}$: Set of constraints specifying allowable combinations of values.

### The AC-3 Arc Consistency Algorithm (Mackworth, 1977)
An arc $(X_i, X_j)$ is **arc-consistent** if for every value $x \in D_i$, there exists at least one value $y \in D_j$ satisfying the binary constraint between $X_i$ and $X_j$.
If not, $x$ is permanently removed from $D_i$.

### Backtracking Heuristics
1. **Minimum Remaining Values (MRV / "Fail-First"):** Choose the unassigned variable with the fewest legal values in its domain.
2. **Degree Heuristic:** Choose the variable involved in the largest number of constraints with other unassigned variables.
3. **Least Constraining Value (LCV):** Order values to leave the maximum flexibility for remaining variables.

---

## 3. Hardcore Mathematical Derivations

### Time Complexity of the AC-3 Algorithm
Let $n$ be the number of variables, $d = \max |D_i|$ the maximum domain size, and $c$ the number of binary constraint arcs:
- A binary constraint between $X_i$ and $X_j$ yields two directed arcs $(X_i, X_j)$ and $(X_j, X_i)$. The total number of arcs is $2c$.
- Each arc $(X_i, X_j)$ can be inserted into the AC-3 queue at most $d$ times, because variable $X_i$ has at most $d$ values that can be deleted.
- Testing consistency of an arc takes $\mathcal{O}(d^2)$ operations.
Total worst-case time complexity of AC-3 is strictly:
$$\mathcal{O}(c \cdot d^3)$$
This polynomial pre-processing step frequently prunes 90%+ of the exponential search tree before backtracking even begins.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: AC-3 (Arc Consistency Algorithm #3)
Inputs: CSP with variables X, domains D, binary constraints C
1. Queue = all directed arcs (X_i, X_j) in C
2. While Queue is not empty:
     pop (X_i, X_j) from Queue
     If Revise(CSP, X_i, X_j):
        If |D_i| == 0:
           return False (Inconsistency detected! No solution possible)
        For each X_k in Neighbors(X_i) \ {X_j}:
           add (X_k, X_i) to Queue
3. Return True (CSP is arc-consistent)

Function Revise(CSP, X_i, X_j):
  revised = False
  For each x in D_i:
    If no y in D_j satisfies constraint(x, y):
       remove x from D_i
       revised = True
  return revised
```

---

## 5. Failure Modes & When It Breaks
1. **Arc Consistency is Not Global Consistency:** A CSP can be 100% arc-consistent, yet possess zero global solutions (e.g. 3-coloring a 4-clique with only 2 colors). Higher-order $k$-consistency is required to eliminate backtracking entirely.
2. **Domain Explosions:** Continuous or unbounded integer domains require interval arithmetic solvers.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Grammar & JSON-Constrained Decoding:** Enforcing that an LLM outputs strictly valid JSON or conforms to a Pydantic schema is implemented as dynamic constraint filtering over the vocabulary logits at each token generation step.

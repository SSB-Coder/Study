# Deep Dive: W09 — SAT Solving (The DPLL Algorithm)

## 1. The Beginner's Analogy: The Deductive Logic Puzzle
Imagine solving an intricate logic grid puzzle with 100 interlocking clues:
- Some clues are blunt: "Suspect 1 was wearing red shoes" (Unit Clause). You immediately lock that down as fact and cross out any clue saying suspect 1 had blue shoes.
- Other clues are one-sided: "Nobody in the building ever mentioned poison" (Pure Literal). You immediately rule poison out.
- When no obvious clues remain, you make a tentative guess and follow the deduction chain. If you hit a blatant contradiction ("Suspect 2 was in London and Paris at 3 PM"), you rewind your guess and take the opposite branch. **The DPLL algorithm** is this exact disciplined detective search.

---

## 2. The Deep Architecture/Algorithm Breakdown
Boolean Satisfiability (SAT) was the first problem proven **NP-complete** (Cook-Levin Theorem, 1971):
1. **Conjunctive Normal Form (CNF):** A boolean formula represented as a conjunction of clauses $\bigwedge_i C_i$, where each clause is a disjunction of literals $C_i = (l_1 \lor l_2 \lor \dots \lor l_k)$.
2. **Davis-Putnam-Logemann-Loveland (DPLL):** The recursive backtracking algorithm (1962) forming the foundation of all modern SAT and SMT solvers:
   - *Unit Propagation:* If a clause contains a single literal $l$, $l$ must evaluate to True.
   - *Pure Literal Rule:* If variable $x$ appears with only one polarity in the whole formula, assign it that value.
3. **CDCL Evolution:** Modern solvers (Z3, MiniSat) extend DPLL with Conflict-Driven Clause Learning (CDCL), non-chronological backjumping, and watched literal data structures.

---

## 3. The Hardcore Mathematics

### 3.1 Unit Clause Rule
Let clause $C = (l_1)$. In CNF, all clauses must hold true:
$$\left( \bigwedge_{i \ne k} C_i \right) \land (l_1) = 1 \implies l_1 = 1$$
Setting literal $l_1 = 1$ allows simplifying the formula $\Phi$:
$$\Phi |_{l_1=1} = \{ C_i \setminus \{\neg l_1\} \mid C_i \in \Phi, l_1 \notin C_i \}$$

### 3.2 Pure Literal Elimination
If literal $l$ appears in $\Phi$ but $\neg l \notin \bigcup_{C \in \Phi} C$:
Setting $l = 1$ satisfies all clauses containing $l$ without shortening any other clause.

---

## 4. The Algorithm Step-by-Step
1. **Unit Propagate:** Find all clauses of length 1. Assign literals and simplify. If empty clause $\emptyset$ appears, return `UNSAT`.
2. **Pure Literals:** Find literals with single polarity. Assign and simplify.
3. **Check Termination:** If clause set is empty, return `SAT` with model.
4. **Branch:** Choose variable $x$ via branching heuristic (e.g. DLIS or VSIDS).
   - Recurse on $\Phi |_{x=1}$. If SAT, return solution.
   - Else recurse on $\Phi |_{x=0}$. If SAT, return solution.
   - Else return `UNSAT`.

---

## 5. Limitations / When It Breaks
- **Worst-Case Exponential Time:** For pigeonhole principle formulas or cryptographic hash inversions, plain DPLL requires $\mathcal{O}(2^n)$ steps without clause learning.

---

## 6. Where This Fits
- **Cross-Reference CSP (L04):** SAT is a Constraint Satisfaction Problem with boolean domains and clausal constraints.
- **Model Checking & Bounded Model Checking (BMC):** Translating digital hardware state transitions unrolled for $k$ clock cycles into a SAT formula to mathematically prove bugs cannot exist.

# N03: Symbolic Expression Trees & Graph Simplification

## 1. The Beginner's Analogy
When you use a computer algebra system like SymPy or Mathematica, the computer doesn't treat $x^2 + 2x$ as numbers—it treats them as a **family tree of symbols**:
- At the top of the tree sits the addition operator `+`.
- Its left child is an exponentiation branch `x ^ 2`.
- Its right child is a multiplication branch `2 * x`.

If you ask the computer to differentiate this tree, it walks through each branch, mechanically applying the rules of calculus you learned in school (product rule, power rule, chain rule). 
However, raw symbolic differentiation produces "expression swell"—formulas like $(1 \cdot x + x \cdot 0) + (0 \cdot x + 2 \cdot 1)$. 
**Graph Simplification** acts like a master algebraic editor: it sweeps through the expression tree, pruning $x \times 0 \to 0$, reducing $x + 0 \to x$, and folding numerical constants $2 + 3 \to 5$, leaving a clean, minimal, human-readable formula: $2x + 2$.

## 2. The Deep Architecture & Mechanics
Symbolic computation operates on explicit expression trees rather than numerical arrays:
1. **Abstract Syntax Tree (AST):**
   - **Leaf Nodes:** Variables (`Var('x')`) and scalar constants (`Const(3.0)`).
   - **Operator Nodes:** Binary operations (`Add`, `Mul`, `Div`, `Pow`) and unary operations (`Sin`, `Cos`, `Exp`, `Log`).
2. **Recursive Transformation Rules:**
   - Differentiating an expression node returns a *new expression node* composed of its children and their derivatives.
3. **Algebraic Rewriting & Constant Folding Pass:**
   - Evaluates sub-trees consisting entirely of constants at compile time.
   - Eliminates additive identities ($u + 0 \to u$) and multiplicative identities ($u \cdot 1 \to u$, $u \cdot 0 \to 0$, $u^1 \to u$, $u^0 \to 1$).
4. **Code Generation (JIT Compilation):**
   - Compiles the simplified symbolic tree into a high-speed NumPy vectorized callable lambda.

## 3. The Hardcore Mathematics

### Recursive Tree Differentiation Operator $\mathcal{D}_x$
1. **Base Cases:**
   $$\mathcal{D}_x[\text{Const}(c)] = 0$$
   $$\mathcal{D}_x[\text{Var}(y)] = \begin{cases} 1 & \text{if } y = x \\ 0 & \text{if } y \ne x \end{cases}$$
2. **Sum Rule:**
   $$\mathcal{D}_x[u + v] = \mathcal{D}_x[u] + \mathcal{D}_x[v]$$
3. **Product Rule:**
   $$\mathcal{D}_x[u \cdot v] = \mathcal{D}_x[u] \cdot v + u \cdot \mathcal{D}_x[v]$$
4. **Quotient Rule:**
   $$\mathcal{D}_x\left[\frac{u}{v}\right] = \frac{\mathcal{D}_x[u] \cdot v - u \cdot \mathcal{D}_x[v]}{v^2}$$
5. **Chain Rule for Non-linear Transcendental Operators:**
   $$\mathcal{D}_x[\sin(u)] = \cos(u) \cdot \mathcal{D}_x[u]$$
   $$\mathcal{D}_x[\exp(u)] = \exp(u) \cdot \mathcal{D}_x[u]$$

### Canonical Simplification Equivalence Rules
Let $\mathcal{S}[E]$ be the canonical recursive simplification operator:
- **Constant Folding:** $\mathcal{S}[\text{Const}(c_1) \oplus \text{Const}(c_2)] = \text{Const}(c_1 \oplus c_2)$.
- **Additive Identities:** $\mathcal{S}[u + 0] = \mathcal{S}[0 + u] = \mathcal{S}[u]$.
- **Multiplicative Identities:** $\mathcal{S}[u \cdot 1] = \mathcal{S}[1 \cdot u] = \mathcal{S}[u]$.
- **Multiplicative Annihilation:** $\mathcal{S}[u \cdot 0] = \mathcal{S}[0 \cdot u] = \text{Const}(0)$.
- **Exponent Identities:** $\mathcal{S}[u^1] = \mathcal{S}[u]$, $\mathcal{S}[u^0] = \text{Const}(1)$, $\mathcal{S}[1^u] = \text{Const}(1)$.

## 4. The Algorithm Step-by-Step
1. Construct the target expression using operator-overloaded Python classes.
2. Call `.diff('x')` to generate the raw symbolic derivative tree.
3. Call `.simplify()` repeatedly until the tree reaches an algebraic fixed point:
   $$\mathcal{S}[E^{(t+1)}] = \mathcal{S}[E^{(t)}]$$
4. Compile the simplified tree into an executable vectorized function via `.compile()`.

## 5. Limitations & Failure Modes
- **Expression Swell:** Unsimplified symbolic derivatives grow exponentially with formula depth $\mathcal{O}(2^d)$, causing catastrophic memory bloat for deep neural networks (which is why reverse-mode autodiff is used for neural nets instead).
- **Equivalence Undecidability (Richardson's Theorem):** Determining whether two arbitrary non-linear elementary expressions are mathematically identical is generally undecidable in theoretical computer science.

## 6. Where This Fits
Cross-reference: Symbolic manipulation complements reverse-mode (N01) and forward-mode (N02) autodiff, serving as the compilation pass in deep learning graph compilers (like XLA, TorchScript, and TVM) that fuses operations and eliminates redundant computations.

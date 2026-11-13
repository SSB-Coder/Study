# Deep Dive: Automated Theorem Proving, Lean 4 & Neural Symbolic Discovery

## 1. The Crisis of Rigor: Probabilistic LLMs vs. Formal Mathematics
Large Language Models (LLMs) predict tokens based on statistical likelihood:
$$P(w_t \mid w_1, \dots, w_{t-1})$$
While this works brilliantly for prose, code generation, and intuitive problem-solving, it has a fatal flaw in pure mathematics and formal verification: **hallucination**.

In mathematics, a proof that is "99% correct" is **100% false**. A single subtle sign error, circular definition, or unjustified division by zero invalidates an entire 100-page proof.

To achieve genuine mathematical discovery and verifiable engineering safety (e.g. verified avionics software, cryptographic protocols, chip tape-out validation), researchers pair neural networks with **Interactive Theorem Provers (ITPs)** such as **Lean 4**, **Isabelle/HOL**, and **Coq**.

---

## 2. Interactive Theorem Provers & Lean 4
In Lean 4, mathematical statements and proofs are expressed in a formal language based on the **Calculus of Inductive Constructions (Dependent Type Theory)**.

Under the **Curry-Howard Isomorphism**:
* A mathematical theorem is a **Type**.
* A valid mathematical proof is a **Program** (an expression) that inhabits that type.
* Verifying a proof is simply running the compiler's **Type Checker**!

```lean
-- A formal theorem in Lean 4
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction m with
  | zero => rw [Nat.add_zero, Nat.zero_add]
  | succ m ih => rw [Nat.add_succ, Nat.succ_add, ih]
```

When Lean's kernel compiles the code without errors, the proof is mathematically guaranteed to be $100\%$ true with absolute mathematical certainty.

---

## 3. The Neural Theorem Proving Architecture

To automate formal proof creation, researchers use a dual-component architecture: **The Neural Proposer + The Formal Kernel Verifier**.

```
Current Proof State (Goal)
[ e.g. ∀ n, 2 * n % 2 = 0 ]
            │
            ▼
 ┌──────────────────────┐
 │  Transformer Model   │ ──> Proposes Next Tactics:
 │ (Fine-tuned on Math) │     1. "intro n; ring"
 └──────────────────────┘     2. "omega"
            │                 3. "induction n"
            ▼
 ┌──────────────────────┐
 │ Lean 4 Proof Kernel  │ ──> Deterministic Verification
 │ (No Hallucinations!) │     Accepts valid tactic -> New Goal
 └──────────────────────┘     Rejects invalid tactic -> Backtrack
            │
            ▼
[ Tree Search: MCTS / BFS ] ──> Closed Proof Q.E.D.
```

1. **Premise Selection (Dense Retrieval):** Lean's mathematical library (*Mathlib*) contains over 100,000 theorems and definitions. When attempting to prove a conjecture, a bi-encoder transformer retrieves the most relevant auxiliary lemmas.
2. **Neural Tactic Generation:** A sequence-to-sequence autoregressive transformer reads the current proof state goal and generates candidates for the next formal proof tactic (e.g., `induction x`, `simp`, `linarith`).
3. **Proof Tree Search (MCTS / Best-First Search):** The search tree explores multiple proof paths. The Lean 4 environment acts as an unyielding environment oracle: invalid tactics fail immediately, completely eliminating LLM hallucinations.

---

## 4. Neural Symbolic Discovery: Rediscovering Laws of Nature
Beyond formal proofs, a major frontier of scientific AI is **Symbolic Regression**: extracting exact closed-form analytic equations from noisy experimental sensor data.

### Why Standard Neural Networks Are Insufficient for Science:
If you train a 1-billion-parameter Transformer on planetary orbits, it can predict where Mars will be tomorrow, but it cannot hand you Kepler's Third Law:
$$\frac{T^2}{a^3} = \frac{4\pi^2}{G(M + m)}$$
The physics is locked inside billions of inscrutable floating-point matrix weights (a black box).

### Neural Symbolic Frameworks:
1. **Kolmogorov-Arnold Networks (KAN):** By placing learnable spline curves on edges, pruning near-zero edges, and snapping learned curves to analytic functions ($\sin, \exp, x^2$), KANs can rediscover physical conservation laws directly from data.
2. **FunSearch (DeepMind, *Nature* 2024):** Couples a pre-trained LLM that writes code solutions in Python with an automated program evaluator. Through evolutionary iteration, FunSearch discovered brand-new, provably optimal mathematical constructions for the **Cap Set Problem** in extremal combinatorics, surpassing the best human mathematicians.

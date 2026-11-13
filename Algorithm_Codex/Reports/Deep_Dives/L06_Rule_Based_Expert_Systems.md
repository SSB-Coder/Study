# L06: Rule-Based Expert Systems

## 1. The Beginner's Analogy
Imagine an experienced physician diagnosing a complex disease or an aviation engineer troubleshooting a jet engine. Over decades of practice, their expertise has crystallized into a structured mental handbook of rules:
*"IF the turbine pressure drops rapidly AND exhaust gas temperature spikes, THEN there is a fuel nozzle blockage."*
A **Rule-Based Expert System** encodes human domain expertise into formal production rules and working memory facts, applying formal deductive logic to solve diagnostic, configuration, and compliance problems without black-box statistical opacity.

## 2. The Deep Architecture & Mechanics
Classical Expert Systems consist of three core architectural components:
1. **Knowledge Base (Rule Base):** Permanent domain knowledge stored as production rules: $\text{IF } \text{Condition}_1 \land \dots \land \text{Condition}_m \text{ THEN } \text{Action / Consequent}$.
2. **Working Memory (Fact Base):** Dynamic repository of currently asserted ground facts reflecting the problem instance.
3. **Inference Engine:** The reasoning mechanism that matches rules against facts:
   - **Forward Chaining (Data-Driven / Bottom-Up):** Starts with observed evidence and fires matching rules to deduce all reachable conclusions until quiescence.
   - **Backward Chaining (Goal-Driven / Top-Down):** Starts with a candidate hypothesis or query goal and searches backwards to verify whether supporting evidence exists in working memory.

## 3. The Hardcore Mathematics

### Modus Ponens
The foundational deductive inference rule:
$$\frac{P, \quad P \implies Q}{Q}$$
If premise $P$ is true, and implication $P \implies Q$ holds, conclusion $Q$ is rigorously derived.

### Forward-Chaining Fixed-Point Semantics
Let $\mathcal{F}_0$ be the initial fact set. At iteration $k+1$, the deductive closure operator $T_{\mathcal{R}}$ generates:
$$\mathcal{F}_{k+1} = \mathcal{F}_k \cup \left\{ Q : \exists (P \implies Q) \in \mathcal{R} \text{ such that } P \subseteq \mathcal{F}_k \right\}$$
Because $\mathcal{F}_k \subseteq \mathcal{F}_{k+1}$ is monotonic and bounded by the finite Herbrand base, it reaches a unique minimal fixed point:
$$T_{\mathcal{R}}^\infty(\mathcal{F}_0) = \bigcup_{k=0}^\infty \mathcal{F}_k$$

### Backward-Chaining AND/OR Goal Trees
Proving goal $G$ decomposes into an AND/OR search tree:
- **OR Nodes:** Multiple rules $R_1, R_2, \dots$ conclude $G$. Proving *any* single rule succeeds.
- **AND Nodes:** A specific rule requires conditions $P_1 \land P_2 \dots \land P_m$. *Every* condition must be proven.

## 4. The Algorithm Step-by-Step
1. **Assert Facts:** Populate working memory with known ground observations $\mathcal{F}$.
2. **Match Phase:** Compare rule antecedents against working memory to find the active conflict set.
3. **Conflict Resolution:** If multiple rules match, resolve priority via Specificity (most antecedents), Recency, or Refractoriness.
4. **Act/Fire Phase:** Execute the chosen rule's consequent, inserting new facts into working memory.
5. **Terminate:** Halt when goal is satisfied or no new rules fire.

## 5. Limitations & Failure Modes
- **Knowledge Acquisition Bottleneck:** Hand-crafting thousands of production rules from human interviews is labor-intensive and fragile.
- **Brittleness under Uncertainty:** Fails when input observations are noisy, contradictory, or missing (led to the development of Bayesian Networks and fuzzy logic).

## 6. Where This Fits
Cross-reference: Expert systems are the direct symbolic ancestor of modern Chain-of-Thought (CoT) prompting, declarative LLM Guardrails (NeMo Guardrails), and neurosymbolic reasoning hybrids.

# L08: Game Theory & Multi-Agent Equilibria (Nash Equilibrium)

## 1. Intuition & The "Grand Analogy"
Imagine two competing gas stations located at the same intersection. 
- If both charge regular prices ($4/gal), both make a healthy profit ($5,000/day).
- If Station A slashes prices to $3.50 while Station B remains at $4, Station A steals all the customers ($8,000 profit) while Station B goes bankrupt ($0 profit).
- Knowing this, Station B slashes their price too. Both end up selling at $3.50, barely breaking even ($1,000 profit).
Neither station can raise prices unilaterally without going out of business. They are trapped in a **Nash Equilibrium**.

A **Nash Equilibrium** (John Nash, Nobel Prize 1994) is a profile of strategies where no player can unilaterally deviate to improve their expected payoff.

---

## 2. Architectural & Mathematical Breakdown
A normal-form game is defined by:
- $N$ players $i \in \{1, \dots, N\}$.
- Action spaces $S_i$ for each player.
- Payoff utility functions $u_i(s_1, \dots, s_N)$.

### Pure Strategy Nash Equilibrium (PNE)
A pure strategy profile $s^* = (s_1^*, \dots, s_N^*)$ is a Nash Equilibrium if:
$$u_i(s_i^*, s_{-i}^*) \ge u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i, \; \forall i \in \{1, \dots, N\}$$

### Mixed Strategy Nash Equilibrium (MNE)
Players randomize their actions according to probability distributions $\sigma_i \in \Delta(S_i)$.
Expected utility:
$$U_i(\sigma_i, \sigma_{-i}) = \sum_{s \in S} \left( \prod_{j=1}^N \sigma_j(s_j) \right) u_i(s)$$
Profile $\sigma^*$ is an equilibrium if:
$$U_i(\sigma_i^*, \sigma_{-i}^*) \ge U_i(\sigma_i, \sigma_{-i}^*) \quad \forall \sigma_i \in \Delta(S_i)$$

---

## 3. Hardcore Mathematical Derivations

### Nash's Existence Theorem via Kakutani's Fixed Point Theorem
Let $\Sigma = \prod_{i=1}^N \Delta(S_i)$ be the compact, convex strategy space.
Define the best-response correspondence for player $i$:
$$B_i(\sigma_{-i}) \triangleq \arg\max_{\sigma_i \in \Delta(S_i)} U_i(\sigma_i, \sigma_{-i})$$
The joint best-response correspondence is $B(\sigma) = \prod_{i=1}^N B_i(\sigma_{-i})$.
- $\Sigma$ is non-empty, compact, and convex.
- Since $U_i$ is linear in $\sigma_i$, $B_i(\sigma_{-i})$ is convex and closed.
By **Kakutani's Fixed-Point Theorem**, there exists at least one strategy profile $\sigma^* \in \Sigma$ such that:
$$\sigma^* \in B(\sigma^*)$$
**Theorem (Nash, 1950):** Every finite game has at least one mixed-strategy Nash equilibrium.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Pure Strategy Nash Equilibrium Finder
Inputs: Payoff matrices U1, U2 for 2-player game
1. For each column c in Player 2's actions:
     Find row r* that maximizes U1[r, c] (Player 1's best responses)
2. For each row r in Player 1's actions:
     Find column c* that maximizes U2[r, c] (Player 2's best responses)
3. Intersect best response pairs:
     Any cell (r, c) where r is a best response to c AND c is a best response to r
     is a Pure Strategy Nash Equilibrium!
4. Return all PNE cells
```

---

## 5. Failure Modes & When It Breaks
1. **Inefficiency (Price of Anarchy):** Nash equilibria are frequently Pareto-suboptimal (e.g. Prisoner's Dilemma defect/defect is far worse than cooperate/cooperate).
2. **Equilibrium Selection Problem:** Games with multiple equilibria (e.g. Battle of the Sexes, Stag Hunt) offer no intrinsic mechanism for players to coordinate on the same equilibrium without communication.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Multi-Agent Negotiation Protocols:** Designing auction and barter mechanisms where autonomous AI agents negotiate contracts.
- **Direct Preference Optimization (DPO):** Recent literature shows DPO aligns models by computing the Nash equilibrium of an implicit two-player game between generator and reference model.

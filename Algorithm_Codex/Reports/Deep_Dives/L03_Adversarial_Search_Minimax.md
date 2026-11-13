# L03: Adversarial Search & Alpha-Beta Pruning

## 1. Intuition & The "Grand Analogy"
Imagine playing chess against a ruthless grandmaster. You cannot simply plan your own moves assuming your opponent will politely make mistakes to let you win. Every time you consider a move that opens a brilliant attack, you must immediately assume the opponent will identify the most vicious, punishing reply to nullify your advantage.

**Minimax Search** (von Neumann, 1928) formalizes optimal play in two-player zero-sum games: the MAX player chooses actions to maximize their payoff, while the MIN player chooses actions to minimize MAX's payoff.
**Alpha-Beta Pruning** is a mathematical theorem of pure efficiency: if you discover that a candidate move allows the opponent a devastating counterattack that is already worse than an alternative move you safely established earlier, you stop evaluating that branch immediately. You prune the subtree without evaluating a single additional descendant.

---

## 2. Architectural & Mathematical Breakdown
Let $s$ be a game state. $u(s)$ is the terminal utility function:
$$\text{Minimax}(s) = \begin{cases} u(s) & \text{if } \text{IsTerminal}(s) \\ \max_{a \in \text{Actions}(s)} \text{Minimax}(\text{Result}(s, a)) & \text{if } \text{Player}(s) = \text{MAX} \\ \min_{a \in \text{Actions}(s)} \text{Minimax}(\text{Result}(s, a)) & \text{if } \text{Player}(s) = \text{MIN} \end{cases}$$

### Alpha-Beta Bounding Parameters
- $\alpha$: Highest (best) utility discovered so far along any choice path for **MAX**.
- $\beta$: Lowest (best) utility discovered so far along any choice path for **MIN**.
Condition for Pruning:
$$\text{Prune branch when } \alpha \ge \beta$$

---

## 3. Hardcore Mathematical Derivations

### The Alpha-Beta Pruning Criterion
Consider a MIN node $n$ with current value $v = \min_i v_i$.
Suppose one of its examined children has value $v_k \le \alpha$.
Since $n$ is a MIN node, its final value cannot possibly exceed $v_k$:
$$v(n) \le v_k \le \alpha$$
However, MAX at an ancestor node already has an alternative option with confirmed payoff at least $\alpha$. Since MAX will never willingly choose a path leading to a value $\le \alpha$, MAX will never allow game state $n$ to be reached.
Therefore, all remaining unexplored children of node $n$ can be safely discarded with **zero impact on the final minimax decision**.

### Computational Savings
- Full Minimax evaluates $\mathcal{O}(b^d)$ leaf nodes.
- With **perfect move ordering** (evaluating best moves first), Alpha-Beta pruning cuts the effective branching factor from $b$ to $\sqrt{b}$:
  $$\mathcal{O}(b^{d/2})$$
This effectively **doubles the search depth** accessible within the same computational time budget (e.g. from 10 ply to 20 ply in chess).

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Alpha-Beta Pruning
Function MinimaxAB(state, depth, alpha, beta, is_max):
1. If depth == 0 or IsTerminal(state):
     return Evaluate(state)
2. If is_max:
     max_eval = - infinity
     For each action in Actions(state):
       eval = MinimaxAB(Result(state, action), depth - 1, alpha, beta, False)
       max_eval = max(max_eval, eval)
       alpha = max(alpha, eval)
       If beta <= alpha:
          break  # Beta cutoff (prune remaining siblings)
     return max_eval
3. Else (is_min):
     min_eval = + infinity
     For each action in Actions(state):
       eval = MinimaxAB(Result(state, action), depth - 1, alpha, beta, True)
       min_eval = min(min_eval, eval)
       beta = min(beta, eval)
       If beta <= alpha:
          break  # Alpha cutoff (prune remaining siblings)
     return min_eval
```

---

## 5. Failure Modes & When It Breaks
1. **The Horizon Effect:** A critical tactical blunder or queen capture is postponed beyond the maximum search depth by forcing a sequence of pointless checks, leading to flawed evaluations.
2. **Poor Move Ordering:** In the worst-case move ordering (examining worst moves first), Alpha-Beta performs zero pruning, degrading back to $\mathcal{O}(b^d)$.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Self-Play & Adversarial Red-Teaming:** Pitting a generator LLM against a critic/adversary LLM to probe boundary cases and discover jailbreaks.
- **Game-Theoretic Reasoning in Debates:** Two LLMs debating opposing viewpoints to arrive at calibrated truth.

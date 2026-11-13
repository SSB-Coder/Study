# G11: Multi-Agent Reinforcement Learning Basics (MARL & Minimax-Q)

## 1. Intuition & The "Grand Analogy"
Imagine playing Rock-Paper-Scissors against an intelligent opponent. If you treat the game as a standard single-agent environment, you might observe that the opponent threw Scissors on round 1 and conclude: *"Rock gives +1 reward! Rock is the optimal action!"* But as soon as you repeatedly throw Rock, the opponent adapts and throws Paper. From your individual perspective, the environment is fundamentally **non-stationary**—the laws of physics are changing because another learning mind is adapting against you.

**Multi-Agent Reinforcement Learning (MARL)** models interactions between multiple strategic decision-makers. In two-player zero-sum games, the goal is not merely to exploit a fixed environment, but to compute a **Nash Equilibrium**: a mixed strategy where neither player can gain an advantage by unilaterally altering their policy.

---

## 2. Architectural & Mathematical Breakdown
A two-player zero-sum Markov game is defined by state space $\mathcal{S}$, action spaces $\mathcal{A}_1, \mathcal{A}_2$, and reward function $R_1(s, a_1, a_2) = - R_2(s, a_1, a_2)$.

### Minimax-Q Learning (Littman, 1994)
Replaces the single-agent $\max_a Q(s, a)$ operator with the **minimax value** over joint actions:
$$V(s) = \max_{\pi \in \Delta(\mathcal{A}_1)} \min_{a_2 \in \mathcal{A}_2} \sum_{a_1 \in \mathcal{A}_1} \pi(a_1) Q(s, a_1, a_2)$$

### Update Rule
Upon observing joint transition $(s, a_1, a_2, r, s')$:
$$Q(s, a_1, a_2) \leftarrow Q(s, a_1, a_2) + \alpha \left[ r + \gamma V(s') - Q(s, a_1, a_2) \right]$$

---

## 3. Hardcore Mathematical Derivations

### Linear Programming Formulation for Minimax Value
Computing $V(s) = \max_\pi \min_{a_2} \sum_{a_1} \pi(a_1) Q(s, a_1, a_2)$ is solved via Linear Programming:
$$\begin{aligned}
\text{Maximize}_{V, \pi} \quad & V \\
\text{Subject to} \quad & \sum_{a_1 \in \mathcal{A}_1} \pi(a_1) Q(s, a_1, a_2) \ge V \quad \forall a_2 \in \mathcal{A}_2 \\
& \sum_{a_1 \in \mathcal{A}_1} \pi(a_1) = 1, \quad \pi(a_1) \ge 0 \quad \forall a_1 \in \mathcal{A}_1
\end{aligned}$$
By Von Neumann's Minimax Theorem, in any zero-sum matrix game:
$$\max_{\pi_1} \min_{\pi_2} \pi_1^\top Q \pi_2 = \min_{\pi_2} \max_{\pi_1} \pi_1^\top Q \pi_2 = V^*$$

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Minimax-Q
1. Initialize Q(s, a1, a2) = 0, V(s) = 0, policy pi(s) = uniform
2. For each step:
     Choose action a1 according to exploration policy (e.g. (1-eps)*pi(s) + eps*random)
     Observe opponent action a2, reward r, next state s'
     Q(s, a1, a2) = Q(s, a1, a2) + alpha * [ r + gamma * V(s') - Q(s, a1, a2) ]
     Solve Linear Program on matrix Q(s, ., .) to find new V(s) and optimal mixed strategy pi(s)
     s = s'
```

---

## 5. Failure Modes & When It Breaks
1. **Curse of Multi-Agent Non-Stationarity:** In Independent Q-Learning (IQL), agents treat others as part of the environment, causing convergence to fail and cyclic behavior.
2. **Exponential Action Joint Space:** With $N$ agents each having $K$ actions, the joint action space $|\mathcal{A}|^N$ explodes exponentially.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Self-Play in Reasoning Models:** Models like AlphaGo, AlphaZero, and modern mathematical LLM debates pit two LLM agents against each other to surface logical flaws and achieve robust equilibrium.
- **Direct Preference Optimization (DPO) as a Two-Player Game:** Recent theoretical literature proves DPO optimizes a minimax equilibrium between the generator policy and an implicit adversary.

# G02: Markov Decision Processes & Dynamic Programming

## 1. Intuition & The "Grand Analogy"
Imagine navigating a grid of rooms in a museum where some doorways lead toward the exit, others trap you in dead ends, and walking through any corridor costs energy. If you possess a complete architectural blueprint of the museum and know the exact probability of slipping on wet floors, you do not need trial-and-error exploration. You can calculate the exact optimal trajectory from every single room sitting at your desk.

This is **Dynamic Programming (DP)** applied to **Markov Decision Processes (MDPs)**. It represents planning under perfect model dynamics: when transition probabilities $P(s' \mid s, a)$ and reward distributions $R(s, a, s')$ are fully known, the optimal value function satisfies Bellman's principle of optimality and can be solved by fixed-point contraction mappings.

---

## 2. Architectural & Mathematical Breakdown
An MDP is formally defined by a 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$
- $\mathcal{S}$: State space.
- $\mathcal{A}$: Action space.
- $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$: Transition probability tensor.
- $\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$: Expected reward function.
- $\gamma \in [0, 1)$: Discount factor.

### Bellman Expectation Equations (Under Policy $\pi$)
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$
$$Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$$

### Bellman Optimality Equations
$$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

---

## 3. Hardcore Mathematical Derivations

### Contraction Mapping & Banach Fixed-Point Theorem
Define the Bellman Optimality Operator $T^*$ on value vectors $V \in \mathbb{R}^{|\mathcal{S}|}$:
$$(T^* V)(s) \triangleq \max_a \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') \right]$$

For any two value functions $U, V$:
$$|(T^* U)(s) - (T^* V)(s)| \le \max_a \left| \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) (U(s') - V(s')) \right| \le \gamma \| U - V \|_\infty$$
Taking the supremum over all states $s$:
$$\| T^* U - T^* V \|_\infty \le \gamma \| U - V \|_\infty$$
Since $\gamma < 1$, $T^*$ is a **$\gamma$-contraction mapping** in the $L_\infty$ norm. By the Banach Fixed-Point Theorem:
1. $V^*$ is the unique fixed point satisfying $T^* V^* = V^*$.
2. Repeated application $V_{k+1} = T^* V_k$ converges linearly at rate $\gamma^k$ to $V^*$ regardless of initial value $V_0$.

---

## 4. Step-by-Step Algorithmic Walkthrough

### Value Iteration
```
Algorithm: Value Iteration
Input: MDP (S, A, P, R, gamma), threshold theta > 0
1. Initialize V(s) = 0 for all s in S
2. Repeat:
     Delta = 0
     For each s in S:
       v = V(s)
       V(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s') ]
       Delta = max(Delta, |v - V(s)|)
   Until Delta < theta
3. Extract deterministic policy:
     pi*(s) = argmax_a [ R(s, a) + gamma * sum_{s'} P(s' | s, a) * V(s') ]
4. Return V*, pi*
```

### Policy Iteration
Alternates two decoupled steps:
1. **Policy Evaluation:** Solve linear system $V^{\pi_k} = (I - \gamma \mathcal{P}^{\pi_k})^{-1} \mathcal{R}^{\pi_k}$.
2. **Policy Improvement:** Set $\pi_{k+1}(s) = \arg\max_a \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^{\pi_k}(s') \right]$.
Terminates strictly when $\pi_{k+1} = \pi_k$.

---

## 5. Failure Modes & When It Breaks
1. **Curse of Dimensionality:** Requires iterating over all states and actions. When state space $|\mathcal{S}|$ is continuous or exponentially large (e.g. combinatorial boards or robotics), tabular DP is computationally impossible.
2. **Model Assumption Violation:** Requires explicit ground-truth transition dynamics $\mathcal{P}(s' \mid s, a)$ and reward distributions $\mathcal{R}(s, a)$, which are rarely accessible in physical real-world systems.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Prefix / Constrained Decoding:** Grammar-guided and regex-constrained generation operates as dynamic programming over the finite state automaton of the grammar.
- **Value-Guided Search in LLMs:** DeepSeek-R1 and Tree-of-Thoughts evaluate intermediate reasoning steps using learned value functions $V(s)$ derived from Bellman principles.

# G10: Model-Based Reinforcement Learning & Monte Carlo Tree Search (MCTS)

## 1. Intuition & The "Grand Analogy"
Imagine a grandmaster playing chess. When it is their turn to move, they do not simply react on instinct or wait to see what happens after moving a piece. They lean back, close their eyes, and simulate mental trees of futures: *"If I move knight to f3, they will respond pawn to d5, then I sacrifice bishop on h7..."* They run hundreds of hypothetical rollouts in their internal world model before ever touching a physical piece.

**Model-Based RL** equips an agent with an internal predictive simulator:
- In **Dyna-Q** (Sutton, 1990), real experiences train a model $T(s, a) \to (s', r)$, and during quiet moments the agent dreams thousands of simulated transitions to update Q-values.
- In **Monte Carlo Tree Search (MCTS)** (Coulom 2006, AlphaGo / AlphaZero), the agent plans online by expanding an asymmetric decision tree guided by Upper Confidence Bounds for Trees (UCT).

---

## 2. Architectural & Mathematical Breakdown

### Dyna-Q Architecture
Combines three tightly coupled loops:
1. **Acting & Real Learning:** Take real step $S \xrightarrow{A} R, S'$, update $Q(S, A)$ via Q-learning.
2. **Model Learning:** Record empirical transition $\text{Model}(S, A) \leftarrow (R, S')$.
3. **Planning:** Sample $N$ previously visited states and actions uniformly from $\text{Model}$, and perform $N$ additional Q-learning updates on simulated hallucinations.

### The MCTS 4-Phase Cycle
1. **Selection:** Traverse tree from root using UCT until a leaf node is reached:
   $$\text{UCT}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$
2. **Expansion:** If leaf is non-terminal, add one or more child nodes.
3. **Simulation (Rollout):** Run lightweight default rollout policy until termination to obtain return $G$.
4. **Backpropagation:** Propagate $G$ upward to update visitation counts $N(s, a) \leftarrow N(s, a) + 1$ and mean action values $Q(s, a) \leftarrow Q(s, a) + \frac{G - Q(s, a)}{N(s, a)}$.

---

## 3. Hardcore Mathematical Derivations

### Sample Efficiency Bound: Model-Free vs. Model-Based
In a finite MDP with horizon $H$ and tolerance $\epsilon$:
- Model-free Q-learning requires $\tilde{\mathcal{O}}\left(\frac{|\mathcal{S}| |\mathcal{A}| H^3}{\epsilon^2}\right)$ environment samples.
- Model-based value iteration over empirical transition tensor $\hat{P}(s' \mid s, a)$ achieves the minimax optimal sample complexity:
  $$\tilde{\mathcal{O}}\left(\frac{|\mathcal{S}| |\mathcal{A}| H^2}{\epsilon^2}\right)$$
Planning inside an internal model improves sample efficiency by a full factor of $H$.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Dyna-Q
Inputs: Planning steps N, step size alpha, discount gamma, epsilon
1. Initialize Q(s, a) and Model(s, a) for all s, a
2. Loop for each episode:
     Initialize S
     Loop for each step:
       A = epsilon_greedy(Q, S)
       Execute A, observe R, S'
       # Real update
       Q(S, A) = Q(S, A) + alpha * [ R + gamma * max_a Q(S', a) - Q(S, A) ]
       Model(S, A) = (R, S')
       S = S'
       
       # Mental Planning (hallucinations)
       Loop N times:
         S_sim = random previously observed state
         A_sim = random previously executed action at S_sim
         R_sim, S_prime_sim = Model(S_sim, A_sim)
         Q(S_sim, A_sim) = Q(S_sim, A_sim) + alpha * [ R_sim + gamma * max_a Q(S_prime_sim, a) - Q(S_sim, A_sim) ]
```

---

## 5. Failure Modes & When It Breaks
1. **Model Bias / Compounding Errors:** If the internal world model makes tiny systematic prediction errors, the planning engine exploits these simulated bugs ("reward hacking" the simulator).
2. **Computational Overhead:** MCTS search consumes massive CPU/GPU time per decision step.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **OpenAI o1 / DeepSeek-R1 "Thinking" Tokens:** Test-time compute scaling is a direct manifestation of Monte Carlo Tree Search, exploring branching reasoning paths, scoring nodes with PRMs (Process Reward Models), and backtracking.
- **Tree of Thoughts (ToT):** Uses MCTS over discrete thought steps in mathematical problem solving.

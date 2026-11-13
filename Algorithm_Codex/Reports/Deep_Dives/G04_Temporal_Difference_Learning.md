# G04: Temporal-Difference Learning (TD(0), SARSA, Q-Learning)

## 1. Intuition & The "Grand Analogy"
Imagine driving on a mountain highway at night. As you round a blind corner, you suddenly spot headlights skidding into your lane. You do not wait until the crash occurs or until you arrive safely home two hours later to conclude that your situation is perilous. The moment you see headlights, your estimate of danger spikes immediately. You adjust your beliefs in real time by comparing your current prediction with your updated prediction one second later.

This is **Temporal-Difference (TD) Learning**. It combines the core strengths of Dynamic Programming and Monte Carlo methods:
- Like Monte Carlo, it learns **model-free** directly from raw experience without requiring transition physics.
- Like Dynamic Programming, it **bootstraps**: updates are driven by subsequent estimates rather than waiting for terminal outcomes.

---

## 2. Architectural & Mathematical Breakdown

### TD(0) Prediction
For an observed transition $S_t \xrightarrow{A_t, R_{t+1}} S_{t+1}$:
$$\delta_t \triangleq R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \quad (\text{TD Error})$$
$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$$

### On-Policy Control: SARSA
Learns the action-value function $Q(s, a)$ of the current behavioral policy:
$$\delta_t^{\text{SARSA}} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

### Off-Policy Control: Q-Learning (Watkins, 1989)
Directly approximates the optimal action-value function $Q^*(s, a)$ independently of the policy being followed:
$$\delta_t^{\text{Q}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$$
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

### Expected SARSA
Replaces the maximum or sampled next-action with the expectation under current policy $\pi$:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

---

## 3. Hardcore Mathematical Derivations

### The Cliff Walking Dilemma: SARSA vs. Q-Learning
In the canonical Cliff Walking problem:
- **Q-Learning** targets $Q^*(s, a)$, assuming optimal future choices ($\max_{a'} Q$). It learns the optimal path running immediately adjacent to the cliff edge. However, because its behavior policy is $\epsilon$-greedy, exploratory slips occasionally throw it off the cliff during training, causing lower online reward.
- **SARSA** evaluates the actual trajectory including exploration ($A_{t+1} \sim \pi$). Recognizing that exploratory missteps near the edge cause catastrophic $-100$ penalties, SARSA learns a longer, safer path along the upper route.

### Asymptotic Convergence of Q-Learning
Under the Robbins-Monro conditions on step sizes:
$$\sum_{t=1}^\infty \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_{t=1}^\infty \alpha_t^2(s, a) < \infty \quad \forall (s, a)$$
and assuming all state-action pairs continue to be visited infinitely often, $Q_t(s, a)$ converges with probability 1 to $Q^*(s, a)$ as $t \to \infty$.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Tabular Q-Learning
Inputs: Learning rate alpha, discount gamma, exploration epsilon
1. Initialize Q(s, a) = 0 for all s in S, a in A; Q(terminal, .) = 0
2. Loop for each episode:
     Initialize state S
     Loop for each step of episode:
       Choose A from S using policy derived from Q (e.g. epsilon-greedy)
       Take action A, observe R, S'
       Q(S, A) = Q(S, A) + alpha * [ R + gamma * max_a Q(S', a) - Q(S, A) ]
       S = S'
     until S is terminal
```

---

## 5. Failure Modes & When It Breaks
1. **Maximization Bias:** Because Q-learning uses $\max_{a'} Q(S', a')$, random noise and estimation errors are positively skewed, leading to severe overestimation of action values (resolved by Double Q-learning).
2. **The Deadly Triad (Sutton & Barto):**
   - Function Approximation (e.g. deep neural networks)
   - Bootstrapping (TD updates)
   - Off-Policy Training
   When all three are combined simultaneously, learning can diverge uncontrollably without target networks and replay buffers.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Critic Networks in RLHF / PPO:** The value model in PPO computes TD errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ along generated token sequences.
- **Generalized Advantage Estimation (GAE):** Blends multi-step TD errors $\delta_t^{\text{TD}}$ exponentially using parameter $\lambda$, serving as the primary advantage engine in DeepSeek-Math and Llama-3-Instruct alignment.

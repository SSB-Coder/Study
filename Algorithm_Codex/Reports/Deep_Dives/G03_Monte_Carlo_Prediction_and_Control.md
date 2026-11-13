# G03: Monte Carlo Prediction and Control

## 1. Intuition & The "Grand Analogy"
Imagine learning to master a new board game where you do not have the rulebook's exact probability formulas for dice rolls. Instead, you play thousands of complete games from start to finish. At the end of every completed match, you inspect every decision you made along the way. If you won, you bump up the score of those choices; if you lost, you dock points.

This is **Monte Carlo (MC) Learning**. Unlike Dynamic Programming, it does **not** require an explicit transition model $P(s' \mid s, a)$ or reward model $R(s, a)$. It learns directly from episodes of experience by averaging observed sample returns.

---

## 2. Architectural & Mathematical Breakdown
Monte Carlo methods apply exclusively to **episodic tasks** where trajectories terminate after a finite horizon $T$:
$$S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T, S_T$$

The total discounted return following time step $t$ is:
$$G_t \triangleq \sum_{k=0}^{T - t - 1} \gamma^k R_{t + k + 1}$$

### MC Prediction (First-Visit vs. Every-Visit)
- **First-Visit MC:** Averages returns $G_t$ only for the *first* time state $s$ is visited in an episode. Produces an **unbiased estimator** of $V^\pi(s)$.
- **Every-Visit MC:** Averages returns $G_t$ for *all* occurrences of state $s$ in an episode. Slightly biased in finite samples, but has lower variance and is mean-square consistent.

### Incremental Return Update
For an observed episode with return $G_t$ from state $S_t$:
$$V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

---

## 3. Hardcore Mathematical Derivations

### Unbiasedness of First-Visit Return Estimator
Let $\tau = \min \{ t \ge 0 : S_t = s \}$ be the first hitting time of state $s$.
By the definition of the value function:
$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s] = \mathbb{E}_\pi [G_\tau \mid S_\tau = s]$$
Let $\{ G_\tau^{(i)} \}_{i=1}^N$ be the observed returns across $N$ independent episodes visiting $s$. The sample mean estimator is:
$$\hat{V}_N(s) = \frac{1}{N} \sum_{i=1}^N G_\tau^{(i)}$$
$$\mathbb{E}[\hat{V}_N(s)] = \frac{1}{N} \sum_{i=1}^N \mathbb{E}[G_\tau^{(i)}] = V^\pi(s) \quad (\text{Strictly Unbiased})$$
By the Strong Law of Large Numbers:
$$\lim_{N \to \infty} \hat{V}_N(s) \xrightarrow{\text{a.s.}} V^\pi(s), \quad \text{Var}(\hat{V}_N(s)) = \frac{\sigma^2_G}{N}$$

### Generalized Policy Iteration with $\epsilon$-Greedy Exploration
To avoid getting stuck in local deterministic choices without exploring other actions, MC Control utilizes an $\epsilon$-soft policy:
$$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$
By the Policy Improvement Theorem, any $\epsilon$-greedy policy with respect to $Q^{\pi}$ improves upon $\pi$:
$$q_\pi(s, \pi'(s)) \ge v_\pi(s) \quad \forall s \in \mathcal{S}$$

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: On-Policy First-Visit MC Control
Inputs: Discount gamma, exploration epsilon
1. Initialize Q(s, a) arbitrarily, Returns(s, a) = empty list
2. Loop forever (for each episode):
     a. Generate episode following pi: S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T
     b. G = 0
     c. Loop for t = T-1 down to 0:
          G = gamma * G + R_{t+1}
          Unless pair (S_t, A_t) appears in S_0, A_0, ..., S_{t-1}, A_{t-1}:
             Append G to Returns(S_t, A_t)
             Q(S_t, A_t) = average(Returns(S_t, A_t))
             a* = argmax_a Q(S_t, a)
             For each a in A:
                pi(a | S_t) = (1 - epsilon + epsilon / |A|) if a == a* else (epsilon / |A|)
```

---

## 5. Failure Modes & When It Breaks
1. **High Variance:** Because returns $G_t$ accumulate stochastic rewards over the entire remaining trajectory without intermediate bootstrapping, sample variance can be enormous, requiring hundreds of thousands of episodes.
2. **Episodic Only:** Completely incapable of learning in non-terminating (continuing) tasks or long-horizon problems where termination is distant or undefined.
3. **Off-Policy Weight Explosion:** When using Importance Sampling for off-policy MC, the product of importance ratios $\prod_{t} \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$ suffers from unbounded or infinite variance.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Monte Carlo Rollouts in AlphaGo / o1-style Reasoning:** Rollouts to terminal states provide ground-truth outcome rewards without intermediate value estimation errors.
- **Outcome-Supervised Reward Models (ORM):** In mathematical reasoning (e.g. GSM8K / MATH), LLM solutions receive binary $+1/-1$ verifier feedback upon reaching the final answer, exactly matching episodic Monte Carlo evaluation.

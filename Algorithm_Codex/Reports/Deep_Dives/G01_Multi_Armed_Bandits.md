# G01: Multi-Armed Bandits & Exploration-Exploitation Primitives

## 1. Intuition & The "Grand Analogy"
Imagine standing in a casino before a bank of $K$ different slot machines ("one-armed bandits"). Each machine dispenses payouts according to an unknown probability distribution. You have a finite budget of tokens. If you exclusively pull the arm that gave you the best payout on your first five tries (pure *exploitation*), you might miss a jackpot machine nearby. If you sample all machines uniformly to discover their exact distributions (pure *exploration*), you spend too many tokens pulling known duds. 

The Multi-Armed Bandit problem is the purest formalization of the fundamental dilemma in decision theory: **exploration vs. exploitation**. In modern AI, bandits govern everything from real-time recommendation systems, dynamic pricing, clinical trial allocations, to active data filtering and decoding strategies in Large Language Models.

---

## 2. Architectural & Mathematical Breakdown
A $K$-armed bandit problem consists of $K$ discrete actions $a \in \{1, \dots, K\}$.
When arm $a$ is selected at time step $t$, the agent receives reward $R_t \sim \mathcal{D}_a$ with expected value:
$$q_*(a) \triangleq \mathbb{E}[R_t \mid A_t = a]$$

The agent maintains empirical value estimates $Q_t(a) \approx q_*(a)$ and action counts $N_t(a) = \sum_{\tau=1}^{t-1} \mathbb{I}(A_\tau = a)$.

### Incremental Sample-Average Update
$$Q_{t+1}(A_t) = Q_t(A_t) + \frac{1}{N_t(A_t)} \left( R_t - Q_t(A_t) \right)$$

### Exploration Strategies
1. **$\epsilon$-Greedy:**
   $$A_t = \begin{cases} \arg\max_a Q_t(a) & \text{with probability } 1 - \epsilon \\ \text{random arm} & \text{with probability } \epsilon \end{cases}$$
2. **Upper Confidence Bound (UCB1):**
   Derived from Hoeffding's Inequality, adding an uncertainty bonus proportional to $1/\sqrt{N(a)}$:
   $$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
3. **Thompson Sampling (Posterior Sampling):**
   Maintains a Bayesian posterior over $q_*(a)$. For Bernoulli rewards $R_t \in \{0, 1\}$, conjugate Beta prior $\text{Beta}(\alpha_a, \beta_a)$:
   $$\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad A_t = \arg\max_a \theta_a$$
   $$\text{Upon observing } R_t: \quad \alpha_{A_t} \leftarrow \alpha_{A_t} + R_t, \quad \beta_{A_t} \leftarrow \beta_{A_t} + (1 - R_t)$$

---

## 3. Hardcore Mathematical Derivations

### Cumulative Regret Formulation
Let $a^* = \arg\max_a q_*(a)$ be the optimal arm, with expected value $q^* = q_*(a^*)$. The suboptimality gap for arm $a$ is $\Delta_a \triangleq q^* - q_*(a)$.
The cumulative expected regret up to horizon $T$ is:
$$R(T) = T q^* - \sum_{t=1}^T \mathbb{E}[R_t] = \sum_{a=1}^K \Delta_a \mathbb{E}[N_T(a)]$$

### Lai-Robbins Lower Bound (1985)
For any consistent bandit algorithm, the asymptotic regret is bounded below logarithmically:
$$\liminf_{T \to \infty} \frac{R(T)}{\ln T} \ge \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\text{KL}(\mathcal{D}_a \parallel \mathcal{D}_{a^*})}$$
- **$\epsilon$-Greedy:** Suffers **linear regret** $\mathcal{O}(T)$ with fixed $\epsilon$, or $\mathcal{O}(\ln T)$ with tuned decaying $\epsilon_t = c/t$.
- **UCB1:** Achieves optimal logarithmic regret $\mathcal{O}\left(\sum_a \frac{\ln T}{\Delta_a}\right)$ with zero knowledge of distribution parameters.
- **Thompson Sampling:** Asymptotically matches the Lai-Robbins lower bound and empirically exhibits superior finite-time constants.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Thompson Sampling for Bernoulli Bandits
Inputs: Number of arms K, Horizon T
1. Initialize prior parameters: alpha_a = 1, beta_a = 1 for all a in {1, ..., K}
2. For t = 1 to T:
   a. For each arm a in {1, ..., K}:
        Draw sample theta_a ~ Beta(alpha_a, beta_a)
   b. Select arm A_t = argmax_a theta_a
   c. Pull arm A_t, observe binary reward R_t in {0, 1}
   d. If R_t == 1:
        alpha_{A_t} = alpha_{A_t} + 1
      Else:
        beta_{A_t} = beta_{A_t} + 1
3. Return history of selections, cumulative reward, and cumulative regret.
```

---

## 5. Failure Modes & When It Breaks
1. **Non-Stationarity:** If the underlying reward distributions drift over time, standard sample averages and unweighted counts become sluggish or completely blind. Mitigated via sliding-window or exponentially discounted updates.
2. **Heavy-Tailed / High-Variance Rewards:** UCB1 relies on bounded Hoeffding assumptions ($R \in [0, 1]$). With heavy tails, Hoeffding bounds underestimate uncertainty, failing to explore adequately. Requires robust UCB or Median-of-Means.
3. **Delayed Feedback:** When rewards arrive hours or days after action selection (e.g. ad conversions), greedy updates overreact to early arrivals while under-sampling delayed high-value arms.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **RLHF & Prompt Exploration:** Multi-armed bandits direct automated prompt selection, instruction router gates, and active sample selection during fine-tuning.
- **Speculative Decoding:** Deciding dynamically how many candidate tokens to draft or which draft model to consult can be formulated as a contextual bandit with latency vs. acceptance reward.
- **Monte Carlo Tree Search (MCTS) for Reasoning Models:** The tree search in OpenAI o1 / DeepSeek-R1 uses PUCT (Polynomial Upper Confidence trees), a direct descendant of UCB1, to balance exploration across reasoning traces.

# G13: Offline Reinforcement Learning & Hierarchical RL (CQL & Options)

## 1. Intuition & The "Grand Analogy"
Imagine training an autonomous vehicle solely from a recorded hard drive of human driving logs, without ever being permitted to run a test drive on a real road. If your algorithm considers an action never seen in the dataset—such as swerving 90 degrees right on a highway bridge—standard Q-learning might accidentally hallucinate: *"I have no negative data for this action; it might yield +1,000,000 reward!"* The agent takes the wild unvalidated action and crashes.

**Offline RL** solves this fundamental **Out-of-Distribution (OOD) Action Extrapolation** problem. Algorithms like **Conservative Q-Learning (CQL)** add an explicit penalty term to push down values of unseen actions.
Complementarily, **Hierarchical RL (HRL)** organizes complex long-horizon behaviors into high-level "Options" (macro-actions) and low-level motor primitives.

---

## 2. Architectural & Mathematical Breakdown

### Distributional Shift in Offline RL
In offline RL, data is drawn from a fixed behavioral distribution $\mathcal{D} \sim \pi_\beta(a \mid s)$.
When updating Q-values:
$$\mathcal{T} Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$
The $\max_{a'}$ operator queries actions outside the support of $\pi_\beta$, causing severe overestimation bias that cannot be corrected through online exploration.

### Conservative Q-Learning (CQL, Kumar et al., 2020)
Augments the Bellman MSE loss with a conservatism regularizer that minimizes Q-values under arbitrary policies while maximizing Q-values on the dataset:
$$\min_Q \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a \mid s)}[Q(s, a)] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)] \right) + \frac{1}{2} \mathbb{E}_{(s, a, s') \sim \mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}} Q(s, a) \right)^2 \right]$$

### The Options Framework (Sutton, Precup, Singh, 1999)
An option $\omega \in \Omega$ is a formal generalization of an action consisting of a triple:
$$\omega = \langle \mathcal{I}_\omega, \pi_\omega(a \mid s), \beta_\omega(s) \rangle$$
- $\mathcal{I}_\omega \subseteq \mathcal{S}$: Initiation set where option can be launched.
- $\pi_\omega$: Internal policy governing actions while option is active.
- $\beta_\omega: \mathcal{S} \to [0, 1]$: Stochastic termination probability condition.

---

## 3. Hardcore Mathematical Derivations

### Semi-Markov Decision Process (SMDP) Bellman Equations
When option $\omega$ executes from state $s$, it runs for a stochastic duration $k \ge 1$ steps before terminating.
The discounted reward over the option duration is:
$$r(s, \omega) \triangleq \mathbb{E}\left[ R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{k-1} R_{t+k} \mid \mathcal{E}(s, \omega) \right]$$
The SMDP Bellman Optimality Equation over options is:
$$Q^*(s, \omega) = r(s, \omega) + \sum_{s'} \sum_{k=1}^\infty P(s', k \mid s, \omega) \gamma^k \max_{\omega' \in \Omega(s')} Q^*(s', \omega')$$

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Conservative Q-Learning (Tabular / Linear)
Inputs: Offline dataset D, conservatism weight alpha > 0, discount gamma
1. Initialize Q(s, a) = 0
2. Loop for each iteration:
     For each transition (s, a, r, s') in D:
       # Standard Bellman error
       target = r + gamma * max_a' Q(s', a')
       bellman_err = target - Q(s, a)
       
       # CQL Penalty: push down unvisited actions, push up dataset action
       # Gradient w.r.t Q(s, a): pushes Q(s, a) up
       # Gradient w.r.t Q(s, a_unseen): pushes Q(s, a_unseen) down
       Q(s, a) = Q(s, a) + lr * (bellman_err + alpha * (1.0 - softmax_prob(Q(s, a))))
       For all other a_other != a:
         Q(s, a_other) = Q(s, a_other) - lr * alpha * softmax_prob(Q(s, a_other))
```

---

## 5. Failure Modes & When It Breaks
1. **Over-Conservatism:** If $\alpha$ is tuned too high, CQL becomes terrified of any action with slight uncertainty, freezing into timid, highly suboptimal trajectories.
2. **Option Collapse:** In Hierarchical RL, options frequently degenerate into single-step primitive actions or one single option dominates all states, destroying hierarchical utility.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Offline Training of LLM Reward Models:** RLHF datasets are purely offline logs of human labelings; offline conservatism prevents reward models from hallucinating positive rewards on adversarial gibberish.
- **Hierarchical Agents & Tool Use:** ReAct and autonomous coding agents execute macro-options (e.g. `run_tests`, `refactor_module`) as high-level SMDP options.

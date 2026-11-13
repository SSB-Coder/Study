# G05: Deep Q-Networks (DQN)

## 1. Intuition & The "Grand Analogy"
Imagine trying to shoot a moving clay pigeon while standing on a roller coaster. Every time you pull the trigger, not only does your target shift, but the track beneath your feet rocks violently. In naive neural network Q-learning, the target you are regressing toward ($r + \gamma \max_{a'} Q(s', a'; \theta)$) is computed using the very same network parameters $\theta$ that you are currently gradient-descending. Updating your gun shifts your target, causing catastrophic divergence and policy loops.

Deep Q-Networks (Mnih et al., DeepMind 2015) tamed this instability by introducing two revolutionary stabilizing anchors:
1. **Experience Replay:** Breaking temporal correlations by storing transitions in a large ring buffer and sampling randomized minibatches.
2. **Target Networks:** Decoupling the moving target by keeping a frozen copy of the network parameters $\theta^-$, updated only periodically.

---

## 2. Architectural & Mathematical Breakdown
A parameterized Deep Q-Network $Q(s, a; \theta)$ takes state vector $s \in \mathbb{R}^d$ and outputs predicted values for all discrete actions $a \in \{1, \dots, |\mathcal{A}|\}$.

### The Bellman Mean Squared Error Loss
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( y^{\text{DQN}} - Q(s, a; \theta) \right)^2 \right]$$
where the target $y^{\text{DQN}}$ is formed using frozen target network weights $\theta^-$:
$$y^{\text{DQN}} = r + (1 - d) \gamma \max_{a'} Q(s', a'; \theta^-)$$

### Gradient Computation
Treating $y^{\text{DQN}}$ as a fixed scalar target during backpropagation:
$$\nabla_\theta \mathcal{L}(\theta) = -2 \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( y^{\text{DQN}} - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

---

## 3. Hardcore Mathematical Derivations

### Resolving the Deadly Triad
Deep Q-learning successfully resolved the deadly triad by neutralizing two of its three components:
1. **Correlation Neutralization:** In sequential environments, consecutive samples $(s_t, a_t, s_{t+1})$ violate the i.i.d. assumption of stochastic gradient descent. By sampling uniform mini-batches $(s_i, a_i, r_i, s'_i) \sim \text{Uniform}(\mathcal{D})$, data variance is minimized and autocorrelation is eliminated.
2. **Fixed-Point Contraction Stabilization:** Without a target network, the objective is not a true gradient of any fixed loss function (it is a semi-gradient). Freezing $\theta^-$ transforms each batch update into standard supervised regression onto a quasi-stationary target:
$$\mathcal{L}_k(\theta) = \frac{1}{B} \sum_{i=1}^B \left( y_i(\theta_k^-) - Q(s_i, a_i; \theta) \right)^2$$

### Double DQN (van Hasselt et al., 2016)
Standard DQN overestimates action values due to the max operation:
$$\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$$
Double DQN decouples action *selection* from action *evaluation*:
$$y^{\text{Double-DQN}} = r + (1 - d) \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)$$

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Deep Q-Learning with Experience Replay
Inputs: Replay buffer capacity N, batch size B, target update freq C
1. Initialize replay buffer D to capacity N
2. Initialize online network Q_theta and target network Q_theta_target = Q_theta
3. For episode = 1 to M:
     Initialize state s_1
     For t = 1 to T:
       With probability epsilon select random action a_t, else a_t = argmax_a Q(s_t, a; theta)
       Execute action a_t, observe reward r_t, next state s_{t+1}, terminal flag d_t
       Store transition (s_t, a_t, r_t, s_{t+1}, d_t) in D
       s_t = s_{t+1}
       
       Sample random minibatch of transitions (s_j, a_j, r_j, s'_j, d_j) from D
       Set y_j = r_j if d_j else r_j + gamma * max_a' Q(s'_j, a'; theta_target)
       Perform gradient descent step on (y_j - Q(s_j, a_j; theta))^2
       
       Every C steps: copy theta_target = theta
```

---

## 5. Failure Modes & When It Breaks
1. **Target Network Lag:** If target network updates occur too infrequently ($C$ too large), learning stalls because targets lag behind improved policies. If $C$ is too small, divergence occurs.
2. **Buffer Overwriting:** In sparse reward settings, early rare successful trajectories get pushed out and overwritten by thousands of boring failed transitions before the network can extract the signal.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Discrete Value Networks:** Estimating expected completion quality at reasoning branching points in chain-of-thought search trees.
- **Double DQN in Prompt Tuning:** Action selection vs evaluation decoupling prevents LLM verifier networks from collapsing onto pathological short-cut tokens.

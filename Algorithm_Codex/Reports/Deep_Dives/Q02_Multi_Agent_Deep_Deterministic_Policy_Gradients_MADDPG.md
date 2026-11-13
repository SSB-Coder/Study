# Q02: Multi-Agent Deep Deterministic Policy Gradients (MADDPG)

## 1. The Beginner's Analogy
Imagine playing a game of basketball where all 10 players are trying to learn how to play at the exact same time.
If Player 1 tries to optimize their strategy assuming everyone else stays predictable, they fail immediately—because while Player 1 is changing their moves, Player 2, 3, and 4 are also changing their moves! The environment appears completely chaotic and non-stationary.
**MADDPG (Multi-Agent Deep Deterministic Policy Gradients)** solves this by introducing a "Game Tape Review" coach:
- During actual play (Execution), each player can only look through their own eyes (local observation $o_i$) and decide their move using their personal actor network $\mu_i(o_i)$.
- During practice (Centralized Training), the coach has access to a giant stadium camera looking at the entire court (joint state $\mathbf{x}$) and records the exact move made by every single player $(a_1, \dots, a_N)$.
- The coach's centralized critic $Q_i(\mathbf{x}, a_1, \dots, a_N)$ tells Player 1: *"Given what everyone else did on that play, here is exactly how your move affected your score."*
Once training is complete, the coach steps away, and each player plays flawlessly using only their local eyes!

## 2. The Deep Architecture & Mechanics
MADDPG adapts continuous action actor-critic methods (like DDPG, G11) to multi-agent settings:
1. **The Non-Stationarity Dilemma:**
   In single-agent RL, the Markov property assumes $P(s' | s, a)$ is stationary. In multi-agent games, $P(s' | s, a_i) = \sum_{\mathbf{a}_{-i}} P(s' | s, a_i, \mathbf{a}_{-i}) \prod_{j \ne i} \pi_j(a_j | o_j)$ changes constantly as other agents learn, invalidating standard replay buffers.
2. **Centralized Critic:**
   By conditioning the critic on the joint state $\mathbf{x} = (o_1, \dots, o_N)$ and joint actions $\mathbf{a} = (a_1, \dots, a_N)$, the transition dynamics $P(s' | \mathbf{x}, \mathbf{a})$ become strictly stationary!
3. **Decentralized Actor:**
   The policy $\mu_i(o_i)$ depends strictly on agent $i$'s private observation $o_i$, enabling fully decentralized deployment without communication overhead during inference.

## 3. The Hardcore Mathematics

### Multi-Agent Deterministic Policy Gradient Theorem
Let $N$ agents have deterministic policies $\boldsymbol{\mu} = \{\mu_{\theta_1}, \dots, \mu_{\theta_N}\}$ with continuous action spaces.
The expected return for agent $i$ is $J(\theta_i) = \mathbb{E}[R_i]$.
The gradient of $J(\theta_i)$ with respect to actor parameters $\theta_i$ is:
$$\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{x}, \mathbf{a} \sim \mathcal{D}} \left[ \nabla_{\theta_i} \mu_i(a_i | o_i) \left. \nabla_{a_i} Q_i^{\boldsymbol{\mu}}(\mathbf{x}, a_1, \dots, a_N) \right|_{a_i = \mu_i(o_i)} \right]$$
*Key Insight:* The gradient $\nabla_{a_i} Q_i$ propagates through the critic directly into agent $i$'s action, while treating the other agents' actions $\mathbf{a}_{-i}$ as fixed context.

### Centralized Critic Loss
Each agent maintains its own centralized critic $Q_i$ trained by minimizing the TD error:
$$\mathcal{L}(\phi_i) = \mathbb{E}_{\mathbf{x}, \mathbf{a}, r, \mathbf{x}' \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(\mathbf{x}, a_1, \dots, a_N) - y_i \right)^2 \right]$$
where the target value $y_i$ is computed using target policies $\boldsymbol{\mu}'$:
$$y_i = r_i + \gamma \left. Q_{\phi_i'}\left(\mathbf{x}', a'_1, \dots, a'_N\right) \right|_{a'_j = \mu'_j(o'_j)}$$

### Policy Ensembling for Robustness
To prevent policies from overfitting to the quirks of specific competitors in competitive or mixed environments, MADDPG trains an ensemble of $K$ sub-policies per agent, maximizing the average expected performance across all opposing sub-policies.

## 4. The Algorithm Step-by-Step
1. Collect multi-agent experience: each agent observes local $o_i$, selects exploratory action $a_i = \mu_{\theta_i}(o_i) + \mathcal{N}(0, \sigma^2)$, executes $\mathbf{a}$, and stores $(\mathbf{x}, \mathbf{a}, \mathbf{r}, \mathbf{x}')$ in joint replay buffer $\mathcal{D}$.
2. Sample mini-batch of $B$ transitions from $\mathcal{D}$.
3. **Critic Update:**
   - Compute next joint actions $a'_j = \mu'_j(o'_j)$.
   - Compute TD targets $y_i = r_i + \gamma Q'_i(\mathbf{x}', \mathbf{a}')$.
   - Update critic parameters $\phi_i$ by minimizing MSE $(Q_i(\mathbf{x}, \mathbf{a}) - y_i)^2$.
4. **Actor Update:**
   - Evaluate policy gradient: $\nabla_{\theta_i} J \approx \frac{1}{B} \sum_{b=1}^B \nabla_{\theta_i} \mu_i(o_i) \left. \nabla_{a_i} Q_i(\mathbf{x}, a_1, \dots, \mu_i(o_i), \dots, a_N) \right.$.
   - Update actor parameters $\theta_i$ via gradient ascent.
5. Soft-update target networks: $\theta'_i \leftarrow \tau \theta_i + (1 - \tau) \theta'_i$, $\phi'_i \leftarrow \tau \phi_i + (1 - \tau) \phi'_i$.

## 5. Limitations & Failure Modes
- **Combinatorial Scaling of Joint Action Space:** As the number of agents $N$ grows beyond 20, the input dimension of the centralized critic $Q(\mathbf{x}, a_1, \dots, a_N)$ grows linearly with $N$, increasing sample complexity.
- **Credit Bleed:** In pure cooperative settings with many agents, the centralized critic can struggle to isolate the gradient contribution of an individual action $\nabla_{a_i} Q$ when total rewards are shared.

## 6. Where This Fits
Cross-reference: MADDPG generalizes DDPG (G11) from single-agent control to competitive and cooperative multi-agent dynamics, providing the foundation for robotic soccer, drone fleet navigation, and competitive game-theoretic multi-agent environments.

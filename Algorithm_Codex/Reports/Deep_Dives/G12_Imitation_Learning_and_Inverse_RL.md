# G12: Imitation Learning & Inverse Reinforcement Learning

## 1. Intuition & The "Grand Analogy"
Imagine an apprentice chef studying under a Michelin-starred master. 
- In **Behavioral Cloning (BC)**, the apprentice simply mimics every chop and stir like a photocopier. But when a pot boils over unexpectedly (an unfamiliar state never demonstrated by the master), the apprentice panics, makes a mistake, drifts into an even weirder state, and burns down the kitchen (**compounding distribution drift**).
- In **Inverse Reinforcement Learning (IRL)**, the apprentice does not merely copy knife movements; they deduce the master's underlying culinary objective: *"Ah! The master is trying to balance acidity, sweetness, and crispness!"* Once the apprentice uncovers the hidden reward function $R(s)$, they can optimize their own cooking under any unforeseen kitchen emergency.

---

## 2. Architectural & Mathematical Breakdown
Given a dataset of expert trajectories:
$$\mathcal{D}_{\text{expert}} = \{ \tau_1, \tau_2, \dots, \tau_M \}, \quad \tau_i = (s_0, a_0, s_1, a_1, \dots)$$

### Behavioral Cloning (BC)
Treated as supervised learning:
$$\theta^* = \arg\max_\theta \sum_{(s, a) \in \mathcal{D}_{\text{expert}}} \ln \pi_\theta(a \mid s)$$
- **Pathology:** Error compounds quadratically $\mathcal{O}(\epsilon T^2)$ over horizon $T$ due to distribution shift.

### Maximum Entropy Inverse RL (Ziebart et al., 2008)
Assumes the expert chooses trajectories with probability proportional to the exponential of cumulative reward:
$$P(\tau \mid \theta) = \frac{1}{Z(\theta)} \exp\left( \sum_{t} R_\theta(s_t) \right) = \frac{1}{Z(\theta)} \exp\left( \theta^\top \mu(\tau) \right)$$
where $\mu(\tau) = \sum_{t} \phi(s_t)$ is the empirical feature count vector.

### Gradient of Maximum Entropy IRL Log-Likelihood
$$\nabla_\theta \mathcal{L}(\theta) = \mu_{\text{expert}} - \mathbb{E}_{\tau \sim \pi_\theta}[\mu(\tau)]$$
The gradient is simply the difference between **expert feature expectations** and **learner feature expectations**!

---

## 3. Hardcore Mathematical Derivations

### Ross & Bagnell Bound on Compounding Error in BC
Let $\epsilon$ be the 1-step classification error rate of the cloned policy $\mathbb{P}_{s \sim d_{\pi^*}}(\pi(s) \neq \pi^*(s)) \le \epsilon$.
The expected total error over horizon $T$ is bounded by:
$$J(\pi^*) - J(\pi) \le \epsilon T^2$$
Because an error shifts the state distribution away from $d_{\pi^*}$ into unvisited territory where the cloned policy has never trained, errors accumulate quadratically.
- **DAgger (Dataset Aggregation):** Queries the expert online on states visited by the learner, reducing compounding error from $\mathcal{O}(\epsilon T^2)$ to linear $\mathcal{O}(\epsilon T)$.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Maximum Entropy IRL
Inputs: Expert demonstrations D_expert, feature mapping phi(s), learning rate alpha
1. Compute empirical expert feature expectations:
     mu_expert = 1/M sum_{i=1}^M sum_{t} phi(s_{i,t})
2. Initialize reward parameters theta
3. Loop for iteration = 1 to K:
     a. Solve MDP using reward R(s) = theta^T phi(s) to obtain current optimal policy pi_theta
     b. Compute state visitation frequencies D(s) under pi_theta
     c. Compute learner expected feature counts:
          mu_learner = sum_s D(s) * phi(s)
     d. Update reward parameters via gradient ascent:
          theta = theta + alpha * (mu_expert - mu_learner)
4. Return recovered reward function R_theta(s)
```

---

## 5. Failure Modes & When It Breaks
1. **Reward Ambiguity (Ill-Posed Problem):** Infinitely many reward functions produce the exact same optimal policy (e.g. $R(s) = 0$ makes all policies optimal).
2. **Suboptimal Demonstrations:** MaxEnt IRL assumes the demonstrator acts near-optimally; if the expert makes mistakes, the inferred reward function internalizes those pathologies.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Supervised Fine-Tuning (SFT) is Behavioral Cloning:** SFT trains LLMs to predict next tokens from human demonstrations, suffering from the exact same exposure bias / distribution shift.
- **Reward Modeling in RLHF:** Training the reward model $r_\psi(x, y)$ from preference comparisons is a modern incarnation of Maximum Entropy IRL.

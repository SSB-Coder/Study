# G07: Actor-Critic Methods & Advantage Estimation (A2C / GAE)

## 1. Intuition & The "Grand Analogy"
Imagine an actor performing a scene on stage before a stern acting director sitting in the front row. The actor (the **Policy**) decides which movements, tones, and gestures to perform in real time. The director (the **Critic**) does not perform, but has watched hundreds of plays; at every pause, the director whispers immediate critique: *"That inflection was much better than expected (+Advantage), do more of that!"* or *"That pause dragged the scene down (-Advantage), quicken the pace!"*

In pure Policy Gradients (REINFORCE), the actor waits until the entire season of performances ends to read newspaper reviews. In **Actor-Critic**, the actor receives instant feedback after every single line from a trained critic, radically reducing variance and enabling continuous online updates.

---

## 2. Architectural & Mathematical Breakdown
Actor-Critic maintains two distinct functional structures:
1. **Actor $\pi_\theta(a \mid s)$:** Parameterizes the decision-making policy.
2. **Critic $V_w(s)$:** Estimates the expected cumulative return from state $s$.

### 1-Step Advantage Actor-Critic (A2C)
Upon transitioning $S_t \xrightarrow{A_t, R_{t+1}} S_{t+1}$:
$$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t) \quad (\text{TD Error serves as unbiased Advantage estimate})$$

### Parameter Updates
- **Critic Update (MSE Loss):**
  $$w \leftarrow w + \beta \delta_t \nabla_w V_w(S_t)$$
- **Actor Update (Policy Gradient with Advantage):**
  $$\theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \ln \pi_\theta(A_t \mid S_t)$$

---

## 3. Hardcore Mathematical Derivations

### Generalized Advantage Estimation (GAE, Schulman et al., 2015)
Define the $k$-step advantage estimator $\hat{A}_t^{(k)} \triangleq \sum_{l=0}^{k-1} \gamma^l R_{t+l+1} + \gamma^k V(S_{t+k}) - V(S_t)$.
In terms of 1-step TD residuals $\delta_t^V = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$:
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$$

GAE$(\gamma, \lambda)$ is defined as the exponentially weighted average of all $k$-step advantage estimators:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
- $\lambda = 0$: Yields $\hat{A}_t = \delta_t^V$ (1-step TD). Minimal variance, but biased if $V$ is inaccurate.
- $\lambda = 1$: Yields $\hat{A}_t = \sum_{l=0}^\infty \gamma^l R_{t+l+1} - V(S_t)$ (Empirical Monte Carlo return minus baseline). Unbiased, but highest variance.
- Tuning $\lambda \in [0.90, 0.98]$ provides the optimal Pareto trade-off between bias and variance.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Online Advantage Actor-Critic (A2C)
Inputs: Step sizes alpha > 0, beta > 0, discount gamma in (0, 1]
1. Initialize actor theta and critic w arbitrarily
2. For each episode:
     Initialize state S
     For each step of episode:
       Sample action A ~ pi_theta(. | S)
       Take action A, observe R, S'
       
       # Compute TD error
       delta = R + gamma * V_w(S') - V_w(S)  (if S' is terminal, V(S') = 0)
       
       # Update Critic
       w = w + beta * delta * grad_w V_w(S)
       
       # Update Actor
       theta = theta + alpha * delta * grad_theta ln pi_theta(A | S)
       
       S = S'
     until S is terminal
```

---

## 5. Failure Modes & When It Breaks
1. **Critic Bias Poisoning:** If the critic's function approximator is poorly calibrated or biased, the TD error $\delta_t$ gives incorrect sign or magnitude to the actor, driving policy degradation.
2. **Two-Timescale Instability:** The critic must learn faster than the actor ($\beta > \alpha$) so that value evaluations track policy shifts; if the actor updates too rapidly, the critic evaluates outdated policies.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Standard RLHF PPO Architecture:** Modern alignment algorithms (InstructGPT, Claude, Gemini) use an Actor-Critic architecture where the frozen base LLM is adapted into the Actor and an identical-size transformer head serves as the token-level Critic.
- **GAE in LLM Alignment:** Advantage tensors across prompt-completion token trajectories are computed precisely using GAE$(\gamma=1.0, \lambda=0.95)$.

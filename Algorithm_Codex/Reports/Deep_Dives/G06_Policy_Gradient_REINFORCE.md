# G06: Policy Gradient Methods & The REINFORCE Algorithm

## 1. Intuition & The "Grand Analogy"
Imagine coaching a tennis player on their serve. You do not hand them a giant lookup table assigning a numerical value to every possible combination of joint angles and wind speeds. Instead, you observe the entire serve trajectory. If the ball lands as an ace, you instruct them: *"Whatever subtle combination of wrist snap and shoulder rotation you just executed, do that with higher probability next time!"* If the ball slammed into the net, you decrease the likelihood of that movement pattern.

This is the **Policy Gradient** paradigm. Rather than indirectly extracting actions by taking $\arg\max_a Q(s, a)$, policy methods directly parameterize a probability distribution $\pi_\theta(a \mid s)$ over actions and perform gradient ascent directly on expected return.

---

## 2. Architectural & Mathematical Breakdown
Let $\tau = (s_0, a_0, r_1, s_1, \dots, s_T)$ denote a complete trajectory. The total discounted return is $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}$.
The expected performance objective is:
$$J(\theta) \triangleq \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P(\tau; \theta) R(\tau) \, d\tau$$

### The Policy Gradient Theorem
Crucially, the gradient $\nabla_\theta J(\theta)$ does **not** depend on the gradient of the environment transition dynamics $\nabla_\theta P(s' \mid s, a)$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) G_t \right]$$
where $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ is the return following step $t$.

### Softmax Discrete Policy Gradients
For linear features $\phi(s)$ with action preference logits $z_a = \theta_a^\top \phi(s)$:
$$\pi_\theta(a \mid s) = \frac{e^{\theta_a^\top \phi(s)}}{\sum_{b} e^{\theta_b^\top \phi(s)}}$$
$$\nabla_{\theta_k} \ln \pi_\theta(a \mid s) = \phi(s) \left( \mathbb{I}(a = k) - \pi_\theta(k \mid s) \right)$$

---

## 3. Hardcore Mathematical Derivations

### Derivation via the Log-Derivative Trick (Likelihood Ratio)
$$P(\tau; \theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$
Taking the logarithm:
$$\ln P(\tau; \theta) = \ln \rho_0(s_0) + \sum_{t=0}^{T-1} \left( \ln \pi_\theta(a_t \mid s_t) + \ln P(s_{t+1} \mid s_t, a_t) \right)$$
Differentiating with respect to $\theta$:
$$\nabla_\theta \ln P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t)$$
Applying the score function identity $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \ln P(\tau; \theta)$:
$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau = \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau$$
$$= \int P(\tau; \theta) \nabla_\theta \ln P(\tau; \theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \ln P(\tau; \theta) R(\tau) \right]$$

### Variance Reduction via State-Dependent Baseline $b(s)$
For any baseline function $b(s)$ that does not depend on action $a$:
$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \ln \pi_\theta(a \mid s) b(s)] = b(s) \sum_{a} \pi_\theta(a \mid s) \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} = b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s) \nabla_\theta (1) = 0$$
Subtracting $b(s_t) = V(s_t)$ **strictly reduces sample variance without introducing any bias**.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: REINFORCE with Baseline
Inputs: Differentiable policy pi_theta, baseline value function V_w, learning rates alpha, beta
1. Initialize parameters theta and w
2. Loop for each episode:
     Generate an episode S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T following pi_theta
     Loop for t = 0 to T-1:
       G = sum_{k=t+1}^T gamma^{k-t-1} R_k
       delta = G - V_w(S_t)            # Advantage estimate
       w = w + beta * delta * grad_w V_w(S_t)
       theta = theta + alpha * gamma^t * delta * grad_theta ln pi_theta(A_t | S_t)
```

---

## 5. Failure Modes & When It Breaks
1. **Destructive Policy Updates:** A single bad batch with unusually high positive rewards on suboptimal actions can cause policy logits to saturate. Once probabilities collapse toward 0 or 1, gradients vanish, permanently trapping the policy.
2. **High Sample Variance:** Monte Carlo trajectory returns $G_t$ compound variance across hundreds of environment steps, requiring huge batch sizes to achieve stable direction.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Direct Foundation of RLHF:** The standard policy gradient objective is the exact mathematical core used to align LLMs against human preferences.
- **DeepSeek-R1 / GRPO:** Group Relative Policy Optimization (GRPO) directly evaluates mathematical and reasoning responses using REINFORCE-style score updates, replacing the critic network with the group mean reward across $K$ sampled completions as the baseline $b(s)$.

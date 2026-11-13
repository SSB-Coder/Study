# G14: Partially Observable MDPs (POMDP) & Belief States

## 1. Intuition & The "Grand Analogy"
Imagine navigating a dense, pitch-black submarine through underwater canyons. You cannot directly observe your exact $(x, y, z)$ coordinates. All you have are intermittent, noisy sonar pings that beep with varying delays. If a sonar ping sounds muffled, you do not know for sure if a rock wall is 5 meters away or if a school of fish swam past.

A **Partially Observable Markov Decision Process (POMDP)** formalizes decision-making when the true underlying state of the world is hidden. The agent cannot act based solely on current sensor readings $o_t$; it must maintain a probability distribution over all possible hidden states—termed the **Belief State** $b(s)$—and update this distribution using Bayes' theorem as observations arrive.

---

## 2. Architectural & Mathematical Breakdown
A POMDP is a 7-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$
- $\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma$: Standard underlying MDP components.
- $\Omega$: Discrete observation space $o \in \Omega$.
- $\mathcal{O}(o \mid s', a) = \mathbb{P}(O_{t+1} = o \mid S_{t+1} = s', A_t = a)$: Observation emission model.

### The Belief State
A belief state $b \in \Delta(\mathcal{S})$ is a probability distribution over hidden states:
$$b(s) \triangleq \mathbb{P}(S_t = s \mid h_t), \quad \text{where } h_t = (a_0, o_1, a_1, \dots, a_{t-1}, o_t)$$

### Bayesian Belief State Update
Upon executing action $a$ and receiving observation $o$:
$$b'(s') = \frac{\mathcal{O}(o \mid s', a) \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a) b(s)}{\mathbb{P}(o \mid b, a)}$$
where the normalizer (evidence) is:
$$\mathbb{P}(o \mid b, a) = \sum_{s' \in \mathcal{S}} \mathcal{O}(o \mid s', a) \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a) b(s)$$

---

## 3. Hardcore Mathematical Derivations

### Conversion to Continuous Belief MDP
A POMDP over discrete states $\mathcal{S}$ is mathematically equivalent to a fully observable MDP over the continuous simplex of belief states $\mathcal{B} = \Delta(\mathcal{S})$:
$$V^*(b) = \max_{a \in \mathcal{A}} \left[ \sum_{s \in \mathcal{S}} b(s) \mathcal{R}(s, a) + \gamma \sum_{o \in \Omega} \mathbb{P}(o \mid b, a) V^*(b') \right]$$
- **Piecewise Linear and Convex (PWLC):** Sondik (1971) proved that for any finite horizon $t$, the optimal value function $V_t^*(b)$ is piecewise linear and convex in $b$:
  $$V_t^*(b) = \max_{\alpha \in \Gamma_t} \sum_{s \in \mathcal{S}} b(s) \alpha(s)$$
  where $\Gamma_t$ is a finite set of hyperplanes ($\alpha$-vectors) representing conditional plans.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Exact Bayesian Belief State Filter
Inputs: Prior belief b_0, transition matrix T, observation matrix O
1. Initialize belief vector b = b_0
2. For each time step t:
     a. Select action a ~ pi(b)
     b. Execute a in environment, receive observation o
     c. Predict step:
          b_pred(s') = sum_s T(s' | s, a) * b(s)
     d. Update step:
          For each s' in S:
            numerator(s') = O(o | s', a) * b_pred(s')
          P(o | b, a) = sum_{s'} numerator(s')
          b'(s') = numerator(s') / P(o | b, a)
     e. b = b'
```

---

## 5. Failure Modes & When It Breaks
1. **Intractable Belief Spaces:** The dimension of the belief simplex is $|\mathcal{S}| - 1$. When $|\mathcal{S}|$ is large or continuous, exact belief maintenance requires particle filters or approximate neural recurrent hidden states (DRQN).
2. **Observation Aliasing:** When different hidden states emit identical observation distributions, the agent cannot disambiguate states without taking deliberate information-gathering actions.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Multi-Turn Conversational Memory:** An LLM does not have full state observability of the user's hidden mental state, goals, or external databases. The conversational token history acts as $h_t$, and internal hidden activations act as the implicit belief state.
- **Information-Seeking Queries in Agents:** When an AI agent invokes search tools or asks clarifying questions before answering, it is executing an information-gathering POMDP action to collapse belief entropy.

# Deep Dive: Hidden Markov Models (Forward-Backward, Viterbi, Baum-Welch)

## 1. The Beginner's Analogy: The Locked Castle and the Guard's Boots
Imagine you are a prisoner locked in a windowless dungeon beneath a castle:
- You cannot see the weather outside ($S = \{\text{Rainy}, \text{Sunny}\}$ are hidden latent states).
- Every morning, a guard walks past your cell wearing boots ($O = \{\text{Muddy}, \text{Clean}\}$ are observable emissions).
- **The Forward Algorithm (Filtering):** By looking at the boots today, what is the exact probability that it is currently raining?
- **The Backward Algorithm (Smoothing):** Looking back at the past month of boots, what was the weather on Day 12?
- **The Viterbi Algorithm (Decoding):** What was the single most likely exact sequence of sunny/rainy days that took place over the entire month?
- **Baum-Welch (Learning):** If you don't even know how likely the weather transitions are, you use EM to deduce both the climate patterns and the guard's walking habits purely from the sequence of muddy and clean boots!

---

## 2. The Deep Architecture/Algorithm Breakdown
An HMM is a generative probabilistic model for sequential data $(x_{1:T}, z_{1:T})$:
1. **First-Order Markov Property:** Future hidden state depends only on current hidden state:
   $$P(z_t \mid z_{t-1}, \dots, z_1) = P(z_t \mid z_{t-1})$$
2. **Conditional Independence of Observations:** Observation $x_t$ depends solely on current hidden state $z_t$:
   $$P(x_t \mid z_{1:T}, x_{1:t-1}) = P(x_t \mid z_t)$$
3. **Model Parameters $\lambda = (\pi, A, B)$:**
   - Initial state distribution: $\pi_i = P(z_1 = i)$.
   - State transition matrix: $A_{ij} = P(z_t = j \mid z_{t-1} = i)$.
   - Emission probability matrix: $B_{jk} = P(x_t = k \mid z_t = j)$.
4. **The Three Fundamental HMM Problems:**
   - **Evaluation (Forward-Backward):** Compute marginal likelihood $P(X \mid \lambda)$.
   - **Decoding (Viterbi):** Find optimal hidden sequence $Z^* = \arg\max_Z P(Z \mid X, \lambda)$ in log-space.
   - **Learning (Baum-Welch):** Estimate parameters $\lambda^* = \arg\max_\lambda P(X \mid \lambda)$ via Expectation-Maximization.

---

## 3. The Hardcore Mathematics

### 3.1 The Forward-Backward Algorithm
**Forward Variable $\alpha_t(i) = P(x_1, \dots, x_t, z_t = i \mid \lambda)$:**
$$\alpha_1(i) = \pi_i B_{i, x_1}$$
$$\alpha_{t}(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) A_{ij} \right] B_{j, x_t}$$
**Backward Variable $\beta_t(i) = P(x_{t+1}, \dots, x_T \mid z_t = i, \lambda)$:**
$$\beta_T(i) = 1$$
$$\beta_t(i) = \sum_{j=1}^N A_{ij} B_{j, x_{t+1}} \beta_{t+1}(j)$$

### 3.2 Viterbi Decoding in Log-Space
To prevent underflow, Viterbi runs in log-probabilities:
Let $V_t(j) = \max_{z_1,\dots,z_{t-1}} \ln P(z_1, \dots, z_t=j, x_{1:t})$.
Initialization:
$$V_1(j) = \ln \pi_j + \ln B_{j, x_1}$$
Induction:
$$V_t(j) = \ln B_{j, x_t} + \max_{i=1,\dots,N} \left[ V_{t-1}(i) + \ln A_{ij} \right]$$
Backpointer tracking:
$$\text{ptr}_t(j) = \arg\max_{i=1,\dots,N} \left[ V_{t-1}(i) + \ln A_{ij} \right]$$
Backtrack from $z_T^* = \arg\max_j V_T(j)$ down to $t=1$.

### 3.3 Baum-Welch (EM) Updates
Posterior probabilities:
$$\gamma_t(i) = P(z_t = i \mid X, \lambda) = \frac{\alpha_t(i)\beta_t(i)}{\sum_j \alpha_t(j)\beta_t(j)}$$
$$\xi_t(i, j) = P(z_t = i, z_{t+1} = j \mid X, \lambda) = \frac{\alpha_t(i) A_{ij} B_{j, x_{t+1}} \beta_{t+1}(j)}{\sum_i \sum_j \alpha_t(i) A_{ij} B_{j, x_{t+1}} \beta_{t+1}(j)}$$
M-Step parameter updates:
$$\pi_i^{\text{new}} = \gamma_1(i)$$
$$A_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$
$$B_{j, k}^{\text{new}} = \frac{\sum_{t=1}^T \gamma_t(j) \mathbb{I}(x_t = k)}{\sum_{t=1}^T \gamma_t(j)}$$

---

## 4. The Algorithm Step-by-Step
1. **Forward Pass:** Recursively compute $\alpha$ table with scaling factors to prevent underflow.
2. **Backward Pass:** Recursively compute $\beta$ table using forward scaling factors.
3. **Viterbi:** Fill dynamic programming log-matrix $V$ and backpointer matrix $\text{ptr}$; trace optimal path backwards.
4. **Baum-Welch:** Compute expectations $\gamma$ and $\xi$; update $(\pi, A, B)$ until log-likelihood converges.

---

## 5. Limitations / When It Breaks
- **Memory Horizon (Markov Assumption):** Hidden states depend only on the immediate previous step; fails on long-range dependencies across paragraphs or temporal lags.
- **Historical Fix:** Recurrent Neural Networks (RNNs/LSTMs) and State Space Models (Mamba).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/03_RNN_Recurrent.md` and `Transformer/Reports/Deep_Dives/15_Transformer_SSM_Mamba_Hybrids.md`. HMMs are the exact discrete mathematical ancestor of continuous linear State Space Models (Mamba).

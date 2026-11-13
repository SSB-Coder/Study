# Deep Dive: The Gradient Descent Optimizer Family (Momentum, NAG, AdaGrad, RMSProp, Adam)

## 1. The Beginner's Analogy: Rolling Down a Mountain in Fog
Imagine hiking down a foggy mountain valley trying to find the lowest lake at the bottom:
- **Vanilla SGD** takes a step directly down the steepest immediate slope under your boots. If you are in a steep canyon with flat floor, SGD bounces violently from canyon wall to canyon wall, making agonizingly slow forward progress.
- **Momentum** acts like a heavy bowling ball: as it rolls down the canyon, its mass builds momentum along the valley floor, smoothing out the side-to-side bounces.
- **Nesterov Accelerated Gradient (NAG)** is a bowling ball with headlights: it looks ahead to where its current momentum will carry it, and applies braking or steering *before* making the turn.
- **AdaGrad** gives each coordinate its own personalized shoes: coordinates that have experienced massive past movements get smaller steps; dormant coordinates get bigger steps.
- **RMSProp** fixes AdaGrad's fatal flaw (which causes step sizes to prematurely shrink to zero) by using an exponentially decaying moving average of past gradient squares.
- **Adam (Adaptive Moment Estimation)** combines the bowling ball's momentum with RMSProp's adaptive per-coordinate shoes, plus mathematical bias correction for early cold starts.

---

## 2. The Deep Architecture/Algorithm Breakdown
First-order optimization methods iteratively minimize a differentiable cost function $\mathcal{L}(\theta)$:
1. **Stochastic Gradient Descent (SGD):** $\theta_{t+1} = \theta_t - \eta g_t$.
2. **Polyak Classical Momentum:** Adds velocity vector $v_t = \gamma v_{t-1} + \eta g_t$, dampening high-frequency oscillations.
3. **Nesterov Accelerated Gradient (NAG):** Computes gradient at the "lookahead" position: $g_t = \nabla \mathcal{L}(\theta_t - \gamma v_{t-1})$.
4. **AdaGrad:** Scales learning rate inversely by the sum of historical squared gradients: $G_t = G_{t-1} + g_t \odot g_t$.
5. **RMSProp:** Replaces cumulative sum with exponential moving average: $v_t = \beta v_{t-1} + (1-\beta) g_t^2$.
6. **Adam:** Tracks exponentially decaying first moment $m_t$ (mean) and second moment $v_t$ (uncentered variance), with bias correction terms $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$.

---

## 3. The Hardcore Mathematics

### 3.1 Mathematical Derivation of Adam
Let $g_t = \nabla_\theta \mathcal{L}(\theta_t)$.
First moment (momentum):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
Second moment (adaptive scale):
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
Because $m_0 = 0$ and $v_0 = 0$, both vectors are biased toward zero at early time steps:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \implies \mathbb{E}[m_t] = \mathbb{E}[g] (1 - \beta_1^t)$$
Bias-corrected moments:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
Parameter update:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize Parameters:** Set $\theta_0$, $m_0 = 0, v_0 = 0, t = 0$. Typical hyperparameters: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.
2. **Iterate:** For step $t = 1, 2, \dots$:
   - Compute gradient $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$.
   - Update biased first moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$.
   - Update biased second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.
   - Compute bias-corrected $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$.
   - Update $\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$.
3. **Stop:** When $\|\theta_t - \theta_{t-1}\| < \text{tol}$ or max iterations reached.

---

## 5. Limitations / When It Breaks
- **Adam Non-Convergence in Extreme Regimes:** Reddi et al. (2018) proved Adam can fail to converge when gradient magnitudes vary wildly, motivating AMSGrad.
- **Weight Decay Interaction:** Applying $L_2$ regularization directly to loss in Adam corrupts the adaptive scaling, motivating decoupled AdamW.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/00_The_Complete_Mathematics_of_Neural_Networks.md`. AdamW is the universal default optimizer used to train virtually every modern Transformer architecture (GPT, Llama, Claude, Gemini).

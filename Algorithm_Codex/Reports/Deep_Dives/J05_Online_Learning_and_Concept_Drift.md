# J05: Online Learning and Concept Drift

## 1. The Beginner's Analogy
Imagine predicting stock market trends or spam emails. A model trained on 2010 financial data or email terminology is completely blind to modern cryptocurrency fluctuations or sophisticated 2026 phishing scams.
The data distribution changes over time—this is **Concept Drift**. Online learning models do not train once and freeze; they ingest continuous streams of data one example at a time, continuously updating their parameters while monitoring data statistics to detect when the underlying ground truth has shifted.

## 2. The Deep Architecture & Mechanics
Streaming systems combine two essential modules:
1. **Online Incremental Learner:** Processes streaming pairs $(x_t, y_t)$ with constant memory $\mathcal{O}(d)$ and constant update time $\mathcal{O}(d)$. Update methods include Online Gradient Descent, Perceptrons, and Passive-Aggressive algorithms.
2. **Concept Drift Detector (ADWIN):** An adaptive windowing algorithm that monitors prediction errors. When a statistically significant difference between two sub-windows is detected, it signals a drift event and automatically shrinks the window.

## 3. The Hardcore Mathematics

### Passive-Aggressive Updates
For a linear classifier $f(x) = w^T x + b$ and margin loss $\ell(w; x, y) = \max(0, 1 - y(w^T x + b))$, the Passive-Aggressive formulation solves:
$$w_{t+1} = \arg\min_w \frac{1}{2} \|w - w_t\|^2 + C \xi \quad \text{s.t.} \quad \ell(w; x_t, y_t) \le \xi$$
Yielding the closed-form analytic update:
$$\tau_t = \min\left( C, \frac{\ell(w_t; x_t, y_t)}{\|x_t\|^2 + 1} \right), \quad w_{t+1} = w_t + \tau_t y_t x_t$$

### ADWIN Hoeffding Cut Condition
Let sliding window $W$ be split into two sub-windows $W_0, W_1$ with sizes $n_0, n_1$ and sample means $\hat{\mu}_0, \hat{\mu}_1$.
The harmonic sample size is $m = \frac{1}{1/n_0 + 1/n_1}$.
Under the null hypothesis that both sub-windows share the same generating distribution, the Hoeffding bound guarantees:
$$P(|\hat{\mu}_0 - \hat{\mu}_1| \ge \epsilon_{\text{cut}}) \le \delta$$
Where the threshold is:
$$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln\left( \frac{4 |W|}{\delta} \right)}$$
If $|\hat{\mu}_0 - \hat{\mu}_1| > \epsilon_{\text{cut}}$, concept drift is declared, and older sub-window $W_0$ is dropped.

## 4. The Algorithm Step-by-Step
1. **Stream Step $t$:** Receive incoming sample $x_t$.
2. **Predict:** Output $\hat{y}_t = \text{sign}(w_t^T x_t + b_t)$.
3. **Observe Truth:** Reveal ground truth $y_t$ and compute error $e_t = \mathbb{I}[\hat{y}_t \ne y_t]$.
4. **Drift Monitoring:** Push $e_t$ into ADWIN. Evaluate cut condition $|\hat{\mu}_0 - \hat{\mu}_1| > \epsilon_{\text{cut}}$.
5. **Drift Handling:** If drift triggered, adjust learning rates or reset model weights.
6. **Online Update:** Update parameters using Passive-Aggressive rule $w_{t+1} = w_t + \tau_t y_t x_t$.

## 5. Limitations & Failure Modes
- **Gradual vs. Abrupt Drift:** ADWIN excels at abrupt distribution shifts; very slow, creeping drift may fall below sub-window thresholds without accumulated evidence.
- **Label Delay:** Online updates assume immediate ground truth feedback; in real-world scenarios (fraud detection), true labels may take weeks to arrive.

## 6. Where This Fits
Cross-reference: Online learning models streaming token prediction, dynamic KV-cache management, and real-time distribution drift in serving LLM APIs.

# F03: Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)

## 1. The Beginner's Analogy
Imagine testing a fleet of electric light bulbs. Some bulbs burn out at 100 hours, some at 300 hours, but at 500 hours you must stop the experiment and publish your report. Several bulbs are still shining brightly. You cannot label their lifetime as 500 hours (which underestimates longevity), nor can you discard them (which biases your sample toward fragile bulbs). This is **right-censoring**.
Survival Analysis is the statistical mathematics of right-censored time-to-event data, answering: *"What is the probability of an entity surviving past time $t$, and how do covariates accelerate or decelerate risk?"*

## 2. The Deep Architecture & Mechanics
Survival analysis decomposes event dynamics into three primary functions:
1. **Survival Function $S(t) = P(T > t)$:** The probability that an individual survives past time $t$.
2. **Hazard Function $h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}$:** The instantaneous failure rate at time $t$ given survival up to $t$.
3. **Cumulative Hazard $H(t) = \int_0^t h(u) du$**, linked via $S(t) = \exp(-H(t))$.

### Kaplan-Meier Product-Limit Estimator
The non-parametric foundation for estimating $S(t)$ without assuming any parametric distribution. At each discrete observed event time $t_i$:
- $n_i$ individuals are currently at risk (observed duration $\ge t_i$).
- $d_i$ failure events occur at $t_i$.
The step survival probability is $(1 - d_i / n_i)$, compounding multiplicatively.

### Cox Proportional Hazards Model
A semi-parametric model where covariates act multiplicatively on a baseline hazard $h_0(t)$:
$$h(t \mid X) = h_0(t) \exp(X^T \beta)$$
The baseline hazard $h_0(t)$ is left completely unspecified, while $\beta$ is estimated by maximizing Cox's **partial log-likelihood**, which conditions only on the relative ordering of event times.

## 3. The Hardcore Mathematics

### Kaplan-Meier Formulation
$$\hat{S}(t) = \prod_{i: t_i \le t} \left( 1 - \frac{d_i}{n_i} \right)$$
Greenwood's formula provides the asymptotic variance of $\hat{S}(t)$:
$$\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{i: t_i \le t} \frac{d_i}{n_i (n_i - d_i)}$$

### Cox Partial Log-Likelihood
For unique event times $t_1 < t_2 < \dots < t_K$ with risk set $R(t_i) = \{j: T_j \ge t_i\}$:
$$L(\beta) = \prod_{i: E_i = 1} \frac{\exp(X_i^T \beta)}{\sum_{j \in R(t_i)} \exp(X_j^T \beta)}$$
$$\ell(\beta) = \sum_{i: E_i = 1} \left[ X_i^T \beta - \ln \sum_{j \in R(t_i)} \exp(X_j^T \beta) \right]$$
Gradient with respect to $\beta$:
$$abla_\beta \ell(\beta) = \sum_{i: E_i = 1} \left[ X_i - \frac{\sum_{j \in R(t_i)} X_j \exp(X_j^T \beta)}{\sum_{j \in R(t_i)} \exp(X_j^T \beta)} \right]$$

## 4. The Algorithm Step-by-Step
1. **Sort Observations:** Order pairs $(T_i, E_i, X_i)$ by ascending duration $T_i$.
2. **Compute Risk Sets:** For each unique event time $t_i$ where $E_i = 1$, define $R(t_i) = \{j : T_j \ge t_i\}$.
3. **Kaplan-Meier Update:** Compute $S(t_{k}) = S(t_{k-1}) \cdot (1 - d_k / n_k)$.
4. **Cox Partial Likelihood Optimization:** Initialize $\beta = 0$. In each iteration, evaluate hazard weights $\theta_j = \exp(X_j^T \beta)$ and accumulate the score gradient $abla_\beta \ell(\beta)$.
5. **Parameter Update:** Update $\beta \leftarrow \beta + \eta abla_\beta \ell(\beta)$ until convergence.

## 5. Limitations & Failure Modes
- **Proportional Hazards Assumption:** Assumes hazard ratio $\exp((X_1 - X_2)^T \beta)$ is constant across time. If hazard curves cross, standard Cox fails (requires time-varying coefficients).
- **Informative Censoring:** Censoring must be non-informative (independent of event time). If dropout correlates with imminent failure, survival estimates are biased upward.

## 6. Where This Fits
Cross-reference: Survival analysis underpins duration modeling, RL terminal state estimation, and customer lifetime value. In LLMs, survival curves model generation length, early-exit stopping thresholds, and prompt termination dynamics.

# F04: Volatility Modeling (ARCH and GARCH)

## 1. Intuition & The "Grand Analogy"
Imagine standing in the ocean at the beach. Most of the time, gentle 1-foot waves lap against your knees. But occasionally, a violent storm rolls in, and for the next four days the waves churn at 15 feet high before the ocean gradually calms down again. The average sea level has not changed, but the **variability (volatility)** of the water has clustered into violent bursts followed by prolonged lulls.

This phenomenon is **Volatility Clustering** (Mandelbrot, 1963): *"Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."* Standard linear time series assume constant noise variance (homoskedasticity). **ARCH** (Engle, Nobel Prize 2003) and **GARCH** (Bollerslev, 1986) model the conditional variance $\sigma_t^2$ as a dynamic, evolving time series.

---

## 2. Architectural & Mathematical Breakdown
Let $\epsilon_t$ denote mean-zero financial asset returns:
$$\epsilon_t = \sigma_t z_t, \quad z_t \sim \text{i.i.d. } \mathcal{N}(0, 1)$$
where $\sigma_t^2 \triangleq \text{Var}(\epsilon_t \mid \mathcal{F}_{t-1})$ is the **conditional variance**.

### ARCH($p$) (Engle, 1982)
Conditional variance is a linear function of past squared shocks:
$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2$$

### GARCH($p, q$) (Bollerslev, 1986)
Adds autoregressive terms over past variances, creating an ARMA-like formulation for variance:
$$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$
where $\omega > 0, \alpha_i \ge 0, \beta_j \ge 0$ guarantee positive variance.

---

## 3. Hardcore Mathematical Derivations

### Unconditional (Long-Run) Variance & Stationarity
In a GARCH(1, 1) process:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
Taking the unconditional expectation $\mathbb{E}[\sigma_t^2] = \sigma^2$:
Since $\mathbb{E}[\epsilon_{t-1}^2] = \mathbb{E}[\sigma_{t-1}^2 z_{t-1}^2] = \mathbb{E}[\sigma_{t-1}^2] \cdot 1 = \sigma^2$:
$$\sigma^2 = \omega + \alpha \sigma^2 + \beta \sigma^2 = \omega + (\alpha + \beta) \sigma^2$$
Solving for $\sigma^2$:
$$\sigma^2 = \frac{\omega}{1 - (\alpha + \beta)}$$
- **Strict Stationarity Condition:** A GARCH(1, 1) model is weakly stationary if and only if:
  $$\alpha + \beta < 1$$
- The sum $\alpha + \beta$ measures **volatility persistence**. In financial equity markets, typically $\alpha \approx 0.05, \beta \approx 0.90$, with persistence $\approx 0.95$, meaning volatility shocks take weeks to decay.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: GARCH(1, 1) Maximum Likelihood Estimation
Inputs: Return series epsilon_1, ..., epsilon_T
1. Define negative log-likelihood function for parameters theta = (omega, alpha, beta):
     sigma_1^2 = var(epsilon)
     log_lik = 0
     For t = 2 to T:
       sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
       log_lik = log_lik - 0.5 * (ln(2*pi) + ln(sigma_t^2) + epsilon_t^2 / sigma_t^2)
     Return - log_lik
2. Minimize negative log-likelihood subject to omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
3. Compute rolling conditional standard deviations sigma_t
```

---

## 5. Failure Modes & When It Breaks
1. **Asymmetric Leverage Effect Blindness:** Standard GARCH treats positive shocks $+2\%$ and negative crashes $-2\%$ identically ($(-2)^2 = (+2)^2 = 4$). In reality, market drops trigger far higher panic and volatility than equal-sized rallies (resolved by EGARCH or GJR-GARCH).
2. **Fat Tails:** Financial asset returns exhibit kurtosis $\gg 3$. Gaussian innovations underestimate extreme crash probabilities; Student-$t$ innovations are preferred.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Perplexity & Generation Volatility Modeling:** Tracking the conditional variance of token entropy during long-form generation to detect hallucination bursts.
- **Dynamic Temperature Scaling:** Modulating LLM sampling temperature dynamically in response to uncertainty surges.

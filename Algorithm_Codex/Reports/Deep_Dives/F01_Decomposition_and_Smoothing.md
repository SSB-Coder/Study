# F01: Time Series Decomposition & Exponential Smoothing (Holt-Winters)

## 1. Intuition & The "Grand Analogy"
Imagine managing retail inventory for a winter apparel brand. Your coat sales follow three distinct physical currents simultaneously:
1. A multi-year upward trajectory as your customer base expands (**Trend** $T_t$).
2. A predictable annual tidal wave that peaks every November and collapses in July (**Seasonality** $S_t$).
3. Day-to-day erratic fluctuations caused by freak blizzards or unseasonable warmth (**Residual Noise** $R_t$).

**Holt-Winters Triple Exponential Smoothing** (Winters, 1960) equips forecasters with three distinct exponential dampening knobs: one tracking the current baseline level ($\alpha$), one tracking the slope of the trend ($\beta$), and one updating the seasonal pattern ($\gamma$).

---

## 2. Architectural & Mathematical Breakdown
A time series $y_t$ can be decomposed additively:
$$y_t = \text{Level}_t + \text{Trend}_t + \text{Seasonality}_t + \epsilon_t$$

### Holt-Winters Additive Updating Equations
Let $m$ be the seasonal period (e.g. $m=12$ for monthly, $m=4$ for quarterly):
- **Level Update:**
  $$\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha) (\ell_{t-1} + b_{t-1})$$
- **Trend Update:**
  $$b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$
- **Seasonal Component Update:**
  $$s_t = \gamma (y_t - \ell_t) + (1 - \gamma) s_{t-m}$$
- **$h$-Step Ahead Forecast:**
  $$\hat{y}_{t+h \mid t} = \ell_t + h b_t + s_{t + h - m(k+1)}$$
  where $k = \lfloor (h - 1) / m \rfloor$.

---

## 3. Hardcore Mathematical Derivations

### Error Correction Form (State-Space Representation)
Define the 1-step forecast error $e_t \triangleq y_t - \hat{y}_{t \mid t-1} = y_t - (\ell_{t-1} + b_{t-1} + s_{t-m})$.
Rewriting the smoothing equations in innovations state-space form:
$$\ell_t = \ell_{t-1} + b_{t-1} + \alpha e_t$$
$$b_t = b_{t-1} + \alpha \beta e_t$$
$$s_t = s_{t-m} + \gamma (1 - \alpha) e_t$$
Every internal state updates linearly proportional to the novel unexpected shock $e_t$. The parameters $(\alpha, \beta, \gamma) \in (0, 1)^3$ act as optimal Kalman-gain projection constants balancing historical memory against empirical innovation.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Holt-Winters Additive Smoothing
Inputs: Series y, period m, alpha, beta, gamma, forecast horizon h
1. Initialize seasonal indices s_0, ..., s_{m-1} by averaging seasonal differences
2. Initialize level l_0 = y_0 - s_0 and trend b_0 = (y_m - y_0) / m
3. For t = 1 to T:
     Compute l_t, b_t, s_t via updating formulas
4. For step = 1 to h:
     y_hat_{T+step} = l_T + step * b_T + s_{T + step - m}
5. Return fitted values and future forecast
```

---

## 5. Failure Modes & When It Breaks
1. **Multiplicative Amplitude Spikes:** If seasonal oscillations grow proportionally with the level (e.g. holiday sales double as the company doubles), additive smoothing under-forecasts peaks.
2. **Structural Trend Breaks:** Linear trend extrapolation $h b_t$ shoots toward infinity if the macroeconomic trend abruptly saturates.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Time-Series Foundation Models (TimeGPT, Chronos):** Pre-tokenizing continuous signals frequently utilizes classical trend-seasonal decomposition to remove non-stationarity before transformer attention.

# F02: Autoregressive & Moving Average Models (ARMA & ARIMA)

## 1. Intuition & The "Grand Analogy"
Imagine monitoring the temperature of an uninsulated cabin in the woods. 
- The temperature inside right now depends heavily on the temperature 1 hour ago because physical heat dissipates gradually (**Autoregression** $AR(p)$).
- Simultaneously, if someone leaves the front door wide open for 5 minutes (a sudden shock $\epsilon_t$), the chill lingers for the next hour before equilibrium is restored (**Moving Average** $MA(q)$).

**ARIMA (Autoregressive Integrated Moving Average)** (Box & Jenkins, 1970) combines these two physical mechanisms: autoregressive feedback loops and moving-average shock persistence, coupled with differencing ($d$) to eliminate non-stationary drifts.

---

## 2. Architectural & Mathematical Breakdown

### Stationarity & Differencing ($\Delta^d$)
A time series is weakly stationary if its mean and variance are constant over time and its autocovariance depends only on lag $k$.
If a series possesses a unit root (integrated of order $d$), we difference it:
$$\Delta y_t = y_t - y_{t-1}, \quad \Delta^2 y_t = \Delta y_t - \Delta y_{t-1}$$

### The General ARMA($p, q$) Model
$$\phi(B) y_t = \theta(B) \epsilon_t, \quad \epsilon_t \sim \text{WN}(0, \sigma^2)$$
where $B$ is the backshift operator ($B^k y_t = y_{t-k}$):
$$\left( 1 - \sum_{i=1}^p \phi_i B^i \right) y_t = c + \left( 1 + \sum_{j=1}^q \theta_j B^j \right) \epsilon_t$$
Explicitly:
$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

---

## 3. Hardcore Mathematical Derivations

### The Yule-Walker Equations for AR($p$)
Multiply both sides of $y_t = \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t$ by $y_{t-k}$ (for $k \ge 1$) and take expectations:
$$\mathbb{E}[y_t y_{t-k}] = \sum_{i=1}^p \phi_i \mathbb{E}[y_{t-i} y_{t-k}] + \mathbb{E}[\epsilon_t y_{t-k}]$$
Since $\epsilon_t$ is future white noise uncorrelated with past states ($k \ge 1$), $\mathbb{E}[\epsilon_t y_{t-k}] = 0$.
Dividing by autocovariance $\gamma(0)$ yields the autocorrelation $\rho(k)$:
$$\rho(k) = \sum_{i=1}^p \phi_i \rho(k - i) \quad \forall k \ge 1$$
In matrix form for AR($p$):
$$\begin{bmatrix} 1 & \rho(1) & \dots & \rho(p-1) \\ \rho(1) & 1 & \dots & \rho(p-2) \\ \vdots & \vdots & \ddots & \vdots \\ \rho(p-1) & \rho(p-2) & \dots & 1 \end{bmatrix} \begin{bmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{bmatrix} = \begin{bmatrix} \rho(1) \\ \rho(2) \\ \vdots \\ \rho(p) \end{bmatrix}$$
Solving this Toeplitz linear system directly inverts empirical autocorrelations into optimal AR coefficients without numerical iteration.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: AR(p) Estimation via Ordinary Least Squares
Inputs: Series y, lag order p
1. Difference series d times if non-stationary (ADF test)
2. Construct design matrix X and target vector Y:
     Y = [y_{p+1}, y_{p+2}, ..., y_T]^T
     X = [ [1, y_p, ..., y_1], [1, y_{p+1}, ..., y_2], ... ]
3. Compute regression coefficients:
     beta = (X^T X)^(-1) X^T Y
4. Extract intercept c = beta_0, AR coefficients phi = beta_{1:p}
5. Residuals: e = Y - X beta; estimate noise variance sigma^2 = mean(e^2)
6. Return model (c, phi, sigma^2)
```

---

## 5. Failure Modes & When It Breaks
1. **Unit-Root Explosions:** If any root of the characteristic polynomial $1 - \sum \phi_i z^i = 0$ lies inside the complex unit circle ($|z| \le 1$), the model is strictly non-stationary and variance explodes exponentially.
2. **Structural Invertibility Failure:** If MA roots violate $|z| > 1$, the noise shocks cannot be reconstructed from observed history.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Autoregressive Masking:** The causal attention mask in GPT models ($y_t \sim P(y_t \mid y_{t-1}, \dots, y_1)$) is the high-dimensional nonlinear neural generalization of the classical Autoregressive $AR(p)$ model.

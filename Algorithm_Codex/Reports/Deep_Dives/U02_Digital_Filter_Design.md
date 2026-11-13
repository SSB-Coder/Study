# Deep Dive: U02 — Digital Filter Design (FIR & IIR)

## 1. The Beginner's Analogy: The Sound Engineer's Equalizer
Imagine recording a singer in a studio next to a humming industrial refrigerator:
- The singer's voice is low and melodic (100–1000 Hz). The refrigerator emits a high-pitched electrical whine (10,000 Hz).
- **A Low-Pass Filter** is a mathematical gatekeeper: it lets low frequencies glide through untouched while crushing high-frequency squeals into silence.
- **FIR (Finite Impulse Response)** is a moving average with carefully shaped weights (the windowed-sinc curve); it is perfectly stable and preserves phase timing identically for all frequencies.
- **IIR (Infinite Impulse Response)** adds feedback: past outputs echo back into the filter. It requires 10x fewer mathematical multiplications than FIR to achieve a steep cutoff, but aggressive feedback can cause ringing or instability.

---

## 2. The Deep Architecture/Algorithm Breakdown
Digital filters transform discrete input sequences $x[n]$ into filtered outputs $y[n]$:
1. **FIR Filter (Non-Recursive):**
   $$y[n] = \sum_{k=0}^{M} h[k] x[n - k]$$
   - Always bounded-input bounded-output (BIBO) stable because all poles are at $z=0$.
   - Symmetric impulse responses ($h[n] = h[M - n]$) guarantee **exact linear phase** (constant group delay $\tau = M/2$).
2. **IIR Filter (Recursive):**
   $$y[n] = \frac{1}{a_0} \left( \sum_{k=0}^P b_k x[n - k] - \sum_{m=1}^Q a_m y[n - m] \right)$$
   - Uses poles inside the unit circle $|z| < 1$ in the Z-domain to create sharp rolloffs with low order.

---

## 3. The Hardcore Mathematics

### 3.1 Windowed-Sinc FIR Derivation
The ideal "brick-wall" low-pass filter with cutoff $\omega_c$ has frequency response $H(\omega) = 1$ for $|\omega| \le \omega_c$ and $0$ otherwise.
Its inverse Fourier transform is the infinite sinc function:
$$h_{\text{ideal}}[n] = \frac{1}{2\pi} \int_{-\omega_c}^{\omega_c} e^{j \omega n} d\omega = \frac{\sin(\omega_c n)}{\pi n} = 2 f_c \text{sinc}(2 f_c n)$$
Truncating this infinite series with a rectangular window produces Gibbs oscillation. Multiplying by a **Blackman window** $w[n]$ crushes side-lobe ripple below $-74$ dB:
$$w[n] = 0.42 - 0.5 \cos\left( \frac{2\pi n}{M} \right) + 0.08 \cos\left( \frac{4\pi n}{M} \right)$$

### 3.2 Direct Form II Transposed State Space
The difference equation for a 2nd-order Biquad filter:
$$\begin{aligned}
y[n] &= b_0 x[n] + s_1[n-1] \\
s_1[n] &= b_1 x[n] - a_1 y[n] + s_2[n-1] \\
s_2[n] &= b_2 x[n] - a_2 y[n]
\end{aligned}$$
Direct Form II Transposed minimizes numerical quantization noise in fixed-point DSPs.

---

## 4. The Algorithm Step-by-Step
1. **Define Specs:** Sampling rate $f_s$, cutoff $f_c$, stopband attenuation.
2. **FIR:** Compute sinc coefficients centered at $M/2$. Apply Blackman window. Normalize sum to 1. Convolve with signal.
3. **IIR:** Compute bilinear transform poles/zeros for Butterworth response. Normalize by $a_0$. Implement DF-II transposed recurrence.

---

## 5. Limitations / When It Breaks
- **IIR Instability:** Quantization of coefficient $a_m$ can push poles outside $|z| = 1$, turning the filter into an unstable oscillator.
- **FIR Latency:** High tap counts introduce delay $\tau = (M-1)/(2 f_s)$ seconds, problematic in real-time control loops (U03).

---

## 6. Where This Fits
- **Sensor Pre-Filtering:** De-noising accelerometer and gyro signals before EKF state estimation (X04, X06).
- **Time Series Smoothing (F01):** Digital filters are the hardware DSP cousins of moving averages and Holt-Winters filters.

# Deep Dive: S05 — The Lomb-Scargle Periodogram

## 1. The Beginner's Analogy: The Stroboscope on a Cloudy Night
Imagine watching a lighthouse beacon flashing on a rocky island:
- If you record a video with a constant 60 FPS camera, standard Fourier transforms (U01) instantly tell you the flashing frequency.
- But suppose you are an astronomer: daylight blinds you for 14 hours every day, clouds obscure the telescope for weeks, and you only catch 80 random glimpses across three months!
- Standard FFT demands equally spaced points and completely chokes on missing data.
- **The Lomb-Scargle Periodogram** takes your randomly scattered timestamps and fits a least-squares sine wave across the gaps. It adjusts for time shifts so that missing daytime hours do not create fake phantom frequencies.

---

## 2. The Deep Architecture/Algorithm Breakdown
Nick Lomb (1976) and Jeffrey Scargle (1982) modified classical spectral analysis for non-uniform sampling:
1. **The Uneven Sampling Problem:** Astronomical observations (radial velocity, transit photometry) have unavoidable gaps due to diurnal cycles, moon phases, and telescope scheduling.
2. **Harmonic Least-Squares Equivalence:**
   Scargle proved that Lomb's modified periodogram is mathematically identical to fitting the data to harmonic model:
   $$y(t) = A \cos(\omega (t - \tau)) + B \sin(\omega (t - \tau))$$
   via linear least squares!
3. **Statistical Foundation:**
   Under the null hypothesis of pure Gaussian white noise, the normalized power $P_{\text{LS}}(\omega)$ follows a simple **exponential probability distribution**:
   $$P(P_{\text{LS}} > z) = e^{-z}$$
   enabling exact False Alarm Probability (FAP) calculation.

---

## 3. The Hardcore Mathematics

### 3.1 Time Offset Formula
$$\tau = \frac{1}{2\omega} \text{atan2}\left( \sum_i \sin(2\omega t_i), \sum_i \cos(2\omega t_i) \right)$$
This choice of $\tau$ guarantees that the basis functions $\cos(\omega(t - \tau))$ and $\sin(\omega(t - \tau))$ are strictly **mutually orthogonal** over the discrete, uneven sample points $\{t_i\}$!

### 3.2 Normalized Power Equation
$$P_{\text{LS}}(\omega) = \frac{1}{2 \sigma^2} \left[ \frac{\left( \sum_i (y_i - \bar{y}) \cos\omega(t_i - \tau) \right)^2}{\sum_i \cos^2\omega(t_i - \tau)} + \frac{\left( \sum_i (y_i - \bar{y}) \sin\omega(t_i - \tau) \right)^2}{\sum_i \sin^2\omega(t_i - \tau)} \right]$$

---

## 4. The Algorithm Step-by-Step
1. **Pre-Process:** Compute sample mean $\bar{y}$ and sample variance $\sigma^2$.
2. **Frequency Grid:** Construct test frequency array $\omega_k \in [\omega_{\min}, \omega_{\max}]$.
3. **Evaluate:** For each $\omega$:
   - Compute phase offset $\tau$.
   - Sum weighted cosine and sine projections.
   - Combine into normalized power $P_{\text{LS}}(\omega)$.
4. **Peak Detection:** Identify $\omega^* = \arg\max P_{\text{LS}}(\omega)$ and calculate False Alarm Probability.

---

## 5. Limitations / When It Breaks
- **Aliasing Peaks:** Regular daily observation gaps create high-power alias peaks at $|f \pm 1.0 \text{ day}^{-1}|$, which can be mistaken for true astronomical signals.

---

## 6. Where This Fits
- **Cross-Reference Classical Time Series (F01/F02):** F01/F02 assumes strictly uniform $\Delta t$; Lomb-Scargle is what you reach for when sampling is inherently irregular.

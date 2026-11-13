"""
S05. The Lomb-Scargle Periodogram for Unevenly-Sampled Time Series
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LombScarglePeriodogram:
    """
    Computes spectral power for unevenly sampled astronomical time series
    (e.g. exoplanet radial velocity detection, variable star photometry).
    Implements the Lomb (1976) and Scargle (1982) modified periodogram.
    """
    
    @staticmethod
    def compute(t: np.ndarray, y: np.ndarray, freqs: np.ndarray):
        """
        t: array of uneven observation timestamps.
        y: observed flux / radial velocity measurements.
        freqs: array of test frequencies (angular omega = 2*pi*f).
        Returns normalized spectral power P_LS(omega) <- eq. 3.1.
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        y_mean = np.mean(y)
        y_var = np.var(y)
        y_cent = y - y_mean
        
        powers = np.zeros(len(freqs), dtype=float)
        
        for i, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            
            # Time offset tau guaranteeing time-shift invariance <- eq. 3.2
            sin_2wt = np.sum(np.sin(2.0 * omega * t))
            cos_2wt = np.sum(np.cos(2.0 * omega * t))
            tau = (1.0 / (2.0 * omega)) * np.arctan2(sin_2wt, cos_2wt) if omega > 0 else 0.0
            
            # Generalized cosine and sine terms
            c = np.cos(omega * (t - tau))
            s = np.sin(omega * (t - tau))
            
            sum_c_sq = np.sum(c**2)
            sum_s_sq = np.sum(s**2)
            
            term_c = (np.sum(y_cent * c)**2) / sum_c_sq if sum_c_sq > 0 else 0.0
            term_s = (np.sum(y_cent * s)**2) / sum_s_sq if sum_s_sq > 0 else 0.0
            
            # Normalized Lomb-Scargle Power
            powers[i] = (1.0 / (2.0 * y_var)) * (term_c + term_s) if y_var > 0 else 0.0
            
        return powers

if __name__ == "__main__":
    print("--- S05: Lomb-Scargle Periodogram Sanity Check ---")
    np.random.seed(42)
    # Generate unevenly-sampled astronomical time series:
    # 80 observations scattered randomly over 50 days (gaps due to daylight & bad weather)
    N_obs = 80
    t_obs = np.sort(np.random.uniform(0.0, 50.0, N_obs))
    
    # Ground truth exoplanet signal: period P = 3.5 days -> frequency f = 1 / 3.5 ~ 0.2857 Hz
    true_period = 3.5
    true_freq = 1.0 / true_period
    flux = 2.0 * np.sin(2.0 * np.pi * true_freq * t_obs) + np.random.normal(0, 0.4, N_obs)
    
    # Sweep frequency grid from 0.05 to 0.6 cycles/day
    test_freqs = np.linspace(0.05, 0.6, 500)
    powers = LombScarglePeriodogram.compute(t_obs, flux, test_freqs)
    
    # Find dominant peak
    best_freq_idx = np.argmax(powers)
    detected_freq = test_freqs[best_freq_idx]
    detected_period = 1.0 / detected_freq
    max_power = powers[best_freq_idx]
    
    print(f"True Exoplanet Period: {true_period:.4f} days (Freq: {true_freq:.4f} 1/day)")
    print(f"Detected Period: {detected_period:.4f} days (Freq: {detected_freq:.4f} 1/day)")
    print(f"Peak Normalized Power: {max_power:.2f}")
    
    # Asserts:
    # 1. Detected frequency accurately identifies true periodic transit
    assert abs(detected_freq - true_freq) < 0.01, "Lomb-Scargle failed to identify orbital frequency"
    # 2. Peak power is highly statistically significant (>> 10 under white noise)
    assert max_power > 15.0, "Spectral peak power insufficient"
    print("S05 Sanity Check: PASSED cleanly.")

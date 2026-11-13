"""
U02. Digital Filter Design (Windowed-Sinc FIR & Biquad IIR Filters)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class DigitalFilter:
    """
    Implements digital low-pass filtering:
    1. Windowed-sinc Finite Impulse Response (FIR) filter with Blackman window.
    2. 2nd-order Direct Form II Transposed Infinite Impulse Response (IIR) Biquad filter.
    """
    
    @staticmethod
    def design_fir_lowpass(cutoff_hz: float, sample_rate_hz: float, num_taps: int = 65) -> np.ndarray:
        """
        Designs a windowed-sinc FIR low-pass filter <- eq. 3.1.
        fc: normalized cutoff frequency in [0, 0.5].
        """
        if num_taps % 2 == 0:
            num_taps += 1 # Force odd length for symmetric Type I linear-phase FIR
            
        fc = cutoff_hz / sample_rate_hz
        m = (num_taps - 1) / 2.0
        n = np.arange(num_taps) - m
        
        # Sinc impulse response: h(n) = 2 * fc * sinc(2 * fc * n)
        # Handle division by zero at center n=0
        h = np.zeros(num_taps, dtype=float)
        for i in range(num_taps):
            if n[i] == 0:
                h[i] = 2.0 * fc
            else:
                h[i] = np.sin(2.0 * np.pi * fc * n[i]) / (np.pi * n[i])
                
        # Blackman window: w(n) = 0.42 - 0.5*cos(2*pi*i/M) + 0.08*cos(4*pi*i/M) <- eq. 3.2
        M = num_taps - 1
        idx = np.arange(num_taps)
        w = 0.42 - 0.5 * np.cos(2.0 * np.pi * idx / M) + 0.08 * np.cos(4.0 * np.pi * idx / M)
        
        h = h * w
        # Normalize to unit gain at DC
        h /= np.sum(h)
        return h

    @staticmethod
    def apply_fir(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Direct time-domain convolution for FIR filtering."""
        return np.convolve(signal, h, mode='same')

    @staticmethod
    def design_iir_biquad(cutoff_hz: float, sample_rate_hz: float, Q: float = 0.7071):
        """
        2nd-order Butterworth-style IIR Biquad filter coefficients (Direct Form II).
        Returns feedforward b = [b0, b1, b2] and feedback a = [a0, a1, a2].
        """
        w0 = 2.0 * np.pi * (cutoff_hz / sample_rate_hz)
        alpha = np.sin(w0) / (2.0 * Q)
        cos_w0 = np.cos(w0)
        
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        return b, a

    @staticmethod
    def apply_iir(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Applies IIR filter using Direct Form II Transposed difference equation <- eq. 3.3.
        y[n] = b0 * x[n] + s1[n-1]
        s1[n] = b1 * x[n] - a1 * y[n] + s2[n-1]
        s2[n] = b2 * x[n] - a2 * y[n]
        """
        n = len(signal)
        y = np.zeros(n, dtype=float)
        s1 = 0.0
        s2 = 0.0
        
        for i in range(n):
            x = signal[i]
            y[i] = b[0] * x + s1
            s1 = b[1] * x - a[1] * y[i] + s2
            s2 = b[2] * x - a[2] * y[i]
            
        return y

if __name__ == "__main__":
    print("--- U02: Digital Filter Design Sanity Check ---")
    sr = 1000.0 # 1 kHz sample rate
    t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
    # Low frequency signal (10 Hz) + High frequency noise (250 Hz)
    clean = np.sin(2.0 * np.pi * 10.0 * t)
    noise = 0.8 * np.sin(2.0 * np.pi * 250.0 * t)
    noisy_sig = clean + noise
    
    # 1. FIR Filter with 50 Hz cutoff
    h_fir = DigitalFilter.design_fir_lowpass(cutoff_hz=50.0, sample_rate_hz=sr, num_taps=65)
    filtered_fir = DigitalFilter.apply_fir(noisy_sig, h_fir)
    
    # 2. IIR Biquad Filter with 50 Hz cutoff
    b, a = DigitalFilter.design_iir_biquad(cutoff_hz=50.0, sample_rate_hz=sr, Q=0.7071)
    filtered_iir = DigitalFilter.apply_iir(noisy_sig, b, a)
    
    # 1. FIR Filter error against clean
    raw_error = np.var(noisy_sig - clean)
    fir_error = np.var(filtered_fir[50:-50] - clean[50:-50])
    print(f"Raw Signal Noise Variance: {raw_error:.4f}, Filtered FIR Error: {fir_error:.6f}")
    assert fir_error < 0.05 * raw_error, "FIR filter failed to suppress noise"
    
    # 2. Check IIR 250 Hz noise attenuation
    filtered_noise_iir = DigitalFilter.apply_iir(noise, b, a)
    noise_var_before = np.var(noise)
    noise_var_after = np.var(filtered_noise_iir[100:]) # Drop initial transient
    reduction = noise_var_before / noise_var_after
    print(f"IIR High-Frequency Noise Variance Reduction: {reduction:.1f}x")
    assert reduction > 100.0, "IIR filter must attenuate 250 Hz noise by at least 100x"
    print("U02 Sanity Check: PASSED cleanly.")

"""
U01. The Fast Fourier Transform (Cooley-Tukey Radix-2 DIT FFT)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class FourierTransform:
    """
    Computes 1D Discrete Fourier Transforms.
    Includes naive O(N^2) DFT and Cooley-Tukey O(N log N) recursive radix-2 FFT.
    """
    
    @staticmethod
    def dft(x: np.ndarray) -> np.ndarray:
        """Naive O(N^2) Discrete Fourier Transform <- eq. 3.1."""
        x = np.asarray(x, dtype=complex)
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        # W_N = exp(-2j * pi / N)
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    @classmethod
    def fft(cls, x: np.ndarray) -> np.ndarray:
        """
        Cooley-Tukey Radix-2 Decimation-in-Time (DIT) FFT <- eq. 3.2.
        Requires len(x) to be a power of 2.
        X_k = E_k + W_N^k * O_k
        X_{k + N/2} = E_k - W_N^k * O_k
        """
        x = np.asarray(x, dtype=complex)
        N = len(x)
        if N <= 1:
            return x
        if N % 2 != 0:
            raise ValueError(f"Length of input must be power of 2, got {N}")
            
        # Divide into even and odd indexed sub-sequences
        even = cls.fft(x[0::2])
        odd = cls.fft(x[1::2])
        
        # Twiddle factors W_N^k = exp(-2j * pi * k / N)
        k = np.arange(N // 2)
        twiddle = np.exp(-2j * np.pi * k / N)
        
        # Butterfly recombination
        first_half = even + twiddle * odd
        second_half = even - twiddle * odd
        return np.concatenate([first_half, second_half])

    @classmethod
    def ifft(cls, X: np.ndarray) -> np.ndarray:
        """Inverse FFT via conjugation identity: IFFT(X) = (1/N) * conj(FFT(conj(X)))."""
        X = np.asarray(X, dtype=complex)
        N = len(X)
        return np.conj(cls.fft(np.conj(X))) / N

if __name__ == "__main__":
    print("--- U01: Fast Fourier Transform Sanity Check ---")
    np.random.seed(42)
    N = 128
    t = np.linspace(0.0, 1.0, N, endpoint=False)
    # Signal: 5 Hz sine + 20 Hz cosine
    sig = np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.cos(2.0 * np.pi * 20.0 * t)
    
    # 1. Compare hand-rolled FFT against hand-rolled O(N^2) DFT
    X_dft = FourierTransform.dft(sig)
    X_fft = FourierTransform.fft(sig)
    diff_dft = np.linalg.norm(X_fft - X_dft)
    print(f"FFT vs DFT Discrepancy: {diff_dft:.2e}")
    assert diff_dft < 1e-10, "FFT must match DFT exactly"
    
    # 2. Test Inverse FFT reconstruction
    sig_recon = np.real(FourierTransform.ifft(X_fft))
    recon_err = np.linalg.norm(sig - sig_recon)
    print(f"IFFT Reconstruction Error: {recon_err:.2e}")
    assert recon_err < 1e-12, "IFFT reconstruction failed"
    
    # Section 3.4 Verification Ground-Truth Check
    # ground truth for verification only — not used by the algorithm above
    X_np = np.fft.fft(sig)
    np.testing.assert_allclose(X_fft, X_np, atol=1e-10)
    print("U01 Sanity Check: PASSED cleanly.")

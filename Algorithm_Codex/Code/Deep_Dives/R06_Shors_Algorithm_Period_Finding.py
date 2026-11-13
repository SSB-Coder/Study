"""
R06: Shor's Algorithm (Quantum Period Finding Component)
Pure NumPy implementation of the Quantum Fourier Transform (QFT), Quantum Phase Estimation,
order/period finding, and classical continued fraction factoring of composite integers.
"""

import numpy as np
import math
from fractions import Fraction

def quantum_fourier_transform_matrix(dim: int) -> np.ndarray:
    """
    Construct the full QFT unitary matrix of dimension M:
    QFT_{j, k} = (1 / sqrt(M)) * exp(2 * pi * i * j * k / M).
    """
    j, k = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    omega = np.exp(2.0j * np.pi / dim)
    return (1.0 / np.sqrt(dim)) * (omega ** (j * k))

def quantum_period_finding(a: int, N: int, n_counting_qubits: int = 8) -> int:
    """
    Simulate Quantum Phase Estimation to determine the period r of f(x) = a^x mod N.
    """
    M = 1 << n_counting_qubits  # Size of counting register
    
    # 1. State vector of size M x N (initialized to |0>_count (x) |1>_work)
    # Work register stores values modulo N (states 0 to N-1)
    joint_state = np.zeros((M, N), dtype=complex)
    
    # Uniform superposition in counting register: (1/sqrt(M)) sum_x |x> |a^x mod N>
    for x in range(M):
        work_val = pow(a, x, N)
        joint_state[x, work_val] = 1.0 / np.sqrt(M)
        
    # 2. Measure work register (Born rule marginal collapse)
    # For simulation clarity, pick the most probable or first non-zero branch
    work_probs = np.sum(np.abs(joint_state) ** 2, axis=0)
    measured_work = int(np.argmax(work_probs))
    
    # Post-measurement state of counting register
    counting_state = joint_state[:, measured_work].copy()
    norm = np.linalg.norm(counting_state)
    counting_state /= norm
    
    # 3. Apply inverse QFT to counting register: QFT^dagger = QFT.conj().T
    qft_mat = quantum_fourier_transform_matrix(M)
    counting_transformed = qft_mat.conj().T @ counting_state
    
    # 4. Measure counting register: peak probabilities correspond to s * M / r
    probs = np.abs(counting_transformed) ** 2
    # Find dominant peaks (ignoring 0)
    top_indices = np.argsort(probs)[::-1]
    
    # Extract candidate period via continued fractions
    for idx in top_indices:
        if idx == 0:
            continue
        phase = idx / M
        frac = Fraction(phase).limit_denominator(N)
        r_cand = frac.denominator
        # Verify if a^r mod N == 1
        if pow(a, r_cand, N) == 1:
            return r_cand
        # Also check small multiples of denominator
        for mult in [2, 3, 4]:
            if pow(a, r_cand * mult, N) == 1:
                return r_cand * mult
                
    return -1

def shors_factorization(N: int) -> tuple[int, int]:
    """
    Factor composite number N using period finding subroutine.
    """
    if N % 2 == 0:
        return 2, N // 2
        
    # Try random coprime bases 'a'
    coprimes = [a for a in range(2, N) if math.gcd(a, N) == 1]
    
    for a in coprimes:
        r = quantum_period_finding(a, N, n_counting_qubits=8)
        if r > 0 and r % 2 == 0:
            factor1 = math.gcd(pow(a, r // 2, N) - 1, N)
            factor2 = math.gcd(pow(a, r // 2, N) + 1, N)
            if 1 < factor1 < N:
                return factor1, N // factor1
            if 1 < factor2 < N:
                return factor2, N // factor2
                
    raise RuntimeError(f"Failed to factor {N}")

if __name__ == "__main__":
    print("--- R06: Shor's Algorithm Sanity Check ---")
    
    # 1. Factoring N = 15 with base a = 7
    # 7^1 = 7, 7^2 = 49 = 4, 7^3 = 28 = 13, 7^4 = 91 = 1 mod 15 -> r = 4
    period = quantum_period_finding(a=7, N=15, n_counting_qubits=8)
    print(f"Detected Period for 7^x mod 15: r = {period} (Expected: 4)")
    assert period == 4, f"Period must be 4, got {period}"
    
    # 2. Complete factorization of 15
    f1, f2 = shors_factorization(15)
    print(f"Factored 15 -> {f1} x {f2}")
    assert f1 * f2 == 15 and {f1, f2} == {3, 5}, "Factors of 15 must be 3 and 5"
    
    # 3. Factoring N = 21 with base a = 2
    # 2^6 = 64 = 1 mod 21 -> r = 6
    f3, f4 = shors_factorization(21)
    print(f"Factored 21 -> {f3} x {f4}")
    assert f3 * f4 == 21 and {f3, f4} == {3, 7}, "Factors of 21 must be 3 and 7"
    
    print("Shor's Algorithm verifications PASSED cleanly.")

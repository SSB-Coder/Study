"""
U06. Error-Correcting Codes — Hamming(7, 4) & Reed-Solomon Basics
From-scratch implementation in pure NumPy.
"""

import numpy as np

class Hamming74:
    """
    Hamming(7, 4) linear block error-correcting code.
    Encodes 4 data bits into 7 code bits by adding 3 parity bits.
    Detects and corrects any single-bit error per codeword (Hamming distance d_min = 3).
    """
    # Generator matrix G (4 x 7) in GF(2): [I_4 | P] <- eq. 3.1
    G = np.array([
        [1, 0, 0, 0,  1, 1, 0],
        [0, 1, 0, 0,  1, 0, 1],
        [0, 0, 1, 0,  0, 1, 1],
        [0, 0, 0, 1,  1, 1, 1]
    ], dtype=int)

    # Parity check matrix H (3 x 7) in GF(2): [P^T | I_3] <- eq. 3.2
    H = np.array([
        [1, 1, 0, 1,  1, 0, 0],
        [1, 0, 1, 1,  0, 1, 0],
        [0, 1, 1, 1,  0, 0, 1]
    ], dtype=int)

    @classmethod
    def encode(cls, data_bits: np.ndarray) -> np.ndarray:
        """Codeword c = d @ G (mod 2)."""
        d = np.asarray(data_bits, dtype=int)
        return (d @ cls.G) % 2

    @classmethod
    def decode(cls, received_bits: np.ndarray):
        """
        Syndrome decoding: s = H @ r^T (mod 2).
        If s == 0: No error.
        If s != 0: s matches a column of H, identifying the bit-flip error position.
        Returns corrected codeword and extracted 4 data bits.
        """
        r = np.asarray(received_bits, dtype=int).copy()
        # Compute syndrome <- eq. 3.3
        s = (cls.H @ r) % 2
        
        error_pos = -1
        if np.any(s != 0):
            # Find matching column in H
            for col in range(7):
                if np.array_equal(cls.H[:, col], s):
                    error_pos = col
                    break
            if error_pos != -1:
                # Correct single bit flip
                r[error_pos] = 1 - r[error_pos]
                
        # Extract first 4 data bits (systematic code)
        extracted_data = r[:4]
        return r, extracted_data, error_pos

class ReedSolomonToy:
    """
    Polynomial-evaluation Reed-Solomon erasure/error code over prime field GF(11).
    Encodes k message symbols into n codeword evaluations (n > k).
    Can recover from up to (n - k) erasures using Lagrange interpolation.
    """
    P = 11 # Prime field GF(11)

    @classmethod
    def encode(cls, message: list, n_evals: int = 6) -> list:
        """
        Message [m0, m1, ...] defines polynomial p(x) = m0 + m1*x + ...
        Codeword consists of evaluations [p(1), p(2), ..., p(n)] mod P.
        """
        codeword = []
        for x in range(1, n_evals + 1):
            val = sum(m * (x ** i) for i, m in enumerate(message)) % cls.P
            codeword.append(val)
        return codeword

    @classmethod
    def decode_erasures(cls, received_pairs: list, k: int) -> list:
        """
        received_pairs: list of (x_i, y_i) received without erasure. Needs at least k points.
        Reconstructs message coefficients using Lagrange interpolation in GF(P).
        """
        if len(received_pairs) < k:
            raise ValueError("Not enough points to reconstruct polynomial")
            
        pts = received_pairs[:k]
        p = cls.P
        
        # Modular inverse using Fermat's Little Theorem: a^{p-2} mod p
        def mod_inv(a):
            return pow(int(a) % p, p - 2, p)
            
        # Reconstruct polynomial coefficients p(x) = sum_i y_i * L_i(x)
        # Evaluate at x = 0 to get m0, and fit remaining via linear system
        # Direct Vandermonde solve in GF(P)
        X_mat = np.zeros((k, k), dtype=int)
        y_vec = np.zeros(k, dtype=int)
        for row, (xi, yi) in enumerate(pts):
            for col in range(k):
                X_mat[row, col] = (xi ** col) % p
            y_vec[row] = yi % p
            
        # Gaussian elimination in GF(P)
        A = X_mat.copy()
        b = y_vec.copy()
        for i in range(k):
            # Find non-zero pivot
            pivot_row = -1
            for r in range(i, k):
                if A[r, i] % p != 0:
                    pivot_row = r
                    break
            A[[i, pivot_row]] = A[[pivot_row, i]]
            b[[i, pivot_row]] = b[[pivot_row, i]]
            
            inv_pivot = mod_inv(A[i, i])
            A[i] = (A[i] * inv_pivot) % p
            b[i] = (b[i] * inv_pivot) % p
            
            for r in range(k):
                if r != i and A[r, i] != 0:
                    factor = A[r, i]
                    A[r] = (A[r] - factor * A[i]) % p
                    b[r] = (b[r] - factor * b[i]) % p
                    
        return list(b % p)

if __name__ == "__main__":
    print("--- U06: Error-Correcting Codes Sanity Check ---")
    # 1. Test Hamming(7, 4)
    data = np.array([1, 0, 1, 1])
    codeword = Hamming74.encode(data)
    print(f"Original Data: {data}")
    print(f"Hamming(7,4) Codeword: {codeword}")
    
    # Corrupt bit at index 2 (bit flip)
    noisy_codeword = codeword.copy()
    noisy_codeword[2] = 1 - noisy_codeword[2]
    print(f"Corrupted Codeword (error at bit 2): {noisy_codeword}")
    
    corrected, extracted, err_pos = Hamming74.decode(noisy_codeword)
    print(f"Detected Error Position: {err_pos}, Corrected Data: {extracted}")
    assert err_pos == 2, "Failed to identify error location"
    assert np.array_equal(extracted, data), "Failed to restore original data bits"
    
    # 2. Test Reed-Solomon Toy Code in GF(11)
    # Message: 3 symbols [5, 2, 7] -> p(x) = 5 + 2x + 7x^2
    msg = [5, 2, 7]
    rs_code = ReedSolomonToy.encode(msg, n_evals=6)
    print(f"Reed-Solomon Codeword: {rs_code}")
    
    # Simulate 3 erasures (packet loss): keep only symbols at x=1, x=3, x=5
    surviving_packets = [(1, rs_code[0]), (3, rs_code[2]), (5, rs_code[4])]
    recovered_msg = ReedSolomonToy.decode_erasures(surviving_packets, k=3)
    print(f"Recovered Message after 50% Packet Loss: {recovered_msg}")
    assert recovered_msg == msg, "Reed-Solomon erasure recovery failed"
    print("U06 Sanity Check: PASSED cleanly.")

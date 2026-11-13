"""
Z02: Zero-Knowledge Proofs & Verifiable Computation
Pure NumPy & Python implementation of:
1. Schnorr Sigma Protocol: Interactive Zero-Knowledge Proof of Knowledge of discrete log.
   - Completeness, Special Soundness (Knowledge Extractor), and Honest-Verifier Zero-Knowledge (HVZK Simulator).
   - Non-interactive Zero-Knowledge (NIZK) via Fiat-Shamir heuristic.
2. Rank-1 Constraint System (R1CS): Arithmetic circuit representation for verifiable computing
   (Aw o Bw = Cw mod p) as used in modern zk-SNARK proof systems.
"""

import hashlib
import numpy as np

class SchnorrZKP:
    """
    Schnorr Sigma Protocol for proving knowledge of secret discrete logarithm x
    such that y = g^x (mod p) without revealing any information about x.
    Operates over a prime-order subgroup of Z_p^*.
    """
    def __init__(self, p: int = 2039, q: int = 1019, g: int = 4):
        """
        Default safe prime parameters:
        p = 2039 (prime), q = 1019 (prime divisor of p - 1).
        g = 4 has order q in Z_p^* (g^q == 1 mod p).
        """
        self.p = p
        self.q = q
        self.g = g
        assert pow(self.g, self.q, self.p) == 1, "g must be a generator of order q"

    def keygen(self, secret_x: int) -> tuple[int, int]:
        """Returns (secret_key x, public_key y = g^x mod p)."""
        x = secret_x % self.q
        y = pow(self.g, x, self.p)
        return x, y

    def commit(self, r: int) -> tuple[int, int]:
        """Prover step 1: Choose ephemeral randomness r in Z_q, return (r, commitment t = g^r mod p)."""
        r_val = r % self.q
        t = pow(self.g, r_val, self.p)
        return r_val, t

    def respond(self, secret_x: int, r: int, challenge_c: int) -> int:
        """Prover step 2: Compute response s = (r + c * x) mod q."""
        return (r + challenge_c * secret_x) % self.q

    def verify(self, public_y: int, commitment_t: int, challenge_c: int, response_s: int) -> bool:
        """
        Verifier step: Check if g^s == t * y^c (mod p).
        """
        left = pow(self.g, response_s, self.p)
        right = (commitment_t * pow(public_y, challenge_c, self.p)) % self.p
        return left == right

    def fiat_shamir_challenge(self, public_y: int, commitment_t: int, message: str = "") -> int:
        """
        Computes non-interactive random oracle challenge c = Hash(g || y || t || message) mod q.
        """
        data = f"{self.g}:{public_y}:{commitment_t}:{message}".encode('utf-8')
        digest = hashlib.sha256(data).hexdigest()
        return int(digest, 16) % self.q

    def extract_knowledge(self, transcript1: tuple[int, int, int], transcript2: tuple[int, int, int]) -> int:
        """
        Special Soundness Knowledge Extractor:
        Given two valid transcripts (t, c1, s1) and (t, c2, s2) sharing the same commitment t with c1 != c2,
        extracts the secret witness x = (s1 - s2) * (c1 - c2)^(-1) mod q.
        """
        t1, c1, s1 = transcript1
        t2, c2, s2 = transcript2
        assert t1 == t2, "Transcripts must share identical commitment t"
        assert c1 != c2, "Challenges must differ for extraction"

        diff_s = (s1 - s2) % self.q
        diff_c = (c1 - c2) % self.q
        inv_c = pow(diff_c, -1, self.q)
        return (diff_s * inv_c) % self.q

    def simulate(self, public_y: int, random_c: int, random_s: int) -> tuple[int, int, int]:
        """
        Honest-Verifier Zero-Knowledge (HVZK) Simulator:
        Without knowing secret x, constructs a valid transcript (t, c, s) by selecting
        random challenge c and response s, then setting t = g^s * y^(-c) mod p.
        """
        c = random_c % self.q
        s = random_s % self.q
        inv_y_c = pow(public_y, -c, self.p)
        t = (pow(self.g, s, self.p) * inv_y_c) % self.p
        return t, c, s


class R1CSCircuit:
    """
    Rank-1 Constraint System (R1CS) Verifiable Computation Engine.
    Represents an arithmetic circuit over prime field F_p as a set of constraints:
        (A * w) o (B * w) == (C * w) (mod p)
    where w is the witness vector [1, out, x, v_1, ..., v_k]^T.
    """
    def __init__(self, p: int = 2147483647):
        """Default prime field: Mersenne prime p = 2^31 - 1."""
        self.p = p
        self.A = []
        self.B = []
        self.C = []
        self.num_vars = 0

    def set_system(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """Set constraint coefficient matrices (m_constraints, n_variables)."""
        self.A = np.array(A, dtype=np.int64) % self.p
        self.B = np.array(B, dtype=np.int64) % self.p
        self.C = np.array(C, dtype=np.int64) % self.p
        self.num_vars = self.A.shape[1]

    def verify_witness(self, witness: np.ndarray) -> bool:
        """
        Verifies if witness vector w satisfies all constraints over F_p:
        (A w mod p) * (B w mod p) == (C w mod p) (mod p).
        """
        w = np.array(witness, dtype=np.int64) % self.p
        assert len(w) == self.num_vars, f"Witness length {len(w)} != circuit variables {self.num_vars}"

        left_a = (self.A @ w) % self.p
        left_b = (self.B @ w) % self.p
        left = (left_a * left_b) % self.p
        right = (self.C @ w) % self.p

        return np.all(left == right)

    @classmethod
    def build_cubic_circuit(cls, p: int = 2147483647):
        """
        Builds R1CS circuit evaluating polynomial: y = x^3 + x + 5 (mod p).
        Variables: w = [1, y, x, v1, v2]
        Constraints:
          1) x * x = v1
          2) v1 * x = v2
          3) (v2 + x + 5) * 1 = y
        """
        circuit = cls(p=p)
        # Columns: [1, y, x, v1, v2]
        A = np.array([
            [0, 0, 1, 0, 0],  # x
            [0, 0, 0, 1, 0],  # v1
            [5, 0, 1, 0, 1]   # v2 + x + 5
        ], dtype=np.int64)

        B = np.array([
            [0, 0, 1, 0, 0],  # x
            [0, 0, 1, 0, 0],  # x
            [1, 0, 0, 0, 0]   # 1
        ], dtype=np.int64)

        C = np.array([
            [0, 0, 0, 1, 0],  # v1
            [0, 0, 0, 0, 1],  # v2
            [0, 1, 0, 0, 0]   # y (out)
        ], dtype=np.int64)

        circuit.set_system(A, B, C)
        return circuit


if __name__ == "__main__":
    print("--- Z02: Zero-Knowledge Proofs & Verifiable Computation Sanity Check ---")

    # 1. Schnorr Sigma Protocol Test
    zkp = SchnorrZKP()
    secret_x = 42
    x, y = zkp.keygen(secret_x)

    # Interactive flow
    r, t = zkp.commit(r=137)
    challenge_c = 99
    s = zkp.respond(x, r, challenge_c)
    assert zkp.verify(y, t, challenge_c, s), "Valid Schnorr proof must be accepted"

    # Tampered response rejected
    assert not zkp.verify(y, t, challenge_c, (s + 1) % zkp.q), "Tampered proof must be rejected"

    # Special Soundness: Knowledge Extraction from two challenges
    c2 = 250
    s2 = zkp.respond(x, r, c2)
    extracted_x = zkp.extract_knowledge((t, challenge_c, s), (t, c2, s2))
    print(f"Schnorr Secret Extracted via Special Soundness: {extracted_x} (Expected: {secret_x})")
    assert extracted_x == secret_x, "Knowledge extractor must perfectly recover secret witness"

    # Honest-Verifier Zero-Knowledge (HVZK) Simulator
    sim_t, sim_c, sim_s = zkp.simulate(public_y=y, random_c=777, random_s=555)
    assert zkp.verify(y, sim_t, sim_c, sim_s), "Simulated transcript must pass verification"
    print("Schnorr Protocol Properties (Completeness, Soundness, Zero-Knowledge): VERIFIED.")

    # 2. R1CS Arithmetic Circuit Test for y = x^3 + x + 5 (mod p)
    circuit = R1CSCircuit.build_cubic_circuit()
    
    # Evaluate at x = 3: y = 27 + 3 + 5 = 35
    x_val = 3
    v1_val = (x_val * x_val) % circuit.p
    v2_val = (v1_val * x_val) % circuit.p
    y_val = (v2_val + x_val + 5) % circuit.p
    assert y_val == 35

    # Valid witness: [1, y, x, v1, v2]
    valid_witness = np.array([1, y_val, x_val, v1_val, v2_val], dtype=np.int64)
    assert circuit.verify_witness(valid_witness), "Valid computation witness must satisfy R1CS constraints"

    # Fraudulent output claimed (e.g. y = 36)
    fraudulent_witness = np.array([1, 36, x_val, v1_val, v2_val], dtype=np.int64)
    assert not circuit.verify_witness(fraudulent_witness), "Fraudulent computation must fail R1CS verification"
    print("R1CS Arithmetic Circuit Verification: VERIFIED.")

    print("Z02 Sanity Check: PASSED cleanly.")

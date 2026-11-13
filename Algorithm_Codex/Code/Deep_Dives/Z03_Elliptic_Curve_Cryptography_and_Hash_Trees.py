"""
Z03: Elliptic Curve Cryptography & Cryptographic Hash Trees
Pure Python & NumPy implementation of:
1. Short Weierstrass Elliptic Curve Arithmetic over Finite Fields F_p (standard secp256k1 parameters).
   - Point addition, point doubling, and O(log k) double-and-add scalar multiplication.
2. Elliptic Curve Digital Signature Algorithm (ECDSA):
   - Key generation, signature generation with ephemeral nonce, and signature verification.
3. Cryptographic Merkle Hash Tree:
   - Full binary tree construction, SHA-256 leaf and branch hashing, logarithmic audit path generation,
     and proof-of-inclusion verification.
"""

import hashlib
from typing import Optional

# Type alias for affine point: (x, y) or None for Point at Infinity O
Point = Optional[tuple[int, int]]

class Secp256k1Curve:
    """
    Implements Weierstrass elliptic curve arithmetic over F_p:
        y^2 = x^3 + a*x + b (mod p)
    Using standard parameters for secp256k1 (Bitcoin/Ethereum curve).
    """
    # secp256k1 domain parameters
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    a = 0
    b = 7
    # Base point G
    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    G: tuple[int, int] = (Gx, Gy)
    # Subgroup order n
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    @classmethod
    def point_add(cls, P1: Point, P2: Point) -> Point:
        """Adds two elliptic curve points P1 + P2 on secp256k1."""
        if P1 is None:
            return P2
        if P2 is None:
            return P1

        x1, y1 = P1
        x2, y2 = P2

        # P + (-P) = O
        if x1 == x2 and (y1 != y2 or y1 == 0):
            return None

        # Point doubling: P1 == P2
        if x1 == x2 and y1 == y2:
            lam = (3 * x1 * x1 + cls.a) * pow(2 * y1, -1, cls.p) % cls.p
        else:
            # Point addition: P1 != P2
            lam = (y2 - y1) * pow(x2 - x1, -1, cls.p) % cls.p

        x3 = (lam * lam - x1 - x2) % cls.p
        y3 = (lam * (x1 - x3) - y1) % cls.p
        return (x3, y3)

    @classmethod
    def scalar_mult(cls, k: int, P: Point) -> Point:
        """
        Computes k * P via the double-and-add algorithm in O(log k) group additions.
        """
        if P is None or k % cls.n == 0:
            return None

        k_val = k % cls.n
        result: Point = None
        current = P

        while k_val > 0:
            if k_val & 1:
                result = cls.point_add(result, current)
            current = cls.point_add(current, current)
            k_val >>= 1

        return result


class ECDSA:
    """
    Elliptic Curve Digital Signature Algorithm (ECDSA) over secp256k1.
    """
    @classmethod
    def keygen(cls, private_key: int) -> tuple[int, tuple[int, int]]:
        """Generates (private_key d, public_key Q = d * G)."""
        d = private_key % Secp256k1Curve.n
        assert d != 0, "Private key cannot be 0"
        Q = Secp256k1Curve.scalar_mult(d, Secp256k1Curve.G)
        assert Q is not None
        return d, Q

    @classmethod
    def sign(cls, private_key: int, message_hash: bytes, nonce_k: int) -> tuple[int, int]:
        """
        Signs message hash e with private key d using deterministic or random nonce k.
        Returns signature pair (r, s).
        """
        n = Secp256k1Curve.n
        e = int.from_bytes(message_hash, byteorder='big') % n
        d = private_key % n
        k = nonce_k % n
        assert k != 0, "Nonce k cannot be 0"

        # 1. Compute curve point (x1, y1) = k * G
        R = Secp256k1Curve.scalar_mult(k, Secp256k1Curve.G)
        assert R is not None
        x1, _ = R

        # 2. r = x1 mod n
        r = x1 % n
        assert r != 0, "Signature scalar r cannot be 0"

        # 3. s = k^(-1) * (e + r * d) mod n
        k_inv = pow(k, -1, n)
        s = (k_inv * (e + r * d)) % n
        assert s != 0, "Signature scalar s cannot be 0"

        return r, s

    @classmethod
    def verify(cls, public_key: tuple[int, int], message_hash: bytes, signature: tuple[int, int]) -> bool:
        """
        Verifies ECDSA signature (r, s) against message hash and public key Q.
        """
        n = Secp256k1Curve.n
        r, s = signature
        if not (0 < r < n and 0 < s < n):
            return False

        e = int.from_bytes(message_hash, byteorder='big') % n
        w = pow(s, -1, n)
        u1 = (e * w) % n
        u2 = (r * w) % n

        # Evaluate u1 * G + u2 * Q
        pt1 = Secp256k1Curve.scalar_mult(u1, Secp256k1Curve.G)
        pt2 = Secp256k1Curve.scalar_mult(u2, public_key)
        R_prime = Secp256k1Curve.point_add(pt1, pt2)

        if R_prime is None:
            return False

        x0, _ = R_prime
        return (x0 % n) == r


class MerkleTree:
    """
    Cryptographic Binary Merkle Tree with SHA-256 hashing.
    Enables logarithmic audit proofs of data inclusion.
    """
    @staticmethod
    def sha256(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    def __init__(self, leaves: list[bytes]):
        assert len(leaves) > 0, "Must provide at least one leaf"
        self.raw_leaves = leaves
        # Pad to power of 2 by duplicating the last leaf if necessary
        self.padded_leaves = list(leaves)
        while (len(self.padded_leaves) & (len(self.padded_leaves) - 1)) != 0:
            self.padded_leaves.append(self.padded_leaves[-1])

        # Build tree levels from leaves to root
        self.levels: list[list[bytes]] = []
        current_level = [self.sha256(leaf) for leaf in self.padded_leaves]
        self.levels.append(current_level)

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = self.sha256(current_level[i] + current_level[i + 1])
                next_level.append(combined)
            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> bytes:
        """Returns the Merkle root hash."""
        return self.levels[-1][0]

    def get_proof(self, leaf_index: int) -> list[tuple[bytes, str]]:
        """
        Generates audit path for leaf at index.
        Returns list of (sibling_hash, 'left' | 'right') directions.
        """
        assert 0 <= leaf_index < len(self.raw_leaves)
        proof = []
        idx = leaf_index
        for level in self.levels[:-1]:
            is_right_child = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right_child else idx + 1
            sibling_hash = level[sibling_idx]
            position = 'left' if is_right_child else 'right'
            proof.append((sibling_hash, position))
            idx //= 2
        return proof

    @classmethod
    def verify_proof(cls, leaf: bytes, proof: list[tuple[bytes, str]], expected_root: bytes) -> bool:
        """
        Verifies that leaf belongs to the tree with expected_root given audit path proof.
        """
        current_hash = cls.sha256(leaf)
        for sibling_hash, position in proof:
            if position == 'right':
                current_hash = cls.sha256(current_hash + sibling_hash)
            else:
                current_hash = cls.sha256(sibling_hash + current_hash)
        return current_hash == expected_root


if __name__ == "__main__":
    print("--- Z03: Elliptic Curve Cryptography & Hash Trees Sanity Check ---")

    # 1. Elliptic Curve Point Arithmetic Tests
    # Identity test: n * G == None (Point at infinity)
    inf = Secp256k1Curve.scalar_mult(Secp256k1Curve.n, Secp256k1Curve.G)
    assert inf is None, "Order n times generator G must equal the point at infinity"

    # Associativity test: 2 * G + G == 3 * G
    G2 = Secp256k1Curve.scalar_mult(2, Secp256k1Curve.G)
    G3 = Secp256k1Curve.scalar_mult(3, Secp256k1Curve.G)
    assert Secp256k1Curve.point_add(G2, Secp256k1Curve.G) == G3, "Point addition must be consistent with scalar mult"
    print("secp256k1 Point Group Law & Double-and-Add Arithmetic: VERIFIED.")

    # 2. ECDSA Keygen, Signing & Verification
    private_key = 0xDEADBEEFCAFE1234567890ABCDEF
    priv_d, pub_Q = ECDSA.keygen(private_key)
    
    message = b"Algorithm Codex: From-Scratch Computational Intelligence"
    msg_hash = hashlib.sha256(message).digest()
    k_nonce = 0x9876543210FEDCBA987654321
    
    sig = ECDSA.sign(priv_d, msg_hash, nonce_k=k_nonce)
    print(f"Generated ECDSA Signature: r={hex(sig[0])[:16]}..., s={hex(sig[1])[:16]}...")
    
    # Verify true signature
    assert ECDSA.verify(pub_Q, msg_hash, sig), "Valid ECDSA signature must verify successfully"
    
    # Tampered message rejected
    tampered_msg_hash = hashlib.sha256(b"Tampered Message").digest()
    assert not ECDSA.verify(pub_Q, tampered_msg_hash, sig), "Tampered message signature must be rejected"
    
    # Tampered signature rejected
    tampered_sig = (sig[0], (sig[1] + 1) % Secp256k1Curve.n)
    assert not ECDSA.verify(pub_Q, msg_hash, tampered_sig), "Tampered signature scalar must be rejected"
    print("ECDSA Signature Verification: VERIFIED.")

    # 3. Merkle Tree Construction & Audit Path Proof
    transactions = [
        b"tx_0: Alice -> Bob 10 BTC",
        b"tx_1: Bob -> Charlie 5 BTC",
        b"tx_2: Charlie -> Dave 2 BTC",
        b"tx_3: Dave -> Eve 1 BTC",
        b"tx_4: Eve -> Frank 0.5 BTC"
    ]
    tree = MerkleTree(transactions)
    print(f"Merkle Tree Root Hash: {tree.root.hex()[:32]}...")

    # Generate and verify proof for transaction 2
    proof_tx2 = tree.get_proof(leaf_index=2)
    assert MerkleTree.verify_proof(transactions[2], proof_tx2, tree.root), "Valid Merkle proof must verify"

    # Verification fails if transaction data is modified
    fake_tx2 = b"tx_2: Charlie -> Mallory 1000 BTC"
    assert not MerkleTree.verify_proof(fake_tx2, proof_tx2, tree.root), "Fraudulent transaction must fail proof"
    print("Merkle Tree Inclusion Proof: VERIFIED.")

    print("Z03 Sanity Check: PASSED cleanly.")

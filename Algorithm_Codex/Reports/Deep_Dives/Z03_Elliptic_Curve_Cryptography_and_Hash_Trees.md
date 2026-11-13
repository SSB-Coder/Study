# Z03: Elliptic Curve Cryptography & Cryptographic Hash Trees

## 1. Mathematical Formulation & Architecture
Public-key cryptography and cryptographic authentication rely on one-way mathematical trapdoors. Elliptic curves over finite fields provide exponential security per bit compared to modular exponentiation (RSA), forming the cryptographic foundation of modern internet security (TLS 1.3), cryptocurrencies, and hardware security modules.

### Short Weierstrass Elliptic Curves over $\mathbb{F}_p$
An elliptic curve $E(\mathbb{F}_p)$ is defined by the Weierstrass equation:
$$y^2 \equiv x^3 + ax + b \pmod p$$
where the discriminant $\Delta = -16(4a^3 + 27b^2) \not\equiv 0 \pmod p$ ensures non-singularity (no cusps or self-intersections).

Together with the **Point at Infinity** $\mathcal{O}$ (acting as the group identity element), the curve points form an abelian group $(E(\mathbb{F}_p), +)$ under the chord-and-tangent group law:
1. **Identity**: $P + \mathcal{O} = \mathcal{O} + P = P$.
2. **Inverse**: For $P = (x, y)$, $-P = (x, -y \pmod p)$. Then $P + (-P) = \mathcal{O}$.
3. **Point Addition ($P \ne Q$)**:
   $$\lambda = \frac{y_2 - y_1}{x_2 - x_1} \pmod p$$
   $$x_3 = \lambda^2 - x_1 - x_2 \pmod p, \quad y_3 = \lambda(x_1 - x_3) - y_1 \pmod p$$
4. **Point Doubling ($P = Q$)**:
   $$\lambda = \frac{3x_1^2 + a}{2y_1} \pmod p$$
   $$x_3 = \lambda^2 - 2x_1 \pmod p, \quad y_3 = \lambda(x_1 - x_3) - y_1 \pmod p$$

**Scalar Multiplication**: Computing $k \cdot P = \underbrace{P + P + \dots + P}_{k \text{ times}}$ via the Double-and-Add binary ladder requires only $\mathcal{O}(\log k)$ group operations. Conversely, finding $k$ given $P$ and $k \cdot P$ is the **Elliptic Curve Discrete Logarithm Problem (ECDLP)**, conjectured to require $\mathcal{O}(\sqrt{n})$ operations via Pollard's rho algorithm.

---

### ECDSA (Elliptic Curve Digital Signature Algorithm)
Let $G$ be the base generator point of prime order $n$.
- **Keygen**: Private key $d \in_R [1, n-1]$. Public key $Q = d \cdot G$.
- **Signing**: For message hash $e = H(m)$:
  1. Sample uniform nonce $k \in_R [1, n-1]$.
  2. Compute curve point $(x_1, y_1) = k \cdot G$.
  3. Compute $r = x_1 \pmod n$.
  4. Compute $s = k^{-1}(e + r \cdot d) \pmod n$. Signature is $(r, s)$.
- **Verification**:
  1. Compute $w = s^{-1} \pmod n$.
  2. Compute $u_1 = e \cdot w \pmod n$ and $u_2 = r \cdot w \pmod n$.
  3. Recover point $(x_0, y_0) = u_1 \cdot G + u_2 \cdot Q$. Valid iff $x_0 \equiv r \pmod n$.

**Proof of Correctness**:
$$u_1 G + u_2 Q = (e w)G + (r w)(d G) = w(e + rd)G = s^{-1}(e + rd)G = k G = (x_1, y_1)$$

---

### Cryptographic Merkle Hash Trees
A Merkle tree organizes $N$ data blocks into a balanced binary tree where every parent node is the cryptographic hash of its two children:
$$\text{Node}_{\text{parent}} = H(\text{Node}_{\text{left}} \parallel \text{Node}_{\text{right}})$$
- The **Merkle Root** provides a 32-byte cryptographic fingerprint of the entire dataset.
- Membership of any block can be proven with a logarithmic **audit path** of $\lceil \log_2 N \rceil$ sibling hashes.

## 2. Implementation Details
The module `Z03_Elliptic_Curve_Cryptography_and_Hash_Trees.py`:
- Implements `Secp256k1Curve` with exact standard parameters ($p = 2^{256} - 2^{32} - 977, a=0, b=7$), affine point addition, doubling, and double-and-add scalar multiplication.
- Implements `ECDSA` digital signature generation and verification with deterministic nonce injection and boundary checks.
- Implements `MerkleTree` with dynamic leaf padding, level-by-level SHA-256 tree synthesis, audit path generation, and proof verification.

## 3. Asymptotic & Numerical Analysis
- **Point Addition / Doubling**: $\mathcal{O}(1)$ field operations (one modular inverse $\mathcal{O}(\log p)$ via Fermat's Little Theorem).
- **Scalar Multiplication**: $\mathcal{O}(\log n)$ point additions (256 operations for 256-bit security).
- **Merkle Proof Generation & Verification**: $\mathcal{O}(\log N)$ hashes, requiring $\approx 10$ hashes for $1,000$ transactions.

## 4. Edge Cases & Numerical Stability
- **Ephemeral Nonce Reuse**: If the same nonce $k$ is used to sign two distinct messages $e_1, e_2$, the private key is instantly compromised:
  $$s_1 - s_2 = k^{-1}(e_1 - e_2) \implies k = (e_1 - e_2)(s_1 - s_2)^{-1} \pmod n \implies d = r^{-1}(s_1 k - e_1) \pmod n$$
- Handled with rigorous nonce uniqueness assertions.

## 5. Standalone Verification
The test suite validates:
- Group identity ($n \cdot G = \mathcal{O}$) and group operation associativity.
- Successful generation and verification of valid ECDSA signatures.
- Immediate rejection of signatures with tampered message digests or modified scalars.
- Full Merkle root calculation, generation of valid inclusion proofs, and rejection of altered transaction leaves.

## 6. Real-World Applications & Context
Forms the security backbone of SSH, TLS certificates, Bitcoin/Ethereum transaction signing, Git commit trees, file integrity validation in BitTorrent, and zero-knowledge rollup state commitment trees.

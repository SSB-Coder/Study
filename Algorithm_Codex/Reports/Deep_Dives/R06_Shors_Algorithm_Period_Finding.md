# R06: Shor's Algorithm (Period Finding Component)

## 1. Mathematical Formulation & Architecture
Shor's Algorithm (1994) achieves superpolynomial speedup, factoring an $n$-bit integer $N$ in $\mathcal{O}(n^3)$ operations by reducing integer factorization to modular order finding.

### Reduction of Factoring to Order Finding
For a randomly chosen $a \in \{2, \dots, N-1\}$ with $\gcd(a, N) = 1$, the order $r$ is the smallest positive integer such that:
$$a^r \equiv 1 \pmod N$$
If $r$ is even, then:
$$(a^{r/2} - 1)(a^{r/2} + 1) = a^r - 1 \equiv 0 \pmod N$$
Unless $a^{r/2} \equiv -1 \pmod N$, the greatest common divisors:
$$d_1 = \gcd(a^{r/2} - 1, N), \quad d_2 = \gcd(a^{r/2} + 1, N)$$
yield non-trivial factors of $N$ with probability $\ge 1/2$.

### Quantum Phase Estimation & QFT
Order finding is mapped to phase estimation of the unitary operator $U_a \lvert y\rangle = \lvert a y \bmod N\rangle$:
1. Prepare counting register in uniform superposition and work register in $\lvert 1\rangle$:
   $$\lvert\psi\rangle = \frac{1}{\sqrt{M}}\sum_{x=0}^{M-1} \lvert x\rangle \lvert a^x \bmod N\rangle$$
2. Measure the work register. The counting register collapses into a periodic train of peaks spaced by $r$.
3. Apply the inverse Quantum Fourier Transform $\text{QFT}_M^\dagger$:
   $$\text{QFT}_M \lvert j\rangle = \frac{1}{\sqrt{M}}\sum_{k=0}^{M-1} e^{2\pi i j k / M} \lvert k\rangle$$
4. Measuring the counting register yields an integer $y$ near $s M / r$ ($s \in \{0, \dots, r-1\}$).
5. The continued fraction algorithm extracts the exact irreducible fraction $s/r$, revealing period $r$.

## 2. Implementation Details
The Python module `R06_Shors_Algorithm_Period_Finding.py`:
- Constructs the full $M \times M$ unitary QFT matrix explicitly.
- Simulates the joint counting-work register evolution over $\mathbb{C}^{M \times N}$.
- Performs projective measurement and state collapse.
- Applies inverse QFT and extracts periods via Python standard library `fractions.Fraction.limit_denominator(N)`.
- Validates factors via Euclidean GCD.

## 3. Asymptotic & Numerical Analysis
- **Quantum Complexity**: $\mathcal{O}(\log^2 N \log \log N \log \log \log N)$ using fast quantum arithmetic.
- **Classical Comparison**: General Number Field Sieve (GNFS) requires subexponential time $\mathcal{O}(\exp(c \sqrt[3]{n \log^2 n}))$.

## 4. Edge Cases & Numerical Stability
- **Register Size**: Requires $M = 2^m \ge N^2$ to guarantee unambiguous fraction extraction via continued fractions.
- **Trivial Roots**: If $a^{r/2} \equiv -1 \pmod N$ or $r$ is odd, the algorithm restarts with a different coprime base $a$.

## 5. Standalone Verification
The test suite validates:
- Order finding for $7^x \bmod 15$, returning exact period $r=4$.
- Complete factorization of $15 = 3 \times 5$.
- Complete factorization of $21 = 3 \times 7$.

## 6. Real-World Applications & Context
Theoretical foundation of quantum cryptanalysis, motivating the worldwide transition to Post-Quantum Cryptography (PQC) standards (lattice-based Kyber/Dilithium).

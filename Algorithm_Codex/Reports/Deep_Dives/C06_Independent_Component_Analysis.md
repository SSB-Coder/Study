# Deep Dive: Independent Component Analysis (FastICA & Negentropy)

## 1. The Beginner's Analogy: The Cocktail Party Problem
Imagine standing in the center of a crowded cocktail party:
- Two people are speaking at the exact same time: Alice is whispering poetry, and Bob is shouting sports scores.
- You place two microphones in the room at different distances. Microphone 1 records a loud Bob and a quiet Alice; Microphone 2 records a loud Alice and a quiet Bob. Both audio tracks are noisy, jumbled mixtures.
- PCA would only find the direction of maximum volume (the shouting sports fan).
- **Independent Component Analysis (ICA)** assumes that Alice and Bob's vocal cords are biologically independent and non-Gaussian. By the Central Limit Theorem, mixing two independent non-Gaussian signals produces a sum that looks *more Gaussian* than either original speaker.
- FastICA rotates an unmixing matrix to make the separated signals **as non-Gaussian as possible**, magically isolating Alice's pure poetry on Channel 1 and Bob's pure sports scores on Channel 2!

---

## 2. The Deep Architecture/Algorithm Breakdown
ICA solves the blind source separation problem $x = A s$:
1. **Generative Mixing Model:** Observed signal vector $x \in \mathbb{R}^d$ is a linear combination of statistically independent latent sources $s \in \mathbb{R}^d$ via unknown mixing matrix $A$: $x = A s$.
2. **Unmixing Matrix $W$:** Goal is to discover $W \approx A^{-1}$ such that recovered signals $y = W x$ recover original independent sources $s$ up to permutation and scaling.
3. **The Non-Gaussianity Principle:** Central Limit Theorem dictates that the sum of independent non-Gaussian random variables is more Gaussian than the components. Therefore, maximizing non-Gaussianity recovers the independent components.
4. **Prewhitening:** Centers data and applies PCA projection to ensure covariance is identity $\mathbb{E}[\tilde{x} \tilde{x}^T] = I$.
5. **FastICA Fixed-Point Algorithm:** Uses Newton's method to maximize negentropy using smooth contrast functions (e.g. $g(u) = \tanh(u)$).

---

## 3. The Hardcore Mathematics

### 3.1 Prewhitening (Sphering)
Let centered data be $X \in \mathbb{R}^{m \times d}$. Compute eigendecomposition of covariance:
$$\Sigma = V \Lambda V^T \implies \text{Whitening Matrix } Q = \Lambda^{-1/2} V^T$$
$$\tilde{X} = X Q^T \implies \text{Cov}(\tilde{X}) = I$$
Prewhitening reduces the search space for $W$ from arbitrary matrices to orthonormal rotation matrices ($W W^T = I$).

### 3.2 Negentropy and Hyvärinen's Contrast Function
Negentropy measures distance from Gaussianity:
$$J(y) = H(y_{\text{gauss}}) - H(y) \approx \left[ \mathbb{E}[G(y)] - \mathbb{E}[G(\nu)] \right]^2$$
Where $\nu \sim \mathcal{N}(0, 1)$ and $G(u) = \frac{1}{a} \ln \cosh(a u)$ with derivative $g(u) = \tanh(a u)$ and $g'(u) = a(1 - \tanh^2(a u))$.

### 3.3 FastICA Fixed-Point Update Rule
For a single unit weight vector $w \in \mathbb{R}^d$:
$$w^+ = \mathbb{E}\left[ \tilde{x} g(w^T \tilde{x}) \right] - \mathbb{E}\left[ g'(w^T \tilde{x}) \right] w$$
Normalize to unit length:
$$w \leftarrow \frac{w^+}{\|w^+\|_2}$$
For multiple components, orthogonalize via Gram-Schmidt or symmetric decorrelation:
$$W \leftarrow (W W^T)^{-1/2} W$$

---

## 4. The Algorithm Step-by-Step
1. **Center Data:** $X_c = X - \mathbb{E}[X]$.
2. **Whiten:** Compute SVD or eigendecomposition to produce whitened data $\tilde{X}$.
3. **Initialize Unmixing Matrix:** Generate random orthonormal matrix $W \in \mathbb{R}^{k \times d}$.
4. **Fixed-Point Iteration:** For each iteration:
   - Compute projections $u = \tilde{X} W^T$.
   - Evaluate $g(u) = \tanh(u)$ and $g'(u) = 1 - \tanh^2(u)$.
   - Update $W^+ = \frac{1}{m} g(u)^T \tilde{X} - \text{diag}(\frac{1}{m}\sum g'(u)) W$.
   - Symmetric decorrelation: $W \leftarrow (W W^T)^{-1/2} W^+$.
5. **Convergence:** Stop when $\max_i |\langle w_i^{(t+1)}, w_i^{(t)} \rangle| \approx 1.0$.
6. **Reconstruct:** Recovered sources $S = \tilde{X} W^T$.

---

## 5. Limitations / When It Breaks
- **Gaussian Mixtures Unidentifiable:** If two or more source signals are Gaussian, their joint distribution is rotationally symmetric, making it mathematically impossible to determine the unique unmixing angle.
- **Ambiguities:** Source signs and source permutations cannot be recovered (e.g. recovered signals might be flipped in sign or permuted in order).
- **Historical Fix:** Non-linear ICA and Independent Subspace Analysis (ISA).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/33_Mechanistic_Interpretability.md`. In modern LLM interpretability, Sparse Autoencoders (SAEs) solve the polysemantic superposition problem by disentangling monosemantic feature directions, directly mirroring ICA's objective of recovering independent latent components from linear mixtures.

# Y03: Gromov-Wasserstein & Sliced Optimal Transport

## 1. Mathematical Formulation & Architecture
Classical optimal transport requires probability measures to reside in the same ambient metric space $\mathcal{X}$. In modern geometric machine learning, comparing datasets across incomparable spaces (e.g. aligning 2D molecular graphs with 3D crystal structures, or comparing high-dimensional representations across distinct latent models) requires structural metric alignment.

### Sliced Wasserstein Distance (SWD)
Computing Wasserstein distance between distributions $P, Q \in \mathcal{P}(\mathbb{R}^d)$ suffers from high computational scaling (cubic LP or quadratic Sinkhorn).
The **Sliced Wasserstein Distance** leverages the 1D Radon transform:
$$\text{SW}_p(P, Q) = \left( \int_{S^{d-1}} W_p^p(\theta_\sharp P, \theta_\sharp Q) \, d\sigma(\theta) \right)^{1/p}$$
where $\theta_\sharp P$ is the orthogonal 1D push-forward projection onto unit direction $\theta \in S^{d-1}$.
In 1D, optimal transport is solved in closed form by sorting samples:
$$W_1(\theta^T X, \theta^T Y) = \frac{1}{N}\sum_{i=1}^N \lvert(\theta^T X)_{(i)} - (\theta^T Y)_{(i)}\rvert$$
Monte Carlo approximation over $K$ uniformly sampled random directions evaluates in $\mathcal{O}(K \cdot N \log N)$ time with zero optimization steps.

### Gromov-Wasserstein (GW) Discrepancy
Let $(X, d_X, \mu)$ and $(Y, d_Y, \nu)$ be two distinct metric measure spaces.
The **Gromov-Wasserstein distance** (Mémoli 2011) evaluates intra-space distance matrix distortions:
$$\text{GW}(D_X, D_Y) = \min_{T \in \Pi(\mu, \nu)} \sum_{i,j,k,l} \lvert D_X(i, k) - D_Y(j, l)\rvert^2 T_{ij} T_{kl}$$
This formulation is completely invariant to global translations, rotations, and isometric embeddings across disparate dimensions ($d_X \ne d_Y$).

### Entropic Regularized Optimization
Adding entropic penalty $-\varepsilon H(T)$ converts the non-convex Quadratic Assignment Problem into a sequence of smoothed entropic optimal transport subproblems:
$$C(T) = (D_X^2 p)\mathbf{1}_M^T + \mathbf{1}_N (D_Y^2 q)^T - 2 D_X T D_Y^T$$
Each entropic proximal step evaluates an inner Sinkhorn-Knopp scaling with log-shift stabilization:
$$T^{(t+1)} = \text{Sinkhorn}\left( C(T^{(t)}), \; p, \; q, \; \varepsilon \right)$$

## 2. Implementation Details
The module `Y03_Gromov_Wasserstein_and_Sliced_Optimal_Transport.py`:
- Implements Sliced Wasserstein with uniform hypersphere direction sampling and 1D sorting.
- Implements entropic Gromov-Wasserstein alignment with cost-shifted Sinkhorn iterations.
- Automatically handles quantile interpolation when sample sizes $N \ne M$.
- Verifies exact node correspondence recovery between 2D and 5D isometric spaces.

## 3. Asymptotic & Numerical Analysis
- **Sliced Wasserstein**: $\mathcal{O}(K \cdot N \log N)$ time, parallelizable across projections.
- **Gromov-Wasserstein**: Each iteration requires matrix multiplications of size $(N \times N) \times (N \times M) \times (M \times M)$ in $\mathcal{O}(N^2 M + N M^2)$ time.

## 4. Edge Cases & Numerical Stability
- In GW Sinkhorn scaling, small regularization $\varepsilon < 10^{-3}$ can induce underflow in $e^{-C/\varepsilon}$. Shifting $C \leftarrow C - \min(C)$ and stabilization with minimum denominator epsilons ($10^{-15}$) maintains numeric convergence.

## 5. Standalone Verification
The test suite validates:
- Identity of indiscernibles: $\text{SW}(X, X) = 0.0$.
- Linear spatial shift sensitivity under translation.
- Exact node permutation and near-zero distortion ($\text{GW} < 10^{-4}$) between an asymmetric 2D planar triangle and an isometric rotated copy embedded in 5D space.

## 6. Real-World Applications & Context
Crucial for cross-lingual word embedding alignment (mapping English to French without bilingual dictionaries), single-cell RNA-to-spatial transcriptomics registration, and graph neural network matching.

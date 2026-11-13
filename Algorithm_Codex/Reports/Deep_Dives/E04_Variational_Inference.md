# Deep Dive: Variational Inference (Mean-Field CAVI for Gaussian Mixtures)

## 1. The Beginner's Analogy: Fitting a Geometric Mold to an Amoeba
Imagine trying to describe an irregular, shifting amoeba under a microscope:
- The true posterior distribution of your model is an intricate, high-dimensional amoeba. Measuring its exact contours would take years of sampling (MCMC).
- **Variational Inference (VI)** changes sampling into **optimization**:
  - You bring a box of simple, rigid geometric molds (e.g. ellipses where the coordinates are independent: the mean-field family $q(z)$).
  - You place an ellipse over the amoeba and turn dials to adjust its size, location, and stretch until the ellipse fits the amoeba as snugly as possible (minimizing KL divergence).
- **Coordinate Ascent Variational Inference (CAVI)** adjusts one dial at a time while holding the others fixed, guaranteeing that the fit strictly improves at every turn.

---

## 2. The Deep Architecture/Algorithm Breakdown
Variational Inference approximates an intractable posterior $p(z \mid x)$ with a tractable distribution $q(z) \in \mathcal{Q}$:
1. **The Evidence Lower Bound (ELBO):**
   $$\ln p(x) = \text{ELBO}(q) + D_{KL}(q(z) \parallel p(z \mid x))$$
   $$\text{ELBO}(q) = \mathbb{E}_{q}[\ln p(x, z)] - \mathbb{E}_{q}[\ln q(z)]$$
   Because $D_{KL} \ge 0$, $\text{ELBO}(q) \le \ln p(x)$. Maximizing the ELBO directly minimizes the KL divergence to the true posterior!
2. **Mean-Field Approximation Family:** Assumes complete factorization across latent variables:
   $$q(z) = \prod_{j=1}^m q_j(z_j)$$
3. **Coordinate Ascent Variational Inference (CAVI):** Derives optimal coordinate update for factor $q_j(z_j)$:
   $$\ln q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\ln p(x, z)] + \text{const}$$

---

## 3. The Hardcore Mathematics

### 3.1 CAVI Optimal Factor Derivation
Isolate factor $q_j$ in the ELBO objective:
$$\text{ELBO}(q) = \int q_j(z_j) \mathbb{E}_{q_{-j}}[\ln p(x, z)] dz_j - \int q_j(z_j) \ln q_j(z_j) dz_j + \text{const}_{\neq j}$$
Notice this is the negative KL divergence between $q_j(z_j)$ and $\exp\left( \mathbb{E}_{q_{-j}}[\ln p(x, z)] \right)$:
$$= - D_{KL}\left( q_j(z_j) \;\parallel\; \frac{1}{Z_j}\exp\left( \mathbb{E}_{q_{-j}}[\ln p(x, z)] \right) \right) + \text{const}$$
Minimizing KL divergence to zero yields the **Optimal CAVI Update**:
$$q_j^*(z_j) \propto \exp\left( \mathbb{E}_{q_{-j}}[\ln p(x, z)] \right)$$

### 3.2 CAVI for Gaussian Mixture Models
For 1D GMM with latent cluster indicators $c_i \in \{1,\dots,K\}$ and cluster means $\mu_k \sim \mathcal{N}(0, \sigma_0^2)$:
Under mean-field family $q(c, \mu) = \prod_{i=1}^N q(c_i) \prod_{k=1}^K q(\mu_k)$:
1. **Update $q(c_i)$:** Categorical distribution with probabilities:
   $$\ln \gamma_{ik} \propto \mathbb{E}[\mu_k] x_i - \frac{1}{2} \mathbb{E}[\mu_k^2]$$
2. **Update $q(\mu_k)$:** Gaussian distribution $\mathcal{N}(m_k, s_k^2)$ where:
   $$s_k^2 = \left( \frac{1}{\sigma_0^2} + \sum_{i=1}^N \gamma_{ik} \right)^{-1}, \quad m_k = s_k^2 \left( \sum_{i=1}^N \gamma_{ik} x_i \right)$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Initialize variational parameters $m_k, s_k^2$ and responsibilities $\gamma_{ik}$.
2. **Iterate CAVI Steps:**
   - For each observation $i$: update categorical variational responsibilities $\gamma_{ik} \propto \exp(m_k x_i - 0.5(m_k^2 + s_k^2))$.
   - For each component $k$: update Gaussian mean parameters $m_k$ and variances $s_k^2$.
3. **Evaluate ELBO:** Compute ELBO at each iteration; verify monotonic ascent.
4. **Stop:** When ELBO improvement $< 10^{-4}$.

---

## 5. Limitations / When It Breaks
- **Variance Underestimation:** Reverse KL divergence $D_{KL}(q \parallel p)$ is mode-seeking (zero-forcing). The variational posterior $q(z)$ clamps tightly around the highest mode, severely underestimating true posterior variance.
- **Historical Fix:** Normalizing Flows (Module K02) and Markovian score matching.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/05_Autoencoders_VAE.md`. CAVI is the analytical, coordinate-wise ancestor of the Variational Autoencoder (VAE), which replaces coordinate updates with stochastic gradient backpropagation through the ELBO.

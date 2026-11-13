# 00. Probability and Mathematical Statistics Primer for Machine Learning

## 1. Probability Spaces & Random Variables

Probability theory provides the axiomatic language for reasoning under uncertainty. Formally, a **probability space** is defined by the Kolmogorov triple $(\Omega, \mathcal{F}, P)$:
1. **Sample Space ($\Omega$):** The set of all possible outcomes $\omega \in \Omega$.
2. **$\sigma$-Algebra ($\mathcal{F}$):** A collection of subsets of $\Omega$ containing $\emptyset$, closed under complementation and countable unions.
3. **Probability Measure ($P$):** A mapping $P: \mathcal{F} \rightarrow [0, 1]$ satisfying $P(\Omega) = 1$ and countable additivity for disjoint events:
   $$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i), \quad A_i \cap A_j = \emptyset \; (i \neq j)$$

A **random variable** $X: \Omega \rightarrow \mathbb{R}^d$ is a measurable function mapping the sample space to Euclidean space. For continuous random variables, the probability density function (PDF) $p(x)$ satisfies:
$$P(X \in B) = \int_B p(x) dx, \quad \int_{\mathbb{R}^d} p(x) dx = 1, \quad p(x) \ge 0$$

---

## 2. Expectations, Covariance, and Moment Generating Functions

### 2.1 Moments and Covariance Matrices
The **expected value** of a function $g(X)$ under probability distribution $p(x)$ is defined as:
$$\mathbb{E}_{X \sim p}[g(X)] = \int_{\mathbb{R}^d} g(x) p(x) dx$$

For a random vector $X \in \mathbb{R}^d$ with mean vector $\mu = \mathbb{E}[X]$, the **covariance matrix** $\Sigma \in \mathbb{R}^{d \times d}$ is positive semi-definite ($\Sigma \succeq 0$) and captures second-order multivariate dependencies:
$$\Sigma = \text{Cov}(X) = \mathbb{E}\left[(X - \mu)(X - \mu)^T\right] = \mathbb{E}[XX^T] - \mu\mu^T$$
Where individual entries correspond to pairwise covariances:
$$\Sigma_{jk} = \text{Cov}(X_j, X_k) = \mathbb{E}\left[(X_j - \mu_j)(X_k - \mu_k)\right]$$

### 2.2 Moment Generating Functions (MGF)
The MGF transforms a probability distribution into an analytic function generating moments via Taylor expansion:
$$M_X(t) = \mathbb{E}[e^{t^T X}], \quad \mathbb{E}[X^k] = \left. \frac{d^k M_X(t)}{dt^k} \right|_{t=0}$$

---

## 3. Parametric Probability Distributions in Machine Learning

### 3.1 The Multivariate Gaussian Distribution
The core distribution of classical machine learning and Bayesian inference:
$$\mathcal{N}(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$
- **Mahalanobis Distance:** $D_M^2(x, \mu) = (x - \mu)^T \Sigma^{-1} (x - \mu)$, measuring statistical distance accounting for coordinate covariances.
- **Precision Matrix:** $\Lambda = \Sigma^{-1}$. Sparse entries in $\Lambda$ signify conditional independence in Gaussian Markov Random Fields.

### 3.2 The Exponential Family
A unified parametric family spanning Gaussian, Bernoulli, Poisson, Gamma, and Dirichlet distributions:
$$p(x; \eta) = h(x) \exp\left( \eta^T T(x) - A(\eta) \right)$$
Where:
- $\eta$: Natural (canonical) parameter vector.
- $T(x)$: Sufficient statistics vector.
- $h(x)$: Base measure.
- $A(\eta) = \log \int h(x) \exp(\eta^T T(x)) dx$: Log-partition function (cumulant generating function).
Key identity: The gradient of the log-partition function yields the expected sufficient statistics:
$$\nabla_\eta A(\eta) = \mathbb{E}[T(X)], \quad \nabla_\eta^2 A(\eta) = \text{Cov}(T(X))$$

---

## 4. Information Theory Foundations

### 4.1 Shannon Entropy
The fundamental measure of uncertainty associated with random variable $X \sim p(x)$:
$$H(X) = - \sum_{x} p(x) \log_2 p(x) \quad \text{or} \quad h(X) = - \int p(x) \log p(x) dx$$

### 4.2 Kullback-Leibler (KL) Divergence
The relative entropy measuring the information lost when approximating true distribution $p$ with model $q$:
$$D_{KL}(p \parallel q) = \int p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{x \sim p}\left[ \log p(x) - \log q(x) \right]$$
Properties:
- Non-negativity (Gibbs' Inequality): $D_{KL}(p \parallel q) \ge 0$, with equality if and only if $p = q$ almost everywhere.
- Asymmetry: $D_{KL}(p \parallel q) \neq D_{KL}(q \parallel p)$. Forward KL is mean-seeking (zero-avoiding); reverse KL is mode-seeking (zero-forcing).

### 4.3 Mutual Information
The reduction in uncertainty about $X$ obtained by observing $Y$:
$$I(X; Y) = D_{KL}(p(x, y) \parallel p(x)p(y)) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

---

## 5. Matrix Calculus Reference for Gradient Derivations

In this repository, all learning rules are derived using vector and matrix calculus without automatic differentiation.

### 5.1 Standard Vector Derivative Identities
Let $x \in \mathbb{R}^n, A \in \mathbb{R}^{n \times n}, y, b \in \mathbb{R}^n$:
1. $\nabla_x (a^T x) = a$
2. $\nabla_x (x^T A x) = (A + A^T) x$. (If $A$ is symmetric, $\nabla_x (x^T A x) = 2Ax$).
3. $\nabla_x \|Ax - b\|_2^2 = 2 A^T (Ax - b)$
4. $\nabla_W \text{Tr}(A W B) = A^T B^T$
5. $\nabla_W \log \det(W) = W^{-T} = (W^{-1})^T$

### 5.2 Denominator Layout vs. Numerator Layout
Unless explicitly specified, we adhere to the **numerator layout convention**: the derivative of scalar $f$ with respect to column vector $x \in \mathbb{R}^d$ is a column vector $\nabla_x f \in \mathbb{R}^{d \times 1}$ whose $j$-th entry is $\frac{\partial f}{\partial x_j}$.

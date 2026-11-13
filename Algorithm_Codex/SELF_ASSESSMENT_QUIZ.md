# Master Self-Assessment Quiz & Diagnostic Guide

Test your mathematical, code-level, and algorithmic comprehension across all 12 modules before deploying models to production or research. Click on **"<details><summary>Check Answer & Explanation</summary>"** under each question to reveal the rigorous derivation, code implementation, and justification.

---

## Module A: Statistical Foundations

### Question 1: Derivation of the Normal Equation
Consider the ordinary least squares objective $\mathcal{L}(\beta) = \frac{1}{2} \|y - X\beta\|_2^2$, where $X \in \mathbb{R}^{m \times d}$ has full column rank ($	ext{rank}(X) = d$) and $y \in \mathbb{R}^{m \times 1}$.
1. Derive the gradient $\nabla_\beta \mathcal{L}(\beta)$ with respect to $\beta$ using vector calculus.
2. Set the gradient to zero and show why $(X^T X)$ is guaranteed to be invertible, arriving at the closed-form Normal Equation.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Expand the squared $\ell_2$ norm:
   $$\mathcal{L}(\beta) = \frac{1}{2} (y - X\beta)^T (y - X\beta) = \frac{1}{2} (y^T y - 2 y^T X \beta + \beta^T X^T X \beta)$$
   Taking the derivative with respect to vector $\beta$:
   $$\nabla_\beta \mathcal{L}(\beta) = -X^T y + X^T X \beta = X^T (X\beta - y)$$
2. Setting $\nabla_\beta \mathcal{L}(\beta) = 0$:
   $$X^T X \beta = X^T y$$
   Because $\text{rank}(X) = d$, the columns of $X$ are linearly independent. For any non-zero $v \in \mathbb{R}^d$, $Xv \neq 0$, which implies $v^T (X^T X) v = \|Xv\|_2^2 > 0$. Thus, $X^T X \in \mathbb{R}^{d \times d}$ is strictly positive definite, guaranteeing that its inverse $(X^T X)^{-1}$ exists:
   $$\hat{\beta} = (X^T X)^{-1} X^T y$$
</details>

---

## Module B: Classical Supervised Learning

### Question 2: Softmax Loss Vectorized Gradient
In multinomial logistic regression with $K$ classes, the probability assigned to class $k$ for input sample $x_i \in \mathbb{R}^{1 \times d}$ with weight matrix $W \in \mathbb{R}^{d \times K}$ is:
$$p_{ik} = \frac{\exp(x_i W_{:,k})}{\sum_{j=1}^K \exp(x_i W_{:,j})}$$
For a one-hot ground truth label vector $y_i \in \mathbb{R}^{1 \times K}$ and cross-entropy loss $\mathcal{L}_i = -\sum_{k=1}^K y_{ik} \ln p_{ik}$, what is the exact matrix expression for the gradient $\nabla_W \mathcal{L}$ over a full batch $X \in \mathbb{R}^{m \times d}$ and $Y \in \mathbb{R}^{m \times K}$?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
The gradient of cross-entropy with respect to the linear logits $Z = XW$ is the residual error $P - Y$, where $P \in \mathbb{R}^{m \times K}$ is the matrix of predicted softmax probabilities:
$$\frac{\partial \mathcal{L}}{\partial Z} = P - Y$$
By applying the matrix chain rule $\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Z}$:
$$\nabla_W \mathcal{L} = \frac{1}{m} X^T (P - Y)$$
In NumPy:
```python
P = np.exp(Z - np.max(Z, axis=1, keepdims=True))
P /= np.sum(P, axis=1, keepdims=True)
dW = (1.0 / m) * (X.T @ (P - Y))
```
</details>

---

## Module C: Unsupervised Learning & Dimensionality Reduction

### Question 3: Equivalence of PCA via Covariance Eigendecomposition and SVD
Let centered data matrix $X \in \mathbb{R}^{m \times d}$ have zero empirical mean ($\sum_{i=1}^m x_{ij} = 0$).
1. State the relationship between the sample covariance matrix $\Sigma = \frac{1}{m} X^T X$ and the compact Singular Value Decomposition of $X = U S V^T$.
2. Prove that the singular vectors $V$ are identical to the eigenvectors of $\Sigma$, and express the eigenvalues $\lambda_j$ in terms of the singular values $s_j$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Substitute the SVD $X = U S V^T$ into the covariance formula:
   $$\Sigma = \frac{1}{m} X^T X = \frac{1}{m} (U S V^T)^T (U S V^T) = \frac{1}{m} V S^T U^T U S V^T$$
   Since $U$ has orthonormal columns ($U^T U = I_m$):
   $$\Sigma = V \left( \frac{1}{m} S^2 \right) V^T$$
2. The spectral theorem for symmetric matrices states $\Sigma = V \Lambda V^T$. Comparing the two factorizations directly:
   - The right singular vectors $V$ of data matrix $X$ are the principal eigenvectors of covariance matrix $\Sigma$.
   - The covariance eigenvalues are related to singular values by:
     $$\lambda_j = \frac{s_j^2}{m}$$
This proves that performing SVD on $X$ computes PCA directly without explicitly forming the $O(d^2)$ covariance matrix $X^T X$, providing superior numerical precision and stability against floating-point catastrophic cancellation.
</details>

---

## Module D: Ensemble Methods

### Question 4: AdaBoost Sample Weight Update Mechanism
In AdaBoost.M1 with binary labels $y_i \in \{-1, +1\}$, why does the sample weight update rule $w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(x_i))$ exponentially upweight misclassified points? What prevents weights from overflowing after 100 boosting rounds?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. For correct classifications, $y_i h_m(x_i) = +1$, so $w_i \leftarrow w_i e^{-\alpha_m}$. For misclassifications, $y_i h_m(x_i) = -1$, so $w_i \leftarrow w_i e^{+\alpha_m}$. Since weak learner error $\epsilon_m < 0.5$, learner coefficient $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right) > 0$. Thus, correctly classified samples are multiplied by a factor $< 1$, while misclassified samples are multiplied by an inverse factor $> 1$.
2. To prevent weight explosion or underflow, AdaBoost divides all weights by the normalization constant $Z_m = \sum_{i=1}^m w_i^{(m+1)}$ after every round:
   $$w_i^{(m+1)} \leftarrow \frac{w_i^{(m+1)}}{Z_m}$$
   This guarantees that $\sum_{i=1}^m w_i^{(m+1)} = 1$ at all times, maintaining a valid probability distribution over training instances.
</details>

---

## Module E: Probabilistic Graphical Models & Bayesian Methods

### Question 5: Viterbi Algorithm Log-Space Invariant
In Hidden Markov Models, why is the Viterbi decoding algorithm $\max_{z_{1:T}} P(z_{1:T}, x_{1:T})$ strictly computed in log-probability space rather than raw probability products?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
The joint probability is a product of $T$ emission and $T-1$ transition probabilities:
$$P(z_{1:T}, x_{1:T}) = P(z_1) P(x_1|z_1) \prod_{t=2}^T P(z_t|z_{t-1}) P(x_t|z_t)$$
Because each probability is bounded in $[0, 1]$, multiplying hundreds of terms causes standard 64-bit IEEE-754 floating-point underflow ($< 10^{-308}$), rounding values to $0.0$ and destroying the argmax path tracking.
In log-space:
$$V_{t,j} = \ln P(x_t|z_t=j) + \max_{i} \left[ V_{t-1,i} + \ln A_{ij} \right]$$
Products become numerical additions, preventing underflow while preserving exact rank ordering since $\ln(x)$ is strictly monotonically increasing.
</details>

---

## Module F: Classical Time Series

### Question 6: Stationarity and the Characteristic Polynomial in AR($p$)
For an autoregressive process of order $p$, $X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t$, what is the necessary and sufficient condition on the roots of its characteristic equation $\Phi(z) = 1 - \sum_{i=1}^p \phi_i z^i = 0$ for the process to be weakly stationary?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
The process is weakly (covariance) stationary if and only if **all roots of the characteristic polynomial $\Phi(z) = 0$ lie strictly outside the unit circle in the complex plane**:
$$|z_i| > 1 \quad \forall i \in \{1, \dots, p\}$$
If a root lies on the unit circle ($|z| = 1$), the process contains a unit root (integrated process $I(1)$), exhibiting non-stationary random walk behavior requiring differencing $(1 - B)X_t$. If a root lies inside the unit circle ($|z| < 1$), the process explodes exponentially.
</details>

---

## Module G: Reinforcement Learning

### Question 7: SARSA vs. Q-Learning Target Difference
Write the exact target equation for SARSA and Q-learning updates for state-action pair $(s_t, a_t)$ after observing reward $r_{t+1}$ and next state $s_{t+1}$. Why is SARSA characterized as **on-policy** while Q-learning is **off-policy**?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
- **SARSA Update:**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$
  Where $a_{t+1} \sim \pi(\cdot | s_{t+1})$ is the actual next action chosen by the current behavioral policy (e.g. $\epsilon$-greedy).
- **Q-Learning Update:**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$
  Where the target uses the greedy optimal action $\max_{a'} Q(s_{t+1}, a')$ regardless of what action the agent actually executes next.
- **Justification:** SARSA evaluates the policy it is currently following (including exploration mistakes), making it safe near hazards (e.g. avoiding cliff edges). Q-learning learns the optimal greedy policy directly while executing an exploratory behavioral policy, making it off-policy.
</details>

---

## Module H: Evolutionary & Nature-Inspired Optimization

### Question 8: NSGA-II Crowding Distance Invariant
In multi-objective optimization using NSGA-II, how does the crowding distance metric prevent population collapse onto a single point of the Pareto front? How are boundary solutions handled?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
Crowding distance measures the perimeter of the cuboid formed by using the nearest neighbors on either side along each objective dimension:
$$d_i = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$
- Boundary individuals with minimum and maximum values for each objective are assigned an **infinite crowding distance** ($d_{\text{boundary}} = \infty$).
- During crowded-comparison tournament selection, between two solutions with equal non-domination rank, the solution with the larger crowding distance is strictly preferred ($i \prec_n j$ if $r_i < r_j$, or ($r_i = r_j$ and $d_i > d_j$)). This forces the genetic operators to preserve diverse solutions along the entire Pareto frontier.
</details>

---

## Module I: Classical Optimization Theory

### Question 9: Adam Bias Correction Mathematical Justification
In Kingma & Ba's Adam optimizer, the running moment estimates are initialized at zero: $m_0 = 0, v_0 = 0$.
Given $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, show why $\mathbb{E}[m_t] = (1 - \beta_1^t) \mathbb{E}[g_t]$ assuming true gradients $g_t$ are identically distributed, and explain why dividing by $(1 - \beta_1^t)$ is necessary in early training steps.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
Unroll the recurrence for $m_t$ starting from $m_0 = 0$:
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
Take the mathematical expectation of both sides:
$$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = \mathbb{E}[g] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$
Using the finite sum formula for a geometric series $\sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$:
$$\mathbb{E}[m_t] = \mathbb{E}[g] (1 - \beta_1) \left( \frac{1 - \beta_1^t}{1 - \beta_1} \right) = (1 - \beta_1^t) \mathbb{E}[g]$$
At step $t=1$ with $\beta_1 = 0.9$, $1 - \beta_1^1 = 0.1$, meaning the uncorrected estimate $m_1$ is biased toward zero by an order of magnitude ($90\%$ shrinkage). Dividing by $(1 - \beta_1^t)$ corrects this discrepancy, ensuring an unbiased estimator $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$ from step 1.
</details>

---

## Module J: Alternative Learning Paradigms

### Question 10: Axiomatic Properties of Shapley Values
In cooperative game theory and Explainable AI (SHAP), what are the 4 fundamental axioms that the Shapley value allocation $\phi_i(v)$ uniquely satisfies?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Efficiency:** The sum of all feature attributions equals the total model prediction shift from base expectation:
   $$\sum_{i \in F} \phi_i(v) = v(F) - v(\emptyset)$$
2. **Symmetry:** If two features $i$ and $j$ contribute equally to all possible coalitions $S \subseteq F \setminus \{i, j\}$ ($v(S \cup \{i\}) = v(S \cup \{j\})$), their attributions are equal: $\phi_i(v) = \phi_j(v)$.
3. **Dummy / Null Player:** If feature $i$ contributes nothing to any coalition ($v(S \cup \{i\}) = v(S)$ for all $S$), its attribution is zero: $\phi_i(v) = 0$.
4. **Additivity:** If a model's prediction is the sum of two independent mechanisms $v(S) = v_1(S) + v_2(S)$, the Shapley values sum directly: $\phi_i(v_1 + v_2) = \phi_i(v_1) + \phi_i(v_2)$.
Shapley values are the *only* attribution method that satisfies all four axioms simultaneously.
</details>

---

## Module K: Neural-Net Techniques Missing From Deep Learning

### Question 11: Equivalence of Modern Continuous Hopfield Networks to Attention
Let $X = [x_1, \dots, x_N]^T \in \mathbb{R}^{N \times d}$ represent stored memory patterns.
In Ramsauer et al. (2020), the modern continuous Hopfield energy is defined as:
$$E(z) = -\text{lse}(\beta X z) + \frac{1}{2} z^T z$$
Where $\text{lse}(u) = \ln \sum_{i=1}^N \exp(u_i)$ is the LogSumExp function.
Derive the concave-convex procedure update step $z^{t+1}$ and show that it is mathematically identical to Transformer Scaled Dot-Product Self-Attention!

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Gradient of the energy function with respect to state vector $z$:
   $$\nabla_z E(z) = -\nabla_z \ln \sum_{i=1}^N \exp(\beta x_i^T z) + z$$
   Applying the chain rule:
   $$\nabla_z \ln \sum_{i=1}^N \exp(\beta x_i^T z) = \frac{\sum_{i=1}^N \exp(\beta x_i^T z) \cdot \beta x_i}{\sum_{j=1}^N \exp(\beta x_j^T z)} = \beta X^T \text{softmax}(\beta X z)$$
2. Setting the fixed-point update step where $z^{t+1}$ cancels the non-linear gradient:
   $$z^{t+1} = X^T \text{softmax}(\beta X z^t)$$
3. Comparing to Transformer Attention: Let state $z$ be the Query ($Q$), stored patterns $X$ be the Keys ($K$) and Values ($V$), and inverse temperature $\beta = \frac{1}{\sqrt{d_k}}$:
   $$z^{t+1} = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
Thus, a single retrieval step in a Continuous Hopfield Network is mathematically equivalent to the Transformer attention mechanism!
</details>

---

## Module L: Classical AI Search & Symbolic Reasoning

### Question 12: Admissibility and Consistency of A* Heuristics
1. Define what makes a heuristic function $h(n)$ **admissible** and **consistent (monotonic)**.
2. Prove that if $h(n)$ is consistent, A* using `closed_set` never needs to re-open a previously expanded node.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Definitions:**
   - **Admissibility:** $h(n) \le h^*(n)$ for all nodes $n$, where $h^*(n)$ is the true optimal cost from $n$ to the nearest goal state.
   - **Consistency (Monotonicity):** For every node $n$ and every successor $n'$ generated by action with cost $c(n, a, n')$:
     $$h(n) \le c(n, a, n') + h(n')$$
     And $h(G) = 0$ for any goal state $G$. (This is the triangle inequality in heuristic space).
2. **Proof:**
   Consider path cost $f(n) = g(n) + h(n)$. For any successor $n'$:
   $$f(n') = g(n') + h(n') = g(n) + c(n, a, n') + h(n')$$
   By consistency, $c(n, a, n') + h(n') \ge h(n)$, therefore:
   $$f(n') \ge g(n) + h(n) = f(n)$$
   This implies that the sequence of $f$-values of expanded nodes is non-decreasing ($f(n_1) \le f(n_2) \le \dots$). When node $n$ is selected from the priority queue for expansion, its discovered path cost $g(n)$ must be optimal; any other open path has $f \ge f(n) \ge g(n)$. Consequently, no subsequent shorter path to $n$ can ever be discovered, guaranteeing that closed nodes never need to be revisited or reopened.
</details>
---

## Advanced Diagnostic Questions

### Question 13: Kaplan-Meier Product-Limit and Right-Censoring (Module F)
In survival analysis, why does discarding right-censored observations or treating their censoring time as event time introduce severe selection bias? Derive the Kaplan-Meier product-limit formula $\hat{S}(t) = \prod_{t_i \le t} (1 - d_i / n_i)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Selection Bias:** Discarding censored individuals systematically biases survival estimates downward (toward fragility), because censored units survived *at least* up to their censoring time. Conversely, treating censoring time as failure time underestimates survival longevity.
2. **Derivation:** Partition the continuous timeline into non-overlapping intervals $[t_{i-1}, t_i]$ at unique observed event times. By the chain rule of conditional probability:
   P(T > t_k) = P(T > t_1) \cdot P(T > t_2 \mid T > t_1) \cdots P(T > t_k \mid T > t_{k-1})
   At time $, the conditional probability of experiencing the event given survival to $ is $rac{d_i}{n_i}$, where $ is the risk set. Hence the conditional survival probability is  - rac{d_i}{n_i}$.
   Multiplying yields the Kaplan-Meier estimator:
   \hat{S}(t) = \prod_{i: t_i \le t} \left( 1 - rac{d_i}{n_i} 
ight)
</details>

---

### Question 14: NEAT Innovation Numbers and Topological Crossover (Module H)
Why does crossing over two neural network graphs with arbitrary differing topologies fail using standard genetic crossover? How do global innovation numbers solve this problem without graph isomorphism?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **The Competing Conventions Problem:** Two neural networks may encode identical functional behavior using different internal node orderings or topologies. A naive coordinate crossover splits and splices incompatible functional pathways, producing catastrophic functional disruption (non-viable offspring).
2. **Global Innovation Tracking:** Every time a structural mutation creates a new gene (new node or connection), a global integer counter increments and tags the gene with an immutable innovation number. When crossing over two genomes, genes with matching innovation numbers are homologous and lined up directly; disjoint and excess genes are inherited from the fitter parent. This enables $\mathcal{O}(G)$ linear alignment without solving NP-complete graph isomorphism.
</details>

---

### Question 15: InfoNCE Variational Mutual Information Lower Bound (Module J)
Prove that minimizing the InfoNCE loss $\mathcal{L}_{	ext{InfoNCE}} = -\mathbb{E}\left[ \log rac{\exp(f(x, y^+))}{\exp(f(x, y^+)) + \sum_{j=1}^K \exp(f(x, y_j^-))} 
ight]$ maximizes a lower bound on the mutual information (X; Y)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
By Bayes' rule, the posterior probability that sample $ is the true positive pair among $ negative distractors drawn from marginal (y)$ is:
p(D=1 \mid X=x, Y=y) = rac{p(x, y) \prod_{j=1}^K p(y_j^-)}{\sum_{i=1}^{K+1} p(x, y_i) \prod_{j 
e i} p(y_j^-)} = rac{rac{p(x, y)}{p(x)p(y)}}{rac{p(x, y)}{p(x)p(y)} + \sum_{j=1}^K rac{p(x, y_j^-)}{p(x)p(y_j^-)}}
Letting neural critic (x, y) = \log rac{p(x,y)}{p(x)p(y)} + c$, the optimal critic satisfies ^{f(x, y)} \propto rac{p(x, y)}{p(x)p(y)}$.
Evaluating expected loss:
\mathcal{L}_{	ext{InfoNCE}} \ge -\mathbb{E}\left[ \log rac{rac{p(x, y)}{p(x)p(y)}}{rac{p(x, y)}{p(x)p(y)} + K} 
ight] pprox \log(K) - I(X; Y)
Hence:
I(X; Y) \ge \log(K) - \mathcal{L}_{	ext{InfoNCE}}
Minimizing InfoNCE directly tightens the lower bound on mutual information between representations.
</details>

---

### Question 16: RealNVP Invertibility & Triangular Jacobian (Module K)
In RealNVP Normalizing Flows, show why the scale and translation sub-networks (\cdot)$ and (\cdot)$ do NOT need to be invertible for the overall affine coupling layer to be strictly invertible with tractable Jacobian.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Coupling Forward Step:**
   y_1 = x_1, \quad y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
2. **Inversion:**
   Because  = x_1$, $ is available unchanged at inversion. Therefore, (x_1) = s(y_1)$ and (x_1) = t(y_1)$ can be computed directly by running $ and $ *forward*:
   x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))
   Neither $ nor $ is ever inverted!
3. **Jacobian:**
   The partial derivative $rac{\partial y_1}{\partial x_2} = 0$, producing a block triangular Jacobian:
   \det J = \det egin{bmatrix} I & 0 \ rac{\partial y_2}{\partial x_1} & 	ext{diag}(\exp(s(x_1))) \end{bmatrix} = \prod_j \exp(s_j(x_1))
   Taking logarithms yields the simple sum $\ln |\det J| = \sum_j s_j(x_1)$, computed in $\mathcal{O}(D)$ time.
</details>

---

### Question 17: PageRank Power Iteration & Perron-Frobenius Theorem (Module L)
In Google's PageRank algorithm, why does adding damping factor  = 0.85$ guarantee unique convergence of power iteration regardless of initial probability vector True

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
A raw link adjacency matrix can contain dead ends (dangling nodes with out-degree 0) and spider traps (isolated cycles), causing power iteration to oscillate or leak probability mass.
The Google Matrix blends the transition matrix $ with random teleportation:
G = d M + rac{1 - d}{N} \mathbf{1} \mathbf{1}^T
Because  - d > 0$, every entry {ij} \ge rac{1 - d}{N} > 0$. Thus, $ is strictly positive, irreducible, and aperiodic.
By the **Perron-Frobenius Theorem**, any positive stochastic matrix possesses a unique dominant eigenvalue $\lambda_1 = 1$, and all other eigenvalues satisfy $|\lambda_i| \le d < 1$. Therefore, the Markov chain has a unique stationary distribution $p^* = G p^*$, and power iteration converges exponentially at rate $\mathcal{O}(d^t)$.
</details>

---

### Question 18: HNSW Diversity Heuristic & LSH S-Curve Amplification (Module M)
1. In HNSW, why does connecting each node to its $M$ closest neighbors fail in high dimensions, and how does the diversity heuristic solve clustering along a single directional ray?
2. In Locality-Sensitive Hashing, derive why partitioning $k = b \cdot r$ hashes into $b$ bands of $r$ rows creates a sharp S-curve candidate collision probability $P_{\text{cand}}(s) = 1 - (1 - s^r)^b$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **HNSW Diversity Heuristic:** In high-dimensional spaces, points often cluster densely along low-dimensional manifolds. If a node connects strictly to its $M$ closest Euclidean neighbors, all $M$ edges point in nearly the same directional Voronoi cone, creating isolated topological cliques with poor global graph navigability. HNSW enforces that a candidate neighbor $e$ is connected only if $d(e, u) < d(e, n)$ for all already-selected neighbors $n$, guaranteeing connections span diverse spatial angles.
2. **LSH S-Curve Derivation:**
   - Two items match on all $r$ rows of a single band with probability $s^r$.
   - The probability that they fail to match in that band is $1 - s^r$.
   - The probability that they fail to match in *all* $b$ independent bands is $(1 - s^r)^b$.
   - Therefore, the probability that they collide in at least one band (becoming a candidate pair) is $P(s) = 1 - (1 - s^r)^b$.
   - This sigmoidal curve has an inflection point at $s^* \approx (1/b)^{1/r}$, sharply rejecting items with $s < s^*$ while retrieving candidates with $s > s^*$.
</details>

---

### Question 19: Reverse-Mode VJP vs. Forward Dual Numbers (Module N)
Why is reverse-mode automatic differentiation $\mathcal{O}(1)$ with respect to input dimension for scalar loss functions, while forward-mode dual numbers are $\mathcal{O}(N)$? When is forward-mode preferred over reverse-mode?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Complexity Asymmetry:**
   - For a mapping $f: \mathbb{R}^N \to \mathbb{R}^M$, the Jacobian $J$ has size $M \times N$.
   - **Forward-mode (Dual numbers)** computes Jacobian-Vector Products (JVPs) $J \cdot v$, producing one column of the Jacobian per forward pass. When $M = 1$ and $N = 10^8$ (deep learning loss), forward-mode requires $10^8$ passes!
   - **Reverse-mode (Backpropagation)** computes Vector-Jacobian Products (VJPs) $v^T \cdot J$, producing one row of the Jacobian per backward pass. When $M = 1$, the full gradient $\nabla_\theta \mathcal{L}$ is computed in a single backward pass.
2. **When Forward-Mode Wins:**
   - When input dimension $N$ is small and output dimension $M$ is massive ($N \ll M$, e.g., robot kinematics, parametric curve generation).
   - When memory is constrained: forward mode requires zero storage of intermediate forward activations ($\mathcal{O}(1)$ memory), completely avoiding GPU out-of-memory errors during sensitivity analysis.
</details>

---

### Question 20: Sinkhorn Entropic Regularization & Gibbs Kernel (Module O)
Show how adding entropy regularization $-\varepsilon H(P)$ converts the discrete Kantorovich optimal transport linear program into an unconstrained smooth convex problem with unique matrix scaling solution $P^* = \text{diag}(u) K \text{diag}(v)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Regularized Lagrangian:**
   $$\mathcal{L}(P, \alpha, \beta) = \sum_{ij} P_{ij} C_{ij} + \varepsilon P_{ij} (\ln P_{ij} - 1) - \alpha^T (P \mathbf{1} - p) - \beta^T (P^T \mathbf{1} - q)$$
2. **First-Order Optimality (KKT Stationarity):**
   $$\frac{\partial \mathcal{L}}{\partial P_{ij}} = C_{ij} + \varepsilon \ln P_{ij} - \alpha_i - \beta_j = 0$$
   $$\ln P_{ij} = \frac{\alpha_i + \beta_j - C_{ij}}{\varepsilon} \implies P_{ij}^* = \exp\left(\frac{\alpha_i}{\varepsilon}\right) \exp\left(-\frac{C_{ij}}{\varepsilon}\right) \exp\left(\frac{\beta_j}{\varepsilon}\right)$$
3. **Matrix Scaling Decomposition:**
   Letting $u_i = \exp(\alpha_i / \varepsilon)$, $v_j = \exp(\beta_j / \varepsilon)$, and Gibbs kernel $K_{ij} = \exp(-C_{ij}/\varepsilon)$:
   $$P^* = \text{diag}(u) K \text{diag}(v)$$
   Satisfying the marginal constraints $P \mathbf{1} = p$ and $P^T \mathbf{1} = q$ reduces to Sinkhorn-Knopp iterations: $u \leftarrow p / (Kv)$ and $v \leftarrow q / (K^T u)$ in $\mathcal{O}(N^2)$ time.
</details>

---

### Question 21: Denoising Score Matching & The Intractable Partition Function (Module P)
Why does maximum likelihood training fail for general Energy-Based Models $p_\theta(x) = \frac{1}{Z_\theta} e^{-E_\theta(x)}$, and how does Denoising Score Matching eliminate $Z_\theta$ without MCMC?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Intractability of Normalizer $Z_\theta$:**
   The log-likelihood is $\ln p_\theta(x) = -E_\theta(x) - \ln Z_\theta$. Its parameter gradient requires evaluating $\nabla_\theta \ln Z_\theta = -\mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]$, which requires expensive, non-convergent MCMC sampling at every gradient step.
2. **Score Function Elimination:**
   Taking the gradient with respect to *input* $x$ rather than parameters $\theta$:
   $$s_\theta(x) \triangleq \nabla_x \ln p_\theta(x) = -\nabla_x E_\theta(x) - \nabla_x \ln Z_\theta = -\nabla_x E_\theta(x)$$
   Because $Z_\theta = \int e^{-E_\theta(x)} dx$ is a constant with respect to $x$, its derivative $\nabla_x Z_\theta = 0$ vanishes identically!
3. **Vincent's DSM Theorem:**
   By perturbing data with Gaussian noise $\tilde{x} = x + \sigma \varepsilon$, the conditional distribution score is known analytically: $\nabla_{\tilde{x}} \ln q(\tilde{x}|x) = -(\tilde{x} - x)/\sigma^2$. Minimizing the expected Euclidean distance between $s_\theta(\tilde{x})$ and $-(\tilde{x} - x)/\sigma^2$ provably trains $s_\theta$ to match the true data score without ever evaluating $Z_\theta$.
</details>

---

### Question 22: QMIX Monotonicity Invariant & VCG Truthfulness (Module Q)
1. In cooperative multi-agent RL, prove why enforcing $\frac{\partial Q_{\text{tot}}}{\partial Q_a} \ge 0$ guarantees that decentralized greedy action selection maximizes joint team return.
2. In mechanism design, prove why the Clarke pivot payment rule makes bidding your exact truthful valuation a weakly dominant strategy in VCG auctions.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **QMIX Monotonicity Invariant:**
   Suppose each agent greedily selects $u^{*a} = \arg\max_{u^a} Q_a(\tau^a, u^a)$. For any alternative joint action $\mathbf{u} = (u^1, \dots, u^N)$, by definition $Q_a(\tau^a, u^{*a}) \ge Q_a(\tau^a, u^a)$ for all $a$. Because all partial derivatives are non-negative ($\frac{\partial Q_{\text{tot}}}{\partial Q_a} \ge 0$), the total function $Q_{\text{tot}}$ is monotonically non-decreasing in every argument. Therefore:
   $$Q_{\text{tot}}(\mathbf{\tau}, \mathbf{u}^*, s) \ge Q_{\text{tot}}(\mathbf{\tau}, \mathbf{u}, s)$$
   This satisfies the Individual-Global-Max (IGM) condition, allowing agents to choose actions in $\mathcal{O}(N \cdot |U|)$ time without communicating.
2. **VCG Truthfulness Proof:**
   Agent $i$'s utility is $u_i(b_i, b_{-i}) = v_i(x^*) - p_i(b)$, where Clarke payment is $p_i(b) = \max_\omega \sum_{j \ne i} b_j(\omega) - \sum_{j \ne i} b_j(x^*)$.
   Substituting $p_i$:
   $$u_i(b_i, b_{-i}) = \left[ v_i(x^*) + \sum_{j \ne i} b_j(x^*) \right] - \max_\omega \sum_{j \ne i} b_j(\omega)$$
   The second term is independent of agent $i$'s bid.
   The mechanism chooses $x^*$ to maximize $\sum_j b_j(x^*) = b_i(x^*) + \sum_{j \ne i} b_j(x^*)$.
   By setting bid $b_i = v_i$, the mechanism's optimization objective becomes identical to agent $i$'s utility bracket. Any deviation $b_i \ne v_i$ can only cause the mechanism to pick a sub-optimal outcome for agent $i$, making truthful bidding a dominant strategy.
</details>

---

## Module R: Quantum Computing

### Question 23: Unitary Evolution and the Born Rule (Module R)
1. Why must all closed quantum system time-evolution operators be represented by unitary matrices ($U^\dagger U = I$)?
2. Given state $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}e^{i\pi/4}|1\rangle$, calculate its Bloch sphere coordinates $(x, y, z)$ and the probability of measuring outcome $|1\rangle$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Conservation of Total Probability:**
   The total probability of measuring any state across the computational basis must equal 1: $\langle\psi|\psi\rangle = 1$. If a state evolves under linear operator $U$ to $|\psi'\rangle = U|\psi\rangle$, then:
   $$\langle\psi'|\psi'\rangle = \langle\psi| U^\dagger U |\psi\rangle = \langle\psi|\psi\rangle = 1$$
   For this equality to hold identically for all vectors in Hilbert space, $U^\dagger U = I$ is both necessary and sufficient.
2. **Bloch Coordinates Calculation:**
   Here $\alpha = 1/\sqrt{3} = \cos(\theta/2)$ and $\beta = \sqrt{2/3}e^{i\pi/4} = \sin(\theta/2)e^{i\phi}$.
   - $P(1) = |\beta|^2 = (\sqrt{2/3})^2 = 2/3 \approx 0.667$.
   - $z = \langle\psi|Z|\psi\rangle = |\alpha|^2 - |\beta|^2 = 1/3 - 2/3 = -1/3 \approx -0.333$.
   - $x = 2\text{Re}(\alpha^* \beta) = 2(1/\sqrt{3})(\sqrt{2/3})\cos(\pi/4) = 2(\sqrt{2}/3)(1/\sqrt{2}) = 2/3 \approx 0.667$.
   - $y = 2\text{Im}(\alpha^* \beta) = 2(1/\sqrt{3})(\sqrt{2/3})\sin(\pi/4) = 2/3 \approx 0.667$.
   Check unit norm: $x^2 + y^2 + z^2 = (4/9) + (4/9) + (1/9) = 9/9 = 1$.
</details>

---

### Question 24: Variational Quantum Eigensolver (VQE) and the Rayleigh-Ritz Bound (Module R)
Prove why the VQE expectation value $\langle\psi(\theta)|H|\psi(\theta)\rangle$ is guaranteed to be an upper bound on the true ground state energy $E_0$, and explain how Jordan-Wigner transformation enables Hamiltonian evaluation on quantum processors.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Rayleigh-Ritz Variational Principle:**
   Let the Hermitian Hamiltonian $H$ have complete orthonormal eigenstates $\{|\phi_k\rangle\}$ with eigenvalues $E_0 \le E_1 \le E_2 \dots$. Any normalized parameterized trial state expands as $|\psi(\theta)\rangle = \sum_k c_k |\phi_k\rangle$ with $\sum_k |c_k|^2 = 1$.
   The expectation value is:
   $$\langle H\rangle_\theta = \langle\psi(\theta)| H |\psi(\theta)\rangle = \sum_k |c_k|^2 E_k \ge E_0 \sum_k |c_k|^2 = E_0$$
   Equality holds if and only if $|\psi(\theta)\rangle$ is an exact ground eigenstate.
2. **Jordan-Wigner Transformation:**
   Second-quantized electronic Hamiltonians contain fermionic creation and annihilation operators $a_i^\dagger, a_j$ satisfying anti-commutation relations $\{a_i, a_j^\dagger\} = \delta_{ij}$. Because physical quantum gates operate on distinguishable qubits obeying tensor product commutation, Jordan-Wigner maps fermionic operators to Pauli spin strings:
   $$a_j^\dagger = \left( \prod_{k=0}^{j-1} Z_k \right) \frac{X_j - i Y_j}{2}$$
   This yields $H = \sum_k c_k P_k$ where each $P_k \in \{I, X, Y, Z\}^{\otimes n}$ is an easily measurable tensor product of single-qubit Pauli observables.
</details>

---

### Question 25: Geometric Derivation of Grover's Optimal Iteration Count (Module R)
In Grover's search algorithm for a single target state in $N = 2^n$ items, derive why the optimal number of iterations is $R \approx \frac{\pi}{4}\sqrt{N}$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Subspace Angle:**
   The uniform initial superposition state is $|s\rangle = \frac{1}{\sqrt{N}}|w\rangle + \sqrt{\frac{N-1}{N}}|w^\perp\rangle = \sin(\theta/2)|w\rangle + \cos(\theta/2)|w^\perp\rangle$.
   For large $N$, $\sin(\theta/2) = 1/\sqrt{N} \implies \theta/2 \approx 1/\sqrt{N} \implies \theta \approx 2/\sqrt{N}$.
2. **Grover Step as Rotation:**
   The Grover operator $G = D \cdot O$ is a composition of two reflections in the 2D plane spanned by $\{|w\rangle, |w^\perp\rangle\}$. The composition of two reflections whose axes form an angle $\theta/2$ is an exact rotation by angle $\theta = 2\arcsin(1/\sqrt{N})$.
3. **Reaching the Target:**
   The initial state starts at angle $\theta/2$ from $|w^\perp\rangle$. We wish to rotate the state to angle $\pi/2$ (aligned with target $|w\rangle$).
   After $R$ iterations, the total angle is:
   $$(2R + 1)\frac{\theta}{2} = \frac{\pi}{2} \implies 2R + 1 = \frac{\pi}{\theta} \approx \frac{\pi}{2/\sqrt{N}} = \frac{\pi}{2}\sqrt{N} \implies R \approx \frac{\pi}{4}\sqrt{N}$$
</details>

---

## Module S: Computational Physics & Celestial Mechanics

### Question 26: Symplectic Integrators vs. Explicit Euler (Module S)
Why does standard Explicit Euler integration cause planetary orbits in N-body simulations to rapidly spiral outward, while the Leapfrog / Velocity-Verlet integrator preserves orbits indefinitely?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Euler Non-Conservation of Energy:**
   Explicit Euler updates $x_{t+1} = x_t + v_t \Delta t$ and $v_{t+1} = v_t + a(x_t)\Delta t$. For a harmonic oscillator $\dot{x}=v, \dot{v}=-\omega^2 x$, the update matrix has determinant $\det(T) = 1 + \omega^2 \Delta t^2 > 1$. This artificially injects energy at every step, inflating the phase-space volume and causing orbits to spiral outward exponentially.
2. **Symplectic Structure & Shadow Hamiltonian:**
   Hamiltonian mechanics requires time evolution to be a canonical transformation that preserves phase-space differential 2-forms $dx \wedge dp$ (Liouville's theorem). Leapfrog / Velocity-Verlet is a **symplectic integrator** (exact $\det(J) = 1$). While it does not conserve the exact system Hamiltonian $H$, it *exactly* conserves a perturbed "shadow Hamiltonian" $\tilde{H} = H + \mathcal{O}(\Delta t^2)$, bounding energy errors within a stable oscillating envelope without secular drift.
</details>

---

### Question 27: Solving Kepler's Transcendental Equation (Module S)
Given mean anomaly $M = 1.25$ radians and orbital eccentricity $e = 0.6$, state Kepler's equation and write the Newton-Raphson update step to solve for eccentric anomaly $E$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Kepler's Equation:**
   $$f(E) = E - e\sin(E) - M = 0$$
2. **Derivative:**
   $$f'(E) = 1 - e\cos(E)$$
3. **Newton-Raphson Iteration:**
   $$E_{k+1} = E_k - \frac{E_k - e\sin(E_k) - M}{1 - e\cos(E_k)}$$
   Because $e < 1$, $f'(E) = 1 - e\cos(E) \ge 1 - e > 0$ is strictly positive for all $E$, ensuring monotonic, rapid quadratic convergence without singular divisions. Starting with initial guess $E_0 = M = 1.25$, convergence to machine precision is achieved in 3-4 iterations.
</details>

---

## Module T: Computational Biology & Bioinformatics

### Question 28: Needleman-Wunsch vs. Smith-Waterman Recurrence (Module T)
Write the dynamic programming recurrences for Needleman-Wunsch (global alignment) and Smith-Waterman (local alignment), and explain how their boundary conditions and backtracking triggers differ.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Needleman-Wunsch (Global):**
   $$D(i, j) = \max \begin{cases} D(i-1, j-1) + S(x_i, y_j) \\ D(i-1, j) + d \\ D(i, j-1) + d \end{cases}$$
   - *Boundary Conditions:* $D(i, 0) = i \cdot d$, $D(0, j) = j \cdot d$ (penalizes terminal gaps).
   - *Backtracking:* Starts strictly at cell $(N, M)$ and traces back to $(0, 0)$.
2. **Smith-Waterman (Local):**
   $$H(i, j) = \max \begin{cases} 0 \\ H(i-1, j-1) + S(x_i, y_j) \\ H(i-1, j) + d \\ H(i, j-1) + d \end{cases}$$
   - *Boundary Conditions:* $H(i, 0) = 0$, $H(0, j) = 0$ (allows local matches to start anywhere without penalty).
   - *Backtracking:* Starts at $\arg\max_{i, j} H(i, j)$ (the global maximum cell anywhere in the table) and terminates the moment $H(i, j) = 0$ is encountered.
</details>

---

## Module U: Signal Processing & Classical Control Theory

### Question 29: Cooley-Tukey Radix-2 FFT Symmetry & Butterfly Operation (Module U)
Derive how decimation-in-time splits an $N$-point DFT into two $N/2$-point DFTs, and write the butterfly update equations using twiddle factors $W_N^k = e^{-2\pi i k / N}$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Splitting into Even and Odd Indices:**
   $$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N} = \sum_{m=0}^{N/2-1} x_{2m} e^{-2\pi i k (2m)/N} + \sum_{m=0}^{N/2-1} x_{2m+1} e^{-2\pi i k (2m+1)/N}$$
   Using $e^{-2\pi i k (2m)/N} = e^{-2\pi i k m / (N/2)} = W_{N/2}^{km}$ and factoring out $W_N^k$:
   $$X_k = E_k + W_N^k O_k$$
   where $E_k$ is the $N/2$-point DFT of the even sequence and $O_k$ is the $N/2$-point DFT of the odd sequence.
2. **Butterfly Equations via Periodic Twiddle Symmetry:**
   Because $E_k$ and $O_k$ have period $N/2$, and $W_N^{k + N/2} = -W_N^k$:
   $$X_k = E_k + W_N^k O_k$$
   $$X_{k + N/2} = E_k - W_N^k O_k$$
   This evaluates both output values with a single complex multiplication, reducing total operations from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.
</details>

---

### Question 30: Derivation of the Linear Quadratic Regulator (LQR) Gain (Module U)
Given linear state dynamics $\dot{x} = Ax + Bu$ and quadratic cost $J = \int_0^\infty (x^T Q x + u^T R u) dt$, state the Continuous Algebraic Riccati Equation (CARE) and express the optimal state feedback control law $u^*(t)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Continuous Algebraic Riccati Equation (CARE):**
   For infinite-horizon regulation, the value function is quadratic $V(x) = x^T P x$. The Hamilton-Jacobi-Bellman (HJB) stationarity condition yields:
   $$A^T P + P A - P B R^{-1} B^T P + Q = 0$$
   where $P = P^T \succ 0$ is the unique positive-definite solution.
2. **Optimal Feedback Control Law:**
   $$u^*(t) = -K x(t), \quad K = R^{-1} B^T P$$
   Substituting $u^*$ into the dynamics yields the closed-loop system $\dot{x} = (A - BK)x$, where the eigenvalues of $(A - BK)$ are guaranteed to be strictly in the left-half complex plane (asymptotically stable).
</details>

---

## Module V: Numerical Linear Algebra & Scientific Computing

### Question 31: Modified Gram-Schmidt vs. Classical Gram-Schmidt (Module V)
Why does Classical Gram-Schmidt (CGS) suffer from catastrophic loss of orthogonality in floating-point arithmetic, and how does Modified Gram-Schmidt (MGS) restore numerical stability?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Classical Gram-Schmidt (CGS) Loss of Orthogonality:**
   In CGS, vector $a_k$ is projected simultaneously onto the initial normalized vectors: $v_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j$. If $a_k$ is nearly collinear with the span of $\{q_1, \dots, q_{k-1}\}$, rounding errors during the sum of subtractions lead to severe cancellation error, causing $\lVert Q^T Q - I \rVert$ to degrade to $\mathcal{O}(\epsilon_{\text{mach}} \kappa(A))$.
2. **Modified Gram-Schmidt (MGS) Sequential Updates:**
   MGS projects sequentially onto each orthogonalized vector: $v_k^{(1)} = a_k$, then $v_k^{(j+1)} = v_k^{(j)} - (q_j^T v_k^{(j)}) q_j$. At each sub-step, components along previously computed directions are purged immediately, mathematically equivalent to Householder reflections on row-augmented systems, preserving orthogonality to within $\mathcal{O}(\epsilon_{\text{mach}} \kappa(A))$.
</details>

---

## Module W: Electronic Design Automation & Engineering Optimization

### Question 32: BDD Canonicity and Shannon Decomposition (Module W)
State Shannon's expansion theorem for Boolean functions, and explain the reduction rules required to make a Binary Decision Diagram canonical (ROBDD).

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Shannon Decomposition:**
   Any Boolean function $f(x_1, x_2, \dots, x_n)$ can be factored about variable $x_i$:
   $$f = x_i \cdot f_{x_i} + x_i' \cdot f_{x_i'}$$
   where $f_{x_i} = f(x_1, \dots, x_i=1, \dots)$ is the positive cofactor, and $f_{x_i'} = f(x_1, \dots, x_i=0, \dots)$ is the negative cofactor.
2. **ROBDD Reduction Rules:**
   - **Ordered:** Variables appear in the exact same sequence along every path from root to leaf.
   - **Rule 1 (Eliminate Duplicate Terminals/Nodes):** If two nodes have the exact same variable and identical low and high children, merge them into a single node.
   - **Rule 2 (Eliminate Redundant Tests):** If a node's low edge and high edge point to the exact same child node, bypass the node completely.
   Under fixed variable ordering, Bryant's theorem proves that ROBDDs are strictly **canonical**: two Boolean functions are logically equivalent if and only if their ROBDD graphs are isomorphic, allowing equivalence checking in $\mathcal{O}(1)$ time using unique node tables.
</details>

---

### Question 33: DPLL Unit Propagation (Module W)
In Boolean Satisfiability (SAT), define a unit clause and trace how Unit Propagation simplifies the formula $\phi = (x_1) \wedge (\neg x_1 \vee x_2) \wedge (\neg x_2 \vee \neg x_3)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Unit Clause Definition:**
   A unit clause is a clause containing exactly one unassigned literal. For the formula to be satisfied, that literal MUST be assigned the value that makes it True.
2. **Trace of Unit Propagation on $\phi$:**
   - Clause $(x_1)$ is unit $\implies$ Must assign $x_1 = \text{True}$.
   - Under $x_1 = \text{True}$, the literal $\neg x_1$ in the second clause becomes False, reducing $(\neg x_1 \vee x_2)$ to the unit clause $(x_2)$.
   - Clause $(x_2)$ is unit $\implies$ Must assign $x_2 = \text{True}$.
   - Under $x_2 = \text{True}$, the literal $\neg x_2$ in the third clause becomes False, reducing $(\neg x_2 \vee \neg x_3)$ to the unit clause $(\neg x_3)$.
   - Clause $(\neg x_3)$ is unit $\implies$ Must assign $x_3 = \text{False}$.
   - All clauses are now satisfied. The unique satisfying assignment is $(x_1=\text{True}, x_2=\text{True}, x_3=\text{False})$ derived purely by deterministic propagation without any branching search!
</details>

---

## Module X: Robotics, Navigation & Classical Computer Vision

### Question 34: RRT vs. RRT* Asymptotic Optimality (Module X)
Why does standard RRT almost surely produce suboptimal paths even as the number of sampled nodes $n \to \infty$, and what two structural operations allow RRT* to achieve asymptotic optimality?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Standard RRT Suboptimality:**
   In standard RRT, once a node is connected to the tree, its parent is fixed forever. Even as sample count $n \to \infty$, vertices cannot rewire existing connections, trapping the solution path in its initial topological homotopy class and yielding jagged, non-optimal paths.
2. **RRT* Asymptotically Optimal Operations:**
   Within a shrinking neighborhood ball of radius $r_n = \gamma (\log n / n)^{1/d}$ around new sample $x_{\text{new}}$:
   - **Choose Best Parent:** Instead of connecting $x_{\text{new}}$ to its nearest neighbor, RRT* evaluates all vertices in the neighborhood and selects the parent that minimizes total cost from root: $\min_{x_{\text{near}}} \{ \text{Cost}(x_{\text{near}}) + \text{Dist}(x_{\text{near}}, x_{\text{new}}) \}$.
   - **Rewire Neighbor Tree:** RRT* checks if routing existing neighbors through $x_{\text{new}}$ lowers their accumulated cost: if $\text{Cost}(x_{\text{new}}) + \text{Dist}(x_{\text{new}}, x_{\text{near}}) < \text{Cost}(x_{\text{near}})$, the edge is rewired to make $x_{\text{new}}$ their new parent.
   Karaman and Frazzoli (2011) proved that these two operations guarantee the cost of the path converges to the optimal cost almost surely as $n \to \infty$.
</details>

---

### Question 35: Extended Kalman Filter (EKF) Linearization vs. UKF Sigma Points (Module X)
Contrast how the EKF and UKF handle non-linear measurement models $z = h(x) + v$. When does EKF diverge while UKF remains stable?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **EKF First-Order Taylor Expansion:**
   EKF linearizes $h(x)$ around the current estimate $\hat{x}$ using the Jacobian matrix: $h(x) \approx h(\hat{x}) + H(x - \hat{x})$, where $H = \left.\frac{\partial h}{\partial x}\right|_{\hat{x}}$. It maps covariance as $P_{zz} = H P H^T + R$. If $h(x)$ has strong curvature or high second derivatives $\nabla^2 h$, discarding higher-order Taylor terms causes severe bias in posterior mean and underestimation of covariance, leading to filter divergence.
2. **UKF Unscented Transform:**
   Instead of approximating non-linear functions, UKF uses the principle that *"it is easier to approximate a probability distribution than an arbitrary non-linear function"*. It generates $2n+1$ deterministic sigma points $\chi_i = \hat{x} \pm (\sqrt{(n+\lambda)P})_i$ that match the true mean and covariance up to the 3rd Taylor order. Each sigma point is passed through the *exact* non-linear function $\mathcal{Z}_i = h(\chi_i)$, and posterior statistics are computed by weighted summation. UKF remains stable and accurate without ever calculating analytical Jacobians.
</details>

---

## Module Y: Advanced Geometry, Topology & Measure Theory

### Question 36: Rodrigues' Formula from Matrix Exponential Series (Module Y)
Prove that for any skew-symmetric matrix $[\omega]_\times \in \mathfrak{so}(3)$ with $\theta = \lVert\omega\rVert > 0$, the infinite matrix exponential series $\exp([\omega]_\times) = \sum_{k=0}^\infty \frac{[\omega]_\times^k}{k!}$ reduces to the finite Rodrigues' formula.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Nilpotent Cyclic Powers of Skew-Symmetric Matrices:**
   Let unit vector $\mathbf{u} = \omega / \theta$, so $[\omega]_\times = \theta [\mathbf{u}]_\times$. Direct matrix multiplication reveals:
   $$[\mathbf{u}]_\times^3 = -[\mathbf{u}]_\times, \quad [\mathbf{u}]_\times^4 = -[\mathbf{u}]_\times^2, \quad [\mathbf{u}]_\times^5 = [\mathbf{u}]_\times$$
   In general, odd powers satisfy $[\mathbf{u}]_\times^{2k+1} = (-1)^k [\mathbf{u}]_\times$ and even powers satisfy $[\mathbf{u}]_\times^{2k+2} = (-1)^k [\mathbf{u}]_\times^2$ for $k \ge 1$.
2. **Grouping Taylor Series Terms:**
   $$\exp([\omega]_\times) = I + \sum_{k=0}^\infty \frac{\theta^{2k+1}}{(2k+1)!}[\mathbf{u}]_\times^{2k+1} + \sum_{k=1}^\infty \frac{\theta^{2k}}{(2k)!}[\mathbf{u}]_\times^{2k}$$
   $$= I + \left( \sum_{k=0}^\infty (-1)^k \frac{\theta^{2k+1}}{(2k+1)!} \right) [\mathbf{u}]_\times + \left( \sum_{k=1}^\infty (-1)^{k-1} \frac{\theta^{2k}}{(2k)!} \right) [\mathbf{u}]_\times^2$$
   Recognizing the standard Taylor expansions for $\sin\theta$ and $1 - \cos\theta$:
   $$= I + \sin\theta [\mathbf{u}]_\times + (1 - \cos\theta) [\mathbf{u}]_\times^2 = I + \frac{\sin\theta}{\theta}[\omega]_\times + \frac{1-\cos\theta}{\theta^2}[\omega]_\times^2$$
   This proves the finite closed form of the exponential map on $\text{SO}(3)$.
</details>

---

### Question 37: Fundamental Lemma of Homology over $\mathbb{Z}_2$ (Module Y)
Prove why $\partial_1 \circ \partial_2 = 0 \pmod 2$ for any triangle simplex $[v_0, v_1, v_2]$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Evaluating Boundary of Triangle $\partial_2$:**
   $$\partial_2([v_0, v_1, v_2]) = [v_1, v_2] + [v_0, v_2] + [v_0, v_1]$$
2. **Applying 1D Boundary $\partial_1$:**
   $$\partial_1(\partial_2([v_0, v_1, v_2])) = \partial_1([v_1, v_2]) + \partial_1([v_0, v_2]) + \partial_1([v_0, v_1])$$
   $$= ([v_2] + [v_1]) + ([v_2] + [v_0]) + ([v_1] + [v_0])$$
   $$= 2[v_0] + 2[v_1] + 2[v_2]$$
   Over field $\mathbb{Z}_2$, $2 \equiv 0$. Therefore:
   $$\partial_1 \circ \partial_2 = \mathbf{0}$$
   Every boundary of a boundary vanishes identically, guaranteeing that cycles mod boundaries form valid vector spaces ($H_k = \ker \partial_k / \operatorname{im} \partial_{k+1}$).
</details>

---

## Module Z: Geometry, Cryptography, Information Theory & Causal Discovery

### Question 38: Parabolic Lift & Empty Circumcircle Determinant (Module Z01)
Prove why testing whether point $D = (x_d, y_d)$ lies inside the circumcircle of counter-clockwise oriented planar triangle $A, B, C$ is equivalent to evaluating the 3D orientation determinant of their parabolic lifts $(x, y, x^2 + y^2)$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Geometric Parabolic Lift:**
   Map every planar point $P = (x, y)$ onto the 3D paraboloid $\Pi: z = x^2 + y^2$.
2. **Intersection of Plane with Paraboloid:**
   The circumcircle of $A, B, C$ in the plane corresponds to the planar cut of $\Pi$ through the lifted 3D points $A', B', C'$.
   The equation of the plane passing through $A', B', C'$ is $Ax + By - z + C_0 = 0$.
   Substituting $z = x^2 + y^2$ yields the exact circular boundary $(x - x_0)^2 + (y - y_0)^2 = R^2$.
3. **Orientation Invariant:**
   A point $D$ lies strictly *inside* the circle if and only if its lifted point $D'$ lies strictly *below* the secant plane spanned by $A', B', C'$.
   In 3D projective coordinates, the signed volume of the tetrahedron formed by $A', B', C', D'$ is given by the determinant:
   $$\det \begin{pmatrix} x_a - x_d & y_a - y_d & (x_a - x_d)^2 + (y_a - y_d)^2 \\ x_b - x_d & y_b - y_d & (x_b - x_d)^2 + (y_b - y_d)^2 \\ x_c - x_d & y_c - y_d & (x_c - x_d)^2 + (y_c - y_d)^2 \end{pmatrix} > 0$$
   This reduces non-linear circular queries to an exact $3 \times 3$ polynomial orientation test.
</details>

---

### Question 39: Special Soundness & Nonce Reuse Vulnerability in Schnorr ZKPs (Module Z02 / Z03)
Derive how an adversary extracts the exact private key $x$ if a prover reuses the same ephemeral commitment nonce $r$ across two distinct challenges $c_1 \ne c_2$ in the Schnorr identification protocol or ECDSA.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Linear Equation System from Transcripts:**
   When the ephemeral randomness $r$ is reused, the commitment $t = g^r \pmod p$ is identical.
   The prover emits responses:
   $$s_1 = r + c_1 x \pmod q$$
   $$s_2 = r + c_2 x \pmod q$$
2. **Eliminating Ephemeral Nonce $r$:**
   Subtracting the two congruence equations cancels out $r$:
   $$s_1 - s_2 = (c_1 - c_2)x \pmod q$$
3. **Inverting Difference over Prime Field:**
   Because $c_1 \ne c_2$ and $q$ is prime, $(c_1 - c_2)$ has a unique modular inverse $\pmod q$:
   $$x \equiv (s_1 - s_2)(c_1 - c_2)^{-1} \pmod q$$
   This proves **Special Soundness** (the existence of an efficient knowledge extractor). In ECDSA, the identical vulnerability famously compromised the Sony PlayStation 3 master signing key: when nonce $k$ is reused, $k = (e_1 - e_2)(s_1 - s_2)^{-1} \pmod n$, from which private key $d$ is derived trivially in $\mathcal{O}(1)$ operations.
</details>

---

### Question 40: Berkson's Bias & Unshielded Colliders in Causal Discovery (Module Z05)
Explain why conditioning on a common effect (collider $X \rightarrow Z \leftarrow Y$) induces spurious association between marginally independent causes ($X \not\perp\!\!\!\perp Y \mid Z$), and how the PC algorithm uses this property to orient causal arrows.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. **Marginal Independence vs. Conditional Dependence:**
   Let $X$ and $Y$ be two independent coin flips, and let $Z = X \oplus Y$ or $Z = X + Y$.
   Marginally, knowing $X$ provides zero information about $Y$ ($\mathbb{P}(Y \mid X) = \mathbb{P}(Y)$).
   However, conditioned on $Z$ (e.g. knowing $X + Y = 1$), observing that $X = 1$ forces $Y = 0$. Hence, conditioning on $Z$ creates an active non-causal path between $X$ and $Y$:
   $$X \perp\!\!\!\perp Y \quad \text{but} \quad X \not\perp\!\!\!\perp Y \mid Z$$
2. **Collider Detection in the PC Algorithm:**
   For any unshielded triple $X - Z - Y$ where $X$ and $Y$ are non-adjacent:
   During Phase 1, the algorithm discovered a conditioning set $\text{SepSet}(X, Y)$ that rendered $X$ and $Y$ independent ($X \perp\!\!\!\perp Y \mid \text{SepSet}(X, Y)$).
   - If $Z$ were a mediator ($X \rightarrow Z \rightarrow Y$) or a common cause ($X \leftarrow Z \rightarrow Y$), conditioning on $Z$ would *block* the path, meaning $Z$ *must* be in $\text{SepSet}(X, Y)$.
   - Therefore, if $Z \notin \text{SepSet}(X, Y)$, $Z$ cannot be a mediator or confounder; it **must be a collider**. The algorithm uniquely orients the arrows inward:
     $$X \longrightarrow Z \longleftarrow Y$$
   This enables the discovery of genuine causal directionality purely from observational probability distributions without running randomized controlled experiments.
</details>

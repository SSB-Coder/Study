# Deep Dive: Bias-Variance Decomposition & Resampling

## 1. The Beginner's Analogy: The Archer at the Target
Imagine an archer firing arrows at a target bullseye:
- **Bias** is the archer's systematic aim error. If their sight is misaligned to the top-right, all their arrows will cluster in the top-right corner, missing the bullseye no matter how steadily they hold their breath. High bias means the model is fundamentally too rigid to hit the truth (underfitting).
- **Variance** is the archer's shakiness. If their hands are trembling violently, their shots will scatter wildly all over the board, even if their sight is aimed directly at the center. High variance means the model changes wildly depending on which random training shots it practiced on (overfitting).
- **Resampling (Cross-Validation / Bootstrap)** is having the archer shoot across multiple days, wind conditions, and target distances to measure both their sight misalignment and hand tremor accurately before sending them to the Olympic finals.

---

## 2. The Deep Architecture/Algorithm Breakdown
Generalization error measures how well an algorithm predicts on unseen instances drawn from the true underlying distribution $P(X, Y)$.
1. **Bias-Variance Tradeoff:** Any regression estimator's expected squared test error decomposes cleanly into three components: squared bias, variance, and irreducible noise $\sigma^2$.
2. **K-Fold Cross-Validation:** Partitions training indices $\{1, \dots, m\}$ into $K$ disjoint subsets. Each fold serves as a validation set for a model trained on the remaining $K-1$ folds.
3. **Leave-One-Out Cross-Validation (LOOCV):** Extreme case where $K=m$. It yields an almost unbiased estimate of generalization error, but has high computational cost and high variance for unstable estimators.
4. **The Non-Parametric Bootstrap:** Resamples $m$ instances *with replacement* from the empirical distribution $\hat{F}_m$. Approximately $1 - \frac{1}{e} \approx 63.2\%$ of samples are selected; the remaining $36.8\%$ form the out-of-bag (OOB) set.

---

## 3. The Hardcore Mathematics

### 3.1 Mathematical Decomposition of Mean Squared Error
Assume the true data-generating process is $y = f(x) + \epsilon$, where $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$. Let $\hat{f}(x; \mathcal{D})$ be the estimator trained on dataset $\mathcal{D}$.
The expected prediction error at test point $x$:
$$\mathbb{E}_{\mathcal{D}, \epsilon}\left[ (y - \hat{f}(x))^2 \right] = \mathbb{E}\left[ (f(x) + \epsilon - \hat{f}(x))^2 \right]$$
Adding and subtracting $\mathbb{E}[\hat{f}(x)]$:
$$= \mathbb{E}\left[ \left( (f(x) - \mathbb{E}[\hat{f}(x)]) + (\mathbb{E}[\hat{f}(x)] - \hat{f}(x)) + \epsilon \right)^2 \right]$$
Cross-terms vanish because $\epsilon$ is independent of $\mathcal{D}$ and $\mathbb{E}[\mathbb{E}[\hat{f}(x)] - \hat{f}(x)] = 0$:
$$= \underbrace{(f(x) - \mathbb{E}[\hat{f}(x)])^2}_{\text{Bias}^2(\hat{f}(x))} + \underbrace{\mathbb{E}\left[ (\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2 \right]}_{\text{Variance}(\hat{f}(x))} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

### 3.2 Bootstrap Asymptotics
Given sample $\mathcal{D}_m = \{x_1, \dots, x_m\}$, each bootstrap sample $\mathcal{D}^*$ draws $m$ times with replacement.
The probability that observation $x_i$ is omitted in a single draw is $1 - \frac{1}{m}$.
For $m$ independent draws:
$$P(x_i \notin \mathcal{D}^*) = \left(1 - \frac{1}{m}\right)^m \xrightarrow{m \rightarrow \infty} e^{-1} \approx 0.3679$$
Thus, an average of $36.8\%$ of data points are Out-Of-Bag (OOB), serving as an internal validation set.

---

## 4. The Algorithm Step-by-Step
1. **K-Fold Splitting:** Shuffle indices, compute fold size $s = \lfloor m/K \rfloor$, and partition indices into $K$ contiguous slices.
2. **Train/Val Loop:** For each fold $k \in \{1, \dots, K\}$, assign fold $k$ as validation and remainder as training. Fit model, record validation metric.
3. **Bootstrap Resampling:** Draw indices $idx = \text{randint}(0, m, \text{size}=m)$. Train model on $\mathcal{D}[idx]$, evaluate on out-of-bag indices.
4. **Bias-Variance Simulation:** Train $B$ models on independent bootstrap datasets $\mathcal{D}_b$, evaluate on fixed test set $X_{test}$, compute average prediction $\bar{y}(x)$, variance $\frac{1}{B}\sum (\hat{y}_b - \bar{y})^2$, and bias $(\bar{y} - y_{true})^2$.

---

## 5. Limitations / When It Breaks
- **Data Leakage:** If data normalization (e.g. centering/scaling) is applied *before* K-fold splitting, information leaks from validation folds to training folds, artificially deflating error estimates.
- **Correlated Time Series:** Standard K-fold destroys temporal dependencies. Resampling time series requires block bootstrap or forward-chaining expansion.
- **Historical Fix:** Purged group time-series cross-validation and stratified sampling for imbalanced classes.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md`. Modern deep learning models operate in the overparameterized double-descent regime where classical bias-variance limits are overcome by implicit regularization in SGD.

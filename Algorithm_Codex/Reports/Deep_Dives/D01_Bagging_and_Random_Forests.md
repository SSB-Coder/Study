# Deep Dive: Bagging & Random Forests (Bootstrap Aggregation & OOB Error)

## 1. The Beginner's Analogy: The Jury vs. The Lone Judge
Imagine a complex legal trial deciding a difficult verdict:
- A single brilliant judge might have subtle personal biases, blind spots, or bad days, rendering an erratic verdict (high variance).
- Instead of relying on one judge, the judicial system convenes a **Jury of 12 Citizens**:
  - Each juror is selected from a random demographic cross-section (Bootstrap sampling).
  - Each juror is instructed to focus on a different set of physical exhibits (Random Feature Subsampling).
  - While individual jurors might make mistakes, their individual errors cancel out in the deliberation room.
- By taking a **majority vote across all 12 jurors**, the collective verdict is far more stable, fair, and reliable than any single juror could deliver.

---

## 2. The Deep Architecture/Algorithm Breakdown
Random Forests combine two core variance-reduction techniques:
1. **Bagging (Bootstrap Aggregating):** Trains $M$ independent base estimators on $M$ bootstrap samples $\mathcal{D}_1^*, \dots, \mathcal{D}_M^*$ drawn with replacement.
2. **Random Subspace Method (Feature Subsampling):** At every split in every tree, only a random subset of features (typically $m_{try} = \sqrt{d}$) is considered. This decorrelates the trees.
3. **Variance Reduction Invariant:** The average of $M$ identical random variables with variance $\sigma^2$ and pairwise correlation $\rho$ has variance:
   $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
   As $M \rightarrow \infty$, variance drops to $\rho \sigma^2$. Feature subsampling reduces correlation $\rho$, driving total variance down.
4. **Out-of-Bag (OOB) Evaluation:** Evaluates each training instance $x_i$ using *only* the subset of trees that did not include $x_i$ in their bootstrap sample.

---

## 3. The Hardcore Mathematics

### 3.1 Mathematical Bias-Variance of Tree Ensembles
Let individual tree estimators $T(x; \Theta_m)$ have variance $\sigma^2(x)$ and pairwise correlation $\rho(x) = \text{Corr}(T(x; \Theta_i), T(x; \Theta_j))$.
The ensemble prediction:
$$F(x) = \frac{1}{M} \sum_{m=1}^M T(x; \Theta_m)$$
Its variance:
$$\text{Var}(F(x)) = \frac{1}{M^2} \sum_{m=1}^M \text{Var}(T_m) + \frac{1}{M^2} \sum_{i \neq j} \text{Cov}(T_i, T_j) = \frac{\sigma^2}{M} + \frac{M(M-1)}{M^2} \rho \sigma^2 = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
Notice:
- As $M$ grows, $\frac{1-\rho}{M} \rightarrow 0$.
- The irreducible floor is $\rho \sigma^2$.
- The only way to lower the variance floor further is to force trees to be less correlated (lower $\rho$) via random feature subsampling!

### 3.2 OOB Error Estimation
For training sample $(x_i, y_i)$:
Let $S_i = \{m \in \{1, \dots, M\} \mid i \notin \text{bootstrap}_m\}$ be the set of trees for which $x_i$ was Out-of-Bag.
The OOB ensemble prediction:
$$\hat{y}_i^{\text{OOB}} = \arg\max_k \sum_{m \in S_i} \mathbb{I}(T_m(x_i) = k)$$
The OOB classification error:
$$\text{Err}_{\text{OOB}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i^{\text{OOB}} \neq y_i)$$

---

## 4. The Algorithm Step-by-Step
1. **Ensemble Initialization:** Choose number of trees $M$, max depth, and feature subset size $m_{try} = \lfloor \sqrt{d} \rfloor$.
2. **For each tree $m = 1, \dots, M$:**
   - Draw bootstrap sample $\mathcal{D}^*_m$ of size $N$ with replacement.
   - Record Out-of-Bag indices.
   - Fit decision tree where each node considers a random subset of $m_{try}$ features.
3. **OOB Scoring:** For each instance $x_i$, aggregate predictions from all trees where $x_i$ was OOB.
4. **Inference:** For test query $x$, predict majority vote across all $M$ trees.

---

## 5. Limitations / When It Breaks
- **Extrapolation Inability:** Tree ensembles cannot extrapolate outside the bounding box of training features (predictions are constant outside observed bounds).
- **High-Dimensional Sparse Data:** For sparse text bag-of-words ($d = 50,000$), random feature subsampling ($\sqrt{d} \approx 223$) frequently picks all-zero features at splits.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/36_Test_Time_Scaling_and_PRMs.md`. Self-Consistency sampling in modern reasoning LLMs (generating multiple reasoning chains and taking majority vote) is the exact cognitive analogue of bootstrap aggregation.

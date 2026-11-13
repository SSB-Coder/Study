# Deep Dive: Probability Calibration (Platt Scaling & Isotonic Regression)

## 1. The Beginner's Analogy: The Overconfident Meteorologist
Imagine a weather forecaster who announces every day: *"There is a 99% chance of rain today!"*
- If you check the weather history over 100 days when the forecaster said "99% chance of rain", it actually only rained on 65 of those days!
- The forecaster's ranking is correct (they know rainy days from sunny days), but their probabilities are wildly uncalibrated.
- **Probability Calibration** is a post-processing translator: whenever the forecaster says "99%", the calibration curve translates it to the true historical empirical rate of "65%".
- **Platt Scaling** fits a smooth logistic S-curve.
- **Isotonic Regression** fits a step-like non-decreasing staircase without assuming any predefined mathematical curve.

---

## 2. The Deep Architecture/Algorithm Breakdown
Many classifiers (SVMs, naive Bayes, deep neural networks with temperature drift) produce decision scores $f(x)$ that correlate with class rank but do not represent empirical probabilities:
1. **Calibration Definition:** A classifier is calibrated if:
   $$P(Y = 1 \mid P(Y=1 \mid X) = p) = p \quad \forall p \in [0, 1]$$
2. **Reliability Diagrams & Brier Score:** Brier score evaluates calibration via mean squared probability error: $\text{BS} = \frac{1}{m} \sum (\hat{p}_i - y_i)^2$.
3. **Platt Scaling (Sigmoid Calibration):** Fits a scalar univariate logistic regression on uncalibrated margins:
   $$P(Y=1 \mid f(x)) = \frac{1}{1 + \exp(A f(x) + B)}$$
4. **Isotonic Regression (PAVA):** Non-parametric monotonic regression solving:
   $$\min_{\hat{p}} \sum_{i=1}^m (\hat{p}_i - y_i)^2 \quad \text{s.t.} \quad \hat{p}_1 \le \hat{p}_2 \le \dots \le \hat{p}_m$$
   Solved in $O(m)$ time via the **Pool Adjacent Violators Algorithm (PAVA)**.

---

## 3. The Hardcore Mathematics

### 3.1 Platt Scaling Regularized Objective
To avoid overfitting on small calibration sets, target labels are smoothed:
$$t_i = \begin{cases} \frac{N_+ + 1}{N_+ + 2} & \text{if } y_i = +1 \\ \frac{1}{N_- + 2} & \text{if } y_i = -1 \end{cases}$$
Where $N_+$ and $N_-$ are positive and negative sample counts.
The objective minimizes cross-entropy with respect to scalars $A, B$:
$$\min_{A, B} -\sum_{i=1}^m \left[ t_i \ln p_i + (1 - t_i) \ln(1 - p_i) \right], \quad p_i = \frac{1}{1 + \exp(A f_i + B)}$$

### 3.2 Pool Adjacent Violators Algorithm (PAVA)
Let data be sorted by prediction score: $f_1 \le f_2 \le \dots \le f_m$.
1. Initialize each observation in its own group $G_i = \{i\}$ with value $v_i = y_i$ and weight $w_i = 1$.
2. Iterate through groups: if adjacent groups violate monotonicity ($v_i > v_{i+1}$):
   - Pool the two groups into a single combined group:
     $$v_{\text{pooled}} = \frac{w_i v_i + w_{i+1} v_{i+1}}{w_i + w_{i+1}}, \quad w_{\text{pooled}} = w_i + w_{i+1}$$
   - Backtrack: check if the new pooled value violates monotonicity with the preceding group. Continue pooling backward until monotonicity is restored.

---

## 4. The Algorithm Step-by-Step
1. **Calibration Data Split:** Train the base classifier on train set; extract raw decision margins $f(X_{\text{cal}})$.
2. **Platt Scaling Path:** Fit univariate logistic regression parameters $(A, B)$ on $(f, y_{\text{cal}})$ via gradient descent.
3. **Isotonic Path:** Sort pairs by margin $f_i$, execute PAVA pooling loop, and store step function knots.
4. **Inference:** Pass raw classifier test margins through fitted calibrator to obtain calibrated probabilities.

---

## 5. Limitations / When It Breaks
- **Non-Monotonic Relationships:** If classifier score is not monotonically related to class probability, isotonic regression fails.
- **Overfitting in Small Datasets:** Isotonic regression is non-parametric and can overfit when calibration sample $m < 1000$. Platt scaling is preferred for small sample sizes.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/28_The_Emergence_of_Intelligence_and_Evaluations.md`. Temperature scaling in modern LLM generation is a one-parameter variant of Platt scaling used to calibrate output logits against human confidence benchmarks.

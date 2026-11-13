# Deep Dive: Gradient Boosting Machines (GBM & Residual Fitting)

## 1. The Beginner's Analogy: The Sculptor and the Chisels
Imagine carving a lifelike marble statue of a lion:
- **Tree 1 (The Sledgehammer):** The first rough sculptor knocks off massive chunks of marble, creating a crude, blocky outline of a lion. It is far from perfect, leaving rough edges and errors everywhere.
- **Tree 2 (The Medium Chisel):** The second sculptor doesn't start from a new rock. They look *only at the mistakes* (the remaining excess marble and hollows) left behind by the first sculptor, chipping away at the residual errors.
- **Tree 3 (The Fine Rasp):** The third sculptor focuses exclusively on the tiny remaining flaws left by Tree 2.
- **Tree $M$ (The Polishing Sandpaper):** After 100 sculptors each shave off a tiny fraction ($\eta = 0.1$) of the remaining errors, the final statue emerges with flawless, lifelike precision.

---

## 2. The Deep Architecture/Algorithm Breakdown
Gradient Boosting builds an additive model in a forward stage-wise manner by fitting new weak base learners to the **negative gradient (pseudo-residuals)** of the loss function:
1. **Functional Gradient Descent:** Instead of optimizing parameters in parameter space $\mathbb{R}^d$, Gradient Boosting performs gradient descent in **function space** $L^2$.
2. **Pseudo-Residuals ($r_{im}$):** The negative gradient of the loss $\mathcal{L}(y_i, F(x_i))$ w.r.t current model prediction $F(x_i)$:
   $$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}$$
3. **Base Learners ($h_m(x)$):** Regression trees fit directly to the pseudo-residuals $r_{im}$.
4. **Shrinkage (Learning Rate $\eta$):** Scales the contribution of each new tree by $\eta \in (0, 1]$, dramatically reducing overfitting.
5. **Loss Flexibility:** Works with any differentiable loss function: Mean Squared Error, Huber loss, Binomial Cross-Entropy, Poisson deviance.

---

## 3. The Hardcore Mathematics

### 3.1 Functional Taylor Series Derivation
Let the ensemble prediction after $m-1$ stages be $F_{m-1}(x)$. We wish to find tree $h_m(x)$ minimizing:
$$\min_{h_m} \sum_{i=1}^N \mathcal{L}(y_i, F_{m-1}(x_i) + h_m(x_i))$$
Taking the first-order Taylor expansion around $F_{m-1}(x_i)$:
$$\mathcal{L}(y_i, F_{m-1}(x_i) + h_m(x_i)) \approx \mathcal{L}(y_i, F_{m-1}(x_i)) + \left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F_{m-1}} h_m(x_i)$$
To minimize this linear approximation subject to $\|h_m\| = \text{const}$, $h_m(x_i)$ must point in the exact opposite direction of the gradient:
$$h_m(x_i) \propto -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F_{m-1}} = r_{im}$$

### 3.2 Mean Squared Error (MSE) Residuals
$$\mathcal{L}(y, F) = \frac{1}{2}(y - F)^2 \implies r_{im} = - (F(x_i) - y_i) = y_i - F_{m-1}(x_i)$$
For MSE, pseudo-residuals are the exact raw arithmetic residuals!

### 3.3 Binary Log-Loss Pseudo-Residuals
For $y_i \in \{0, 1\}$ and log-odds $F(x) = \ln\frac{p}{1-p}$:
$$\mathcal{L}(y, F) = -[y \ln \sigma(F) + (1-y)\ln(1 - \sigma(F))]$$
Differentiating w.r.t $F$:
$$r_{im} = -(\sigma(F(x_i)) - y_i) = y_i - p_i$$
The pseudo-residual is the difference between ground-truth label and predicted probability!

---

## 4. The Algorithm Step-by-Step
1. **Initialize Base Model:** $F_0(x) = \arg\min_c \sum_{i=1}^N \mathcal{L}(y_i, c)$. (For MSE, $F_0 = \bar{y}$).
2. **For $m = 1$ to $M$:**
   - Compute pseudo-residuals $r_{im} = y_i - F_{m-1}(x_i)$ for all $i$.
   - Fit a regression tree $h_m(x)$ to predict targets $r_{im}$.
   - Update model with shrinkage: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$.
3. **Final Ensemble:** $F_M(x) = F_0(x) + \eta \sum_{m=1}^M h_m(x)$.

---

## 5. Limitations / When It Breaks
- **Sequential Latency:** Trees must be constructed sequentially; tree $m$ cannot be trained until tree $m-1$ has evaluated residuals across all training data.
- **Overfitting with Deep Trees:** Large trees ($>6$ depth) with high learning rates ($\eta > 0.3$) quickly memorize label noise.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/04_Limitations_and_Improvements.md`. Gradient boosting's residual fitting is mathematically equivalent to the residual stream connections ($x_{l+1} = x_l + \text{Block}(x_l)$) in modern Transformer blocks, where each layer refines the residual representation.

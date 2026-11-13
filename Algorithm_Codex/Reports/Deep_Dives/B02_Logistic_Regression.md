# Deep Dive: Logistic Regression (Binary & Softmax Multinomial)

## 1. The Beginner's Analogy: The Pressure Valve
Imagine a pressurized pipe with a safety valve:
- The pressure inside the pipe is a weighted sum of various physical factors: temperature, steam volume, pump speed ($z = w^T x + b$). This raw sum can range from negative infinity (vacuum) to positive infinity (massive pressure).
- **The Logistic Sigmoid** is a calibrated release valve. It takes whatever raw pressure $z$ exists and squashes it into a probability between $0.0$ and $1.0$ ($0\%$ to $100\%$ chance of explosion).
- If the output is $0.98$, the alarm rings: "Class 1: Danger!".
- In **Softmax Regression**, there are multiple pipes leading to different reservoirs (e.g. Cat, Dog, Bird). The valves open proportionally so the total volume escaping across all reservoirs sums strictly to $100\%$.

---

## 2. The Deep Architecture/Algorithm Breakdown
Logistic regression is a linear classification model mapping real-valued feature vectors to class probabilities:
1. **Linear Logits ($z$):** $z = x^T w + b$.
2. **Sigmoid Activation $\sigma(z)$:** Maps real numbers to $(0, 1)$ via $\sigma(z) = \frac{1}{1 + e^{-z}}$.
3. **Softmax Function:** Generalizes sigmoid to $K \ge 2$ mutually exclusive classes: $\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$.
4. **Cross-Entropy Loss (Log-Loss):** Derived from Maximum Likelihood Estimation of a Bernoulli or Categorical distribution. Cross-entropy is strictly convex for logistic regression, guaranteeing a unique global minimum.
5. **Batch Gradient Descent:** Updates parameter weights in the direction of the steepest decrease in cross-entropy error.

---

## 3. The Hardcore Mathematics

### 3.1 Binary Logistic Regression
Given $y_i \in \{0, 1\}$ and predicted probability $\hat{y}_i = \sigma(x_i^T w + b)$:
The likelihood function:
$$L(w, b) = \prod_{i=1}^m \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$
Negative Log-Likelihood (Binary Cross-Entropy):
$$\mathcal{L}(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right]$$
Derivative of Sigmoid:
$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$
Applying the chain rule for sample $i$:
$$\frac{\partial \mathcal{L}_i}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \sigma'(z_i) + \frac{1 - y_i}{1 - \hat{y}_i} \sigma'(z_i) = \hat{y}_i - y_i$$
Over the full batch $X \in \mathbb{R}^{m \times d}$ and $y \in \mathbb{R}^{m \times 1}$:
$$\nabla_w \mathcal{L} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$

### 3.2 Multinomial Softmax Regression
For $K$ classes with weight matrix $W \in \mathbb{R}^{d \times K}$ and bias $b \in \mathbb{R}^{1 \times K}$:
Logits $Z = XW + \mathbf{1}b \in \mathbb{R}^{m \times K}$.
Softmax probabilities:
$$P_{ik} = \frac{\exp(Z_{ik} - \max_j Z_{ij})}{\sum_{j=1}^K \exp(Z_{ij} - \max_j Z_{ij})}$$
*(Subtracting $\max_j Z_{ij}$ prevents numerical overflow without altering probabilities).*
Categorical Cross-Entropy:
$$\mathcal{L}(W, b) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K Y_{ik} \ln P_{ik}$$
Matrix gradient:
$$\nabla_W \mathcal{L} = \frac{1}{m} X^T (P - Y), \quad \nabla_b \mathcal{L} = \frac{1}{m} \mathbf{1}^T (P - Y)$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize Parameters:** Set $w = 0$ (or small Gaussian noise) and $b = 0$.
2. **Forward Pass:** Compute linear logits $z = Xw + b$, apply Sigmoid or Softmax.
3. **Compute Loss:** Evaluate Cross-Entropy loss with numerical clipping ($10^{-15}$).
4. **Backward Pass (Gradient):** Compute residual error matrix $E = \hat{Y} - Y$.
5. **Update Parameters:** Apply gradient descent step $w \leftarrow w - \alpha \cdot \frac{1}{m} X^T E$.
6. **Inference:** For binary, predict 1 if $\hat{y} \ge 0.5$; for multinomial, predict $\arg\max_k P_{ik}$.

---

## 5. Limitations / When It Breaks
- **Linear Decision Boundary:** Logistic regression fails on non-linearly separable manifolds (e.g. XOR) without non-linear feature kernels.
- **Complete Separation:** If a linear hyperplane perfectly separates the classes, MLE weights grow toward infinity ($w \rightarrow \infty$), blowing up gradients unless $L_2$ regularization is enforced.
- **Historical Fix:** Kernel SVMs, Multi-Layer Perceptrons, and $L_2$ regularized cross-entropy.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/13_Classic_Transformer_Dense.md`. Softmax regression is the exact final output layer of all autoregressive Transformer language models, generating probability distributions over vocabulary token logits.

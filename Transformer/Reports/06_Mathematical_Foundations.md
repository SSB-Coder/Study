# The Mathematical Foundations of Neural Networks

This report provides a deep dive into the hardcore mathematics required to build neural networks from absolute scratch, covering linear algebra matrices, calculus-based backpropagation, and weight optimization.

## 1. Linear Algebra: The Structure

Neural networks are fundamentally massive series of matrix multiplications. We use linear algebra because GPUs are highly optimized to perform thousands of matrix operations in parallel.

### Defining the Variables
*   $X$: The Input Matrix. Shape $(m, n_x)$ where $m$ is the batch size (number of examples) and $n_x$ is the number of features.
*   $W^{[l]}$: The Weight Matrix for layer $l$. Shape $(n_{l}, n_{l-1})$, where $n_l$ is the number of neurons in the current layer, and $n_{l-1}$ is the number of neurons in the previous layer.
*   $b^{[l]}$: The Bias Vector for layer $l$. Shape $(n_l, 1)$.
*   $Z^{[l]}$: The linear output of layer $l$ before activation.
*   $A^{[l]}$: The activated output of layer $l$.

### The Forward Pass Equations
For any given layer $l$:
1.  **Linear Transformation:** $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$
2.  **Activation:** $A^{[l]} = g^{[l]}(Z^{[l]})$ (where $g$ is an activation function like Sigmoid or ReLU).

*(Note: $A^{[0]}$ is the input $X$)*

---

## 2. Calculus: The Chain Rule & Backpropagation

How does the network learn what the weights should be? By calculating how much each weight contributed to the final error, and adjusting it in the opposite direction. This requires partial derivatives and the Chain Rule of Calculus.

### 2.1 The Loss Function
Let's use Binary Cross-Entropy (Log Loss) for a binary classification problem:
$$ \mathcal{L}(y, \hat{y}) = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right) $$
Where $y$ is the true label (0 or 1) and $\hat{y}$ is the predicted probability (which is $A^{[L]}$, the output of the final layer).

### 2.2 The Goal
We want to find the gradient of the Loss with respect to every weight in the network: $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ and $\frac{\partial \mathcal{L}}{\partial b^{[l]}}$.

### 2.3 The Chain Rule (Working Backwards)
To find the gradient for a weight in the final layer $L$, we must step backward through the computational graph:

$$ \frac{\partial \mathcal{L}}{\partial W^{[L]}} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial W^{[L]}} $$

Let's break this down for a final layer using a **Sigmoid** activation function:
1.  **Derivative of Loss w.r.t Activation:**
    $$ \frac{\partial \mathcal{L}}{\partial A^{[L]}} = - \frac{y}{A^{[L]}} + \frac{1 - y}{1 - A^{[L]}} $$
2.  **Derivative of Activation w.r.t Linear Output (Sigmoid Derivative):**
    $$ \frac{\partial A^{[L]}}{\partial Z^{[L]}} = A^{[L]} (1 - A^{[L]}) $$
3.  **Combining 1 & 2 (The error term $dZ$):**
    Remarkably, when you multiply the derivative of the Log Loss by the derivative of Sigmoid, the math simplifies perfectly to:
    $$ dZ^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = A^{[L]} - y $$
    *(This is why Sigmoid + Cross-Entropy is a perfect pairing!)*
4.  **Derivative of Linear Output w.r.t Weights:**
    $$ \frac{\partial Z^{[L]}}{\partial W^{[L]}} = A^{[L-1]} $$
5.  **Final Weight Gradient:**
    $$ dW^{[L]} = \frac{\partial \mathcal{L}}{\partial W^{[L]}} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T $$
    *(We divide by $m$ to get the average gradient across the entire batch).*

### 2.4 Propagating to Hidden Layers
To update weights in layer $l-1$, we must pass the error backward from layer $l$:
$$ dZ^{[l-1]} = (W^{[l]})^T dZ^{[l]} * g'^{[l-1]}(Z^{[l-1]}) $$
*(Where $*$ represents element-wise multiplication, and $g'$ is the derivative of the hidden layer's activation function, such as ReLU).*

Then we calculate the gradients for that hidden layer:
$$ dW^{[l-1]} = \frac{1}{m} dZ^{[l-1]} (A^{[l-2]})^T $$
$$ db^{[l-1]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l-1](i)} $$

---

## 3. Optimization Algorithms (Updating the Weights)

Once we have the gradients ($dW$ and $db$), how do we apply them?

### 3.1 Standard Gradient Descent
The simplest approach. Take the current weight, and subtract the gradient multiplied by a Learning Rate ($\alpha$).
$$ W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]} $$
$$ b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]} $$

### 3.2 Adam (Adaptive Moment Estimation)
Modern LLMs do not use standard gradient descent; they use Adam or its variants (AdamW). Adam keeps track of both the exponentially decaying average of past gradients (momentum) AND the squared gradients (velocity).

1.  **Momentum ($v$):** Helps the network push past local minimums.
    $$ v_{dW} = \beta_1 v_{dW} + (1 - \beta_1) dW $$
2.  **RMSprop ($s$):** Dampens oscillations in the gradient.
    $$ s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) dW^2 $$
3.  **Update Rule:**
    $$ W = W - \alpha \frac{v_{dW}}{\sqrt{s_{dW}} + \epsilon} $$

*(Where $\beta_1$, $\beta_2$ are decay rates, and $\epsilon$ is a tiny number to prevent division by zero).*

## Summary
Training an LLM involves running the Forward Pass equations across billions of parameters, calculating the loss against the target text, using the Chain Rule (Backpropagation) to calculate the $dW$ and $db$ gradients for every single parameter, and then using the Adam optimizer to adjust those billions of parameters by a tiny fraction. This is repeated trillions of times.

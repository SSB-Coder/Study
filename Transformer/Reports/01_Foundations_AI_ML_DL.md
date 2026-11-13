# Foundations of Artificial Intelligence, Machine Learning, and Deep Learning

This report covers the foundational theory, mathematics, and concepts behind how Artificial Intelligence (AI) models learn, from the atomic scale to basic neural networks.

## 1. Core Definitions

*   **Artificial Intelligence (AI):** The broad field of creating systems capable of performing tasks that typically require human intelligence.
*   **Machine Learning (ML):** A subset of AI that uses algorithms to parse data, learn from it, and make informed decisions. Instead of explicit programming, ML uses statistical methods to improve performance on a task over time.
*   **Deep Learning (DL):** A subset of ML based on Artificial Neural Networks (ANNs) with multiple layers (hence "deep"). DL models attempt to simulate the behavior of the human brain to learn from large amounts of data.
*   **Large Language Models (LLMs):** Massive DL models, typically based on the Transformer architecture, trained on vast amounts of text data to understand and generate human-like language.

## 2. The Learning Process: How Models "Know" the Answer

At a fundamental level, models do not "know" anything in a human sense. They perform highly complex pattern recognition through mathematical approximation.

### 2.1. Data Representation (Embeddings)

Computers cannot process raw text, images, or sound directly. Data must be converted into numerical vectors.
*   **Embeddings:** An embedding is a dense vector of real numbers that represents the meaning or features of a piece of data.
*   *Example:* The word "Apple" might be represented as an array of 512 numbers `[0.12, -0.45, 0.89, ...]`. In this multi-dimensional space, words with similar meanings (like "Apple" and "Orange") are located closer together mathematically than unrelated words (like "Apple" and "Car").

### 2.2. The Mathematical Foundation of a Neural Network

A basic neural network consists of layers of interconnected nodes (neurons).

#### 2.2.1. The Atomic Scale: A Single Neuron (Perceptron)

A single neuron takes inputs $X = [x_1, x_2, ..., x_n]$, multiplies them by weights $W = [w_1, w_2, ..., w_n]$, adds a bias $b$, and passes the sum through an activation function $f$.

**Equation:**
$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b = W \cdot X + b$$
$$a = f(z)$$

*   **Weights ($W$):** Represent the strength of the connection. They dictate how much influence an input has on the output. These are the values the model "learns" during training.
*   **Bias ($b$):** Allows the model to shift the activation function left or right, helping it fit the data better.
*   **Activation Function ($f$):** Introduces non-linearity. Without it, the network would just be a series of linear transformations (effectively a single linear regression). Common examples:
    *   **Sigmoid:** $f(z) = \frac{1}{1 + e^{-z}}$ (squashes values between 0 and 1)
    *   **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$ (widely used in deep networks)

#### 2.2.2. Forward Propagation

When data is fed into the network, it moves forward layer by layer. The output of layer $l$ becomes the input to layer $l+1$.
For layer $l$:
$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = f^{[l]}(Z^{[l]})$$

### 2.3. Training: Loss and Backpropagation

How do we find the correct weights and biases? Through training.

#### 2.3.1. The Loss Function

The model makes a prediction ($\hat{y}$), which is compared to the actual true target value ($y$). The difference is calculated using a Loss Function ($L$).
*   **Mean Squared Error (MSE) (for regression):** $L = \frac{1}{n} \sum (\hat{y} - y)^2$
*   **Cross-Entropy Loss (for classification):** $L = -\sum y \log(\hat{y})$

#### 2.3.2. Gradient Descent and Backpropagation

The goal of training is to minimize the Loss. We do this by adjusting the weights using **Gradient Descent**.
We calculate the gradient (derivative) of the loss function with respect to every weight in the network. The gradient tells us the direction to move the weight to reduce the loss.

**Weight Update Rule:**
$$W_{new} = W_{old} - \alpha \cdot \frac{\partial L}{\partial W}$$
Where $\alpha$ is the **learning rate** (how big of a step we take).

**Backpropagation** is the algorithm used to efficiently calculate these gradients using the Chain Rule of calculus from the output layer backwards to the input layer.

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}} \cdot \dots \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$$

## 3. Dimensionality and Complexity

As networks grow, the dimensions of the matrices (tensors) increase.
*   A simple feedforward network might have weight matrices of size $100 \times 100$.
*   Modern LLMs have billions of parameters. If a layer has 4096 neurons and connects to another 4096 neurons, the weight matrix is $4096 \times 4096 \approx 16.7$ million weights *just for one layer*.

The model "knows" the answer because these billions of weights have been precisely tuned over trillions of training examples to approximate the incredibly complex, high-dimensional probability distribution of human language.

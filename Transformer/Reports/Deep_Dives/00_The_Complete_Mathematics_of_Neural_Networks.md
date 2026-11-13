# Deep Dive 00: The Complete Mathematics of Neural Networks

This document is the absolute foundational guide to the mathematics of AI. We will define every symbol, write out every equation, and walk through a **complete numerical example** of how a neural network calculates weights, biases, and learns from its mistakes.

---

## 1. The Universal Symbol Glossary

Before doing any math, you must understand the alphabet of Neural Networks.

### Parameters (The "Brain" of the AI)
*   $W$ (**Weight Matrix**): The strength of the connection between two neurons. If $W$ is high, the AI pays a lot of attention to that input.
*   $b$ (**Bias Vector**): A baseline number added to the result. It allows the neuron to shift its activation threshold left or right.

### The Forward Pass (Data flowing In $\rightarrow$ Out)
*   $x$ (**Input Vector**): The raw data entering the network (e.g., pixel values, word embeddings).
*   $z$ (**Pre-Activation**): The raw mathematical result of multiplying weights by inputs and adding bias. ($z = Wx + b$).
*   $\sigma$ or $f$ (**Activation Function**): A mathematical curve applied to $z$ to introduce non-linearity (e.g., ReLU, Sigmoid).
*   $a$ (**Post-Activation**): The final output of the neuron after passing through the activation function. ($a = \sigma(z)$).
*   $\hat{y}$ (**Prediction**): The final guess made by the network (the $a$ of the very last layer).
*   $y$ (**Ground Truth**): The actual, correct answer we want the network to guess.

### The Backward Pass (Learning from mistakes)
*   $\mathcal{L}$ (**Loss**): A single number measuring how horribly wrong the prediction $\hat{y}$ was compared to the truth $y$.
*   $\partial$ (**Partial Derivative**): Measures how sensitive the Loss is to a tiny change in a specific variable (e.g., $\frac{\partial \mathcal{L}}{\partial W}$).
*   $\nabla$ (**Gradient**): A vector containing all the partial derivatives. It points in the direction that makes the Loss *increase* the fastest.
*   $\alpha$ (**Learning Rate**): A tiny number (e.g., $0.01$) that controls how big of a step the AI takes when updating its weights.

---

## 2. A Simple Neural Network: Full Numerical Example

Let's build a tiny neural network with 1 input neuron, 1 hidden neuron, and 1 output neuron. We want to teach it a simple rule: if the input is `2.0`, the output should be `1.0`.

### Step 2.1: The Setup (Initialization)
When a network is born, its parameters ($W$ and $b$) are completely random.
*   **Input ($x$):** `2.0`
*   **Target Truth ($y$):** `1.0`
*   **Weight 1 ($W_1$):** `0.5` *(Randomly chosen)*
*   **Bias 1 ($b_1$):** `0.1` *(Randomly chosen)*
*   **Weight 2 ($W_2$):** `-0.2` *(Randomly chosen)*
*   **Bias 2 ($b_2$):** `0.3` *(Randomly chosen)*

### Step 2.2: The Forward Pass (Making a Guess)
The data flows forward through the network. The formula for a neuron is: **$z = W \cdot x + b$**

**Layer 1 (Hidden Layer):**
1.  Calculate pre-activation: $z_1 = (W_1 \cdot x) + b_1 = (0.5 \cdot 2.0) + 0.1 = 1.1$
2.  Apply Activation Function (ReLU): $a_1 = \max(0, z_1) = \max(0, 1.1) = 1.1$

**Layer 2 (Output Layer):**
1.  Calculate pre-activation: $z_2 = (W_2 \cdot a_1) + b_2 = (-0.2 \cdot 1.1) + 0.3 = -0.22 + 0.3 = 0.08$
2.  Apply Activation Function (None/Linear for this example): $\hat{y} = z_2 = 0.08$

**The AI's Prediction:** $\hat{y} = 0.08$ (It is very wrong. It was supposed to guess $1.0$).

### Step 2.3: Calculating the Loss (Measuring Error)
We use Mean Squared Error (MSE) to measure how wrong the network is.
*   **Equation:** $\mathcal{L} = \frac{1}{2} (\hat{y} - y)^2$
*   **Calculation:** $\mathcal{L} = \frac{1}{2} (0.08 - 1.0)^2 = \frac{1}{2} (-0.92)^2 = \frac{1}{2} (0.8464) = 0.4232$

The Loss is `0.4232`. The AI's goal is to make this number `0`.

### Step 2.4: The Backward Pass (Backpropagation)
To fix the network, we must calculate the **Gradient**—how much the Loss changes if we wiggle each weight. We use the **Chain Rule** from Calculus.

**Finding the gradient for $W_2$:**
We want to find $\frac{\partial \mathcal{L}}{\partial W_2}$. The Chain Rule splits this into three parts:
$\frac{\partial \mathcal{L}}{\partial W_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2}$

Let's calculate each piece:
1.  $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ (Derivative of $\frac{1}{2}(\hat{y}-y)^2$): $= (\hat{y} - y) = (0.08 - 1.0) = -0.92$
2.  $\frac{\partial \hat{y}}{\partial z_2}$ (Derivative of linear activation $\hat{y}=z_2$): $= 1$
3.  $\frac{\partial z_2}{\partial W_2}$ (Derivative of $W_2 \cdot a_1 + b_2$ with respect to $W_2$): $= a_1 = 1.1$

Multiply them together:
$\frac{\partial \mathcal{L}}{\partial W_2} = -0.92 \cdot 1 \cdot 1.1 = -1.012$
*(This means if we increase $W_2$, the Loss will drop rapidly).*

**Finding the gradient for $b_2$:**
$\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2} = -0.92 \cdot 1 \cdot 1 = -0.92$

### Step 2.5: The Weight Update (Gradient Descent)
Now we actually change the weights to make the AI smarter.
*   **Equation:** $W_{\text{new}} = W_{\text{old}} - (\alpha \cdot \text{Gradient})$
*   Let's use a Learning Rate ($\alpha$) of `0.1`.

**Update $W_2$:**
$W_{2\text{\_new}} = -0.2 - (0.1 \cdot -1.012) = -0.2 + 0.1012 = -0.0988$

**Update $b_2$:**
$b_{2\text{\_new}} = 0.3 - (0.1 \cdot -0.92) = 0.3 + 0.092 = 0.392$

If we run the exact same input (`2.0`) through the network again with these new weights, the prediction $\hat{y}$ will be much closer to `1.0`. The AI has learned.

---

## 3. The Mathematics of Complex Matrices

In reality, we don't pass single numbers into neural networks. We pass massive grids of numbers (Matrices).

If our input $X$ is a batch of 3 images, and each image has 4 pixels, $X$ is a Matrix of shape `[3, 4]`.
If our hidden layer has 5 neurons, the Weight matrix $W$ has shape `[4, 5]`.

The Forward Pass equation $Z = X \cdot W + b$ uses **Matrix Multiplication**:
$$ \begin{bmatrix} x_{11} & x_{12} & x_{13} & x_{14} \\ x_{21} & x_{22} & x_{23} & x_{24} \\ x_{31} & x_{32} & x_{33} & x_{34} \end{bmatrix} \times \begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} & w_{15} \\ w_{21} & w_{22} & w_{23} & w_{24} & w_{25} \\ w_{31} & w_{32} & w_{33} & w_{34} & w_{35} \\ w_{41} & w_{42} & w_{43} & w_{44} & w_{45} \end{bmatrix} = \begin{bmatrix} z_{11} & \dots & z_{15} \\ \vdots & \ddots & \vdots \\ z_{31} & \dots & z_{35} \end{bmatrix} $$
The result $Z$ has shape `[3, 5]`. 

> [!TIP]
> **Matrix Matching Rule:** To multiply matrix A and matrix B, the inner dimensions must match. `[3, 4] x [4, 5]` works because the 4s match. The result is the outer dimensions: `[3, 5]`.

---

## 4. The Mathematics of the Transformer

The Transformer uses the exact same Matrix Multiplication principles, but applies them in a highly specific architecture called **Attention**.

### 4.1 Creating Q, K, and V
An LLM turns a sentence like "The cat sat" into an embedding matrix $X$ of shape `[Sequence_Length, Embed_Dim]`. (e.g., `[3, 512]`).

The Transformer creates three separate weight matrices: $W_Q$, $W_K$, $W_V$. It multiplies $X$ by these weights to create Queries, Keys, and Values.
*   $Q = X \cdot W_Q$ (What am I looking for?)
*   $K = X \cdot W_K$ (What do I contain?)
*   $V = X \cdot W_V$ (What is my actual underlying meaning?)

### 4.2 Scaled Dot-Product Attention (The Core Equation)
This is the equation that makes modern AI work:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V $$

**Step-by-Step Breakdown:**
1.  **$Q \cdot K^T$ (The Dot Product):** We multiply the Query matrix by the transposed Key matrix. If Word 1's Query vector closely matches Word 3's Key vector, the dot product will be a massive positive number. This creates a square "Attention Score" matrix of shape `[Seq_Len, Seq_Len]`.
2.  **$\frac{1}{\sqrt{d_k}}$ (The Scaling Factor):** If the embedding dimension is huge (e.g., 4096), dot products get massively large. Large numbers break the Softmax function (gradients vanish to zero). We divide the scores by the square root of the dimension to keep the variance stable at `1.0`.
3.  **$\text{softmax}(\dots)$:** This forces every row in the Attention Score matrix to sum to exactly `1.0`. Word 1 might now assign `0.8` attention to Word 3, `0.1` to Word 2, and `0.1` to itself. 
4.  **$\dots \cdot V$ (The Value Multiplication):** We multiply the percentage probabilities by the actual Value vectors. Word 1's new embedding becomes a blend containing 80% of Word 3's meaning, 10% of Word 2's meaning, and 10% of its own meaning.

### 4.3 Layer Normalization (LayerNorm)
Before passing the output to the next layer, the Transformer stabilizes the numbers using LayerNorm.
$$ \text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$
1.  Calculate the Mean ($\mu$) of the vector.
2.  Calculate the Variance ($\sigma^2$) of the vector.
3.  Subtract the mean and divide by the variance. This forces the vector to have a mean of `0` and a spread of `1`.
4.  Multiply by a learned weight $\gamma$ and add a learned bias $\beta$ so the network can shift the normalized data if it wants to.

### 4.4 The Final Output (Cross-Entropy)
At the very end of the network, the Transformer outputs a matrix of logits (raw scores) of shape `[Seq_Len, Vocab_Size]`.
It uses the **Softmax** function to turn these scores into percentages. 
The AI then uses the **Categorical Cross-Entropy Loss** to calculate how wrong it was compared to the actual next word in the training data:
$$ \mathcal{L} = -\sum_{c=1}^{\text{Vocab}} y_c \log(\hat{y}_c) $$
Since $y_c$ is `1` for the correct word and `0` for all other words, this beautifully simplifies to just:
$$ \mathcal{L} = -\log(\hat{y}_{\text{correct}}) $$
The network then runs the exact same Backpropagation algorithm we did in Section 2.4, sending gradients backward through the softmax, the LayerNorm, the Attention matrix, and finally updating $W_Q, W_K, W_V$ to be slightly smarter for the next token.

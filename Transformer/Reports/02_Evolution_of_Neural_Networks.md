# The Evolution of Neural Networks

This report details the progression of neural network architectures, explaining how early atomic-scale models evolved into complex structures capable of handling image and sequential data.

## 1. The Perceptron (The Atomic Unit)
Invented in 1957 by Frank Rosenblatt, the Perceptron is the simplest type of artificial neural network. It consists of a single layer of input nodes connected to an output node.

*   **Mechanism:** Takes inputs, multiplies them by weights, sums them, and passes the result through a step function (acting as an on/off switch).
*   **Limitation:** It can only solve **linearly separable** problems (e.g., AND/OR logic gates). It famously failed to solve the XOR problem, leading to the "AI Winter."
*   **Math:** $y = \text{step}(W \cdot X + b)$

## 2. Multi-Layer Perceptrons (MLPs) / Feedforward Neural Networks
To solve non-linear problems like XOR, researchers added "hidden layers" between the inputs and outputs.
*   **Mechanism:** Information flows in one direction (feedforward). The addition of non-linear activation functions (like Sigmoid or ReLU) allows the network to approximate any continuous mathematical function (Universal Approximation Theorem).
*   **Training:** This required the invention of **Backpropagation** (popularized in the 1980s) to effectively calculate how much each weight in a deep layer contributed to the final error.
*   **Pros:** Can learn complex non-linear relationships.
*   **Cons:** Treats all inputs independently. It loses spatial information (bad for images) and has no memory of past inputs (bad for text/time-series).

## 3. Convolutional Neural Networks (CNNs)
Developed specifically to handle grid-like data, such as images.

*   **Mechanism:** Instead of connecting every input pixel to every neuron (which explodes the number of weights), CNNs use **Convolutional Layers**. They slide a small mathematical matrix (a "filter" or "kernel") over the image.
*   **The Math of Convolution:** For a 2D image $I$ and a 2D filter $K$, the convolution operation is:
    $$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n)K(m, n)$$
*   **Key Features:**
    *   **Weight Sharing:** The same filter is used across the entire image, drastically reducing the number of parameters.
    *   **Translation Invariance:** A feature learned in the top-left corner can be recognized in the bottom-right corner.
    *   **Pooling Layers:** Compress the image dimensions (e.g., Max Pooling) to retain only the most important features, reducing computational load.
*   **Pros:** Excellent for Computer Vision (Object detection, facial recognition).
*   **Cons:** Fixed input size; struggles with sequential data where order matters.

## 4. Recurrent Neural Networks (RNNs)
Designed to handle sequential data like text, speech, or time-series data.

*   **Mechanism:** RNNs have loops. They process sequences step-by-step, maintaining an internal "hidden state" that acts as memory. The output at time step $t$ depends not just on the input at time $t$, but also on the hidden state from time $t-1$.
*   **The Math:**
    $$h_t = f_h(W_h \cdot h_{t-1} + W_x \cdot x_t + b_h)$$
    $$y_t = f_y(W_y \cdot h_t + b_y)$$
*   **Limitation (The Vanishing Gradient Problem):** During backpropagation through time (BPTT), gradients are multiplied repeatedly. If gradients are < 1, they shrink to zero exponentially fast. The RNN "forgets" long-term dependencies (e.g., words at the beginning of a long paragraph).

## 5. LSTMs (Long Short-Term Memory) and GRUs
To solve the vanishing gradient problem in standard RNNs, LSTMs were introduced in 1997.

*   **Mechanism:** They introduce a "cell state" (a conveyer belt running through the network) and "gates" (Forget, Input, Output gates) that optionally let information through.
*   **Gates:** These are tiny neural networks inside the LSTM cell (using Sigmoid functions to output values between 0 and 1) that learn what to keep and what to throw away from memory.
*   **Math (Forget Gate example):** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
*   **Pros:** Can remember information over long sequences. Excellent for translation, speech recognition (pre-2017).
*   **Cons:** Sequential nature means they *cannot be parallelized* well. You must compute step $t$ before computing step $t+1$. This makes them incredibly slow to train on massive datasets.

This limitation in parallelization is what drove the AI community to look for a completely new architecture, leading to the invention of the **Transformer**.

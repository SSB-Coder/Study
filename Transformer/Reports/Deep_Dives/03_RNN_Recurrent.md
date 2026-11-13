# Deep Dive: Recurrent Neural Networks (RNN)

## 1. The Beginner's Analogy: Reading a Book

Imagine trying to understand the sentence: *"I grew up in France... so I speak fluent ____"*. 

If you used a standard neural network (MLP), it would look at the word "fluent" completely in isolation. It has no memory. It wouldn't know what language to guess.

An RNN works exactly like a human reading a book. When you read the word "France", you store it in your short-term memory. As you read the rest of the sentence, you carry that memory with you. When you hit the blank space, you consult your current word ("fluent") AND your short-term memory ("France"), allowing you to confidently guess "French".

*   The Current Word = **The Input ($X_t$)**
*   Your Short-Term Memory = **The Hidden State ($h_t$)**
*   The process of carrying the memory to the next word = **The Recurrent Loop**

---

## 2. The Deep Architecture Breakdown

Unlike an MLP which maps one input to one output, an RNN operates over **Time Steps** ($t=1, 2, 3...$). It is designed for sequences (Text, Audio, Stock Prices).

At every single time step $t$, the RNN cell receives two things:
1.  **$x_t$**: The data at the current time step (e.g., the 5th word in a sentence).
2.  **$h_{t-1}$**: The "Hidden State" output by the RNN cell at the *previous* time step (e.g., the memory of words 1 through 4).

The RNN cell merges these two pieces of information, processes them, and spits out a new updated memory: **$h_t$**.
This new memory is then used to make a prediction ($y_t$), AND it is passed forward to time step $t+1$.

### Unrolling in Time
When we draw an RNN, it looks like a single box with a loop pointing back to itself. However, to train it, we mathematically "unroll" it. A 5-word sentence means we copy-paste the RNN box 5 times in a row, stringing the hidden state from box 1 to box 2, box 2 to box 3, and so on.

*Crucially, the weights inside the box are exactly the same across all time steps.* The network learns one set of rules applied repeatedly.

---

## 3. The Hardcore Mathematics

### 3.1 The Forward Pass Equations
Let $x_t$ be an input vector of size $(n_x, 1)$.
Let $h_t$ be a hidden state vector of size $(n_h, 1)$.

The RNN maintains three distinct Weight Matrices:
1.  $W_{hx}$: Weights applied to the incoming input. Shape $(n_h, n_x)$.
2.  $W_{hh}$: Weights applied to the previous hidden state (The Memory Weights). Shape $(n_h, n_h)$.
3.  $W_{yh}$: Weights applied to the hidden state to generate a prediction. Shape $(n_y, n_h)$.

**Step 1: Update the Hidden State (Memory)**
The network concatenates the input and the past memory, applies the weights, adds a bias, and crushes it through a $\tanh$ activation function to keep values bounded between -1 and 1.
$$ h_t = \tanh(W_{hh} h_{t-1} + W_{hx} x_t + b_h) $$

**Step 2: Generate the Output**
$$ \hat{y}_t = \text{softmax}(W_{yh} h_t + b_y) $$

### 3.2 BPTT: Backpropagation Through Time
Training an RNN is notoriously difficult because to find the gradient of the error at time step $t=10$, you have to apply the chain rule backward through time step 9, 8, 7... all the way to 1.

The gradient of the Loss with respect to the memory weight $W_{hh}$ requires summing up the gradients at every time step $k$ from $1$ to the current time $t$:
$$ \frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial h_t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \frac{\partial h_k}{\partial W_{hh}} $$

### 3.3 The Vanishing / Exploding Gradient Problem
Look closely at this part of the BPTT equation: $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$
This is a giant chain of multiplications.
*   If the weights in $W_{hh}$ are slightly greater than 1, multiplying them 50 times causes the gradient to explode to infinity (Exploding Gradient $\rightarrow$ NaN errors).
*   If the weights are slightly less than 1, multiplying them 50 times causes the gradient to shrink to 0.0000001. The network completely stops learning from the beginning of the sequence (Vanishing Gradient $\rightarrow$ Complete Amnesia for early words).

Because standard RNNs suffer from rapid amnesia, they are almost never used in modern AI. They were replaced by their upgraded cousins: **LSTMs and GRUs**.

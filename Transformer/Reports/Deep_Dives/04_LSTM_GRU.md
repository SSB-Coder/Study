# Deep Dive: Long Short-Term Memory (LSTM) & GRU

## 1. The Beginner's Analogy: The Conveyor Belt and the Bouncers

A standard RNN forgets things quickly (the vanishing gradient problem). Imagine someone reading a novel but they suffer from severe short-term memory loss; by chapter 5, they've forgotten the main character's name from chapter 1.

An **LSTM** fixes this by introducing a **Conveyor Belt** (the Cell State) that runs straight through the entire network from start to finish. Information can flow down this conveyor belt unchanged, preserving memories over huge distances.

But the network can't just keep everything. It needs to know what to add and what to throw away. It uses three "Bouncers" (called Gates) to manage the conveyor belt:
1.  **The Forget Bouncer:** Looks at the current word and the recent memory, and decides what old stuff on the conveyor belt is no longer relevant. *(e.g., The story just started a new chapter, time to forget the location of the old scene).*
2.  **The Input Bouncer:** Decides what *new* information from the current word is important enough to place *onto* the conveyor belt.
3.  **The Output Bouncer:** Looks at the conveyor belt and decides what part of it is actually relevant to the current prediction we need to make right now.

---

## 2. The Deep Architecture Breakdown

Inside an LSTM cell at time step $t$, we have the same inputs as an RNN ($x_t$ and $h_{t-1}$), but we add the all-important Cell State ($C_{t-1}$).

### The Three Gates (The Bouncers)
A "Gate" in a neural network is just a tiny, standard neural network layer that uses a **Sigmoid** activation function.
Sigmoid squashes any number between `0` and `1`.
*   If the gate outputs `0`, it means "completely block this / forget this".
*   If the gate outputs `1`, it means "let this entirely through / remember this".

By multiplying data by the output of these sigmoid gates, the LSTM tightly controls information flow.

---

## 3. The Hardcore Mathematics

Let's trace the math inside the LSTM step-by-step.
$x_t$: Input vector.
$h_{t-1}$: Previous short-term hidden state.
$C_{t-1}$: Previous long-term cell state.

### Step 1: The Forget Gate ($f_t$)
We concatenate $h_{t-1}$ and $x_t$, multiply by a weight matrix $W_f$, add a bias, and apply sigmoid.
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
*Result:* A vector of numbers between 0 and 1. We will multiply this directly against the old conveyor belt $C_{t-1}$.

### Step 2: The Input Gate ($i_t$) and Candidate Value ($\tilde{C}_t$)
We need to decide what new information to add.
First, the **Input Gate** decides *which values* we will update:
$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
Second, a **$\tanh$ layer** creates a vector of *candidate* new values (between -1 and 1) that could be added to the state:
$$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

### Step 3: Updating the Cell State ($C_t$)
Now we actually update the conveyor belt.
We multiply the old state $C_{t-1}$ by the forget gate $f_t$. Then we add the new candidate values, scaled by how much the input gate $i_t$ decided to let through:
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$
*(Crucial Note: This is where the magic happens. Because we use addition ($+$) to update $C_t$, gradients during backpropagation flow cleanly backwards across the addition without vanishing!)*

### Step 4: The Output Gate ($o_t$) and Next Hidden State ($h_t$)
Finally, we decide what our actual output for this step is, which also serves as the short-term memory $h_t$ for the next step.
The **Output Gate** decides what parts of the cell state to output:
$$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
We push the cell state through $\tanh$ (to push values between -1 and 1) and multiply it by the output gate:
$$ h_t = o_t * \tanh(C_t) $$

---

## 4. Gated Recurrent Units (GRU)

LSTMs are brilliant, but they are computationally expensive because they require 4 separate neural networks (weight matrices) inside every single cell.

In 2014, the **GRU** was introduced as a streamlined version of the LSTM.
*   **The Simplification:** It merges the Cell State and Hidden State into a single vector. It also merges the Forget and Input gates into a single "Update Gate."

**The GRU Math:**
1.  **Update Gate ($z_t$):** Decides how much past memory to keep vs. overwrite.
    $$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) $$
2.  **Reset Gate ($r_t$):** Decides how much of the past memory to just ignore right now.
    $$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) $$
3.  **Candidate Hidden State:**
    $$ \tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t] + b) $$
4.  **Final Hidden State:**
    $$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $$

GRUs perform almost as well as LSTMs but train significantly faster due to having fewer parameters. However, both were ultimately dethroned by the Transformer for long-sequence tasks due to their inability to be processed in parallel.

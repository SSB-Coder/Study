# Comprehensive Encyclopedia of Neural Network Architectures

This report details every major type of neural network that exists to date. It covers their algorithmic working principles and the core mathematics driving them.

---

## 1. Multi-Layer Perceptron (MLP) / Feedforward Neural Network (FFNN)
The foundational deep learning architecture. Data moves in one direction from input to output.

*   **Working Algorithm:** Data is passed through multiple layers of nodes. Each node multiplies inputs by weights, adds a bias, and applies a non-linear activation function.
*   **The Math:**
    *   Forward Pass: $A^{[l]} = \sigma(W^{[l]} A^{[l-1]} + b^{[l]})$
    *   Where $\sigma$ is the activation (e.g., ReLU: $f(x) = \max(0, x)$).

### 17. Multi-Agent & Orchestration (2024-2026)
*   **Concept:** Instead of one massive model, you have a mixture of specialized agents (A planner, a coder, a verifier) that talk to each other to solve a task.

---

## 5. Specialized Brain Architectures (2025-2026)

### 18. Large Concept Models (LCMs)
*   **Concept:** Moving beyond next-token prediction, LCMs are trained specifically to map out and reason over structured, causal relationships between abstract concepts rather than just generating fluent text.

### 19. Large Action Models (LAMs)
*   **Concept:** Architectures explicitly optimized to output API calls, UI interactions, and executable workflows. They are the "hands" of the AI system, trained via RL to manipulate software environments rather than chat with humans.

## 2. Convolutional Neural Networks (CNN)
Designed for grid-like topology, mostly images.

*   **Working Algorithm:** Instead of matrix multiplication across all inputs, a CNN "slides" a small filter (kernel) across the image to extract local features (edges, textures). This is followed by Pooling layers to downsample the image.
*   **The Math (Convolution):**
    *   Output Feature Map $S$ at position $(i, j)$:
    *   $S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$
    *   Where $I$ is the Image, and $K$ is the Kernel.

## 3. Recurrent Neural Networks (RNN)
Designed for sequential data (time-series, text).

*   **Working Algorithm:** RNNs have an internal "hidden state" (memory). When processing step $t$ in a sequence, it uses the input at step $t$ AND the hidden state from step $t-1$.
*   **The Math:**
    *   Hidden State: $h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$
    *   Output: $y_t = W_{yh} h_t + b_y$

## 4. Long Short-Term Memory (LSTM) & Gated Recurrent Units (GRU)
Solves the "vanishing gradient" problem of standard RNNs which causes them to forget long sequences.

*   **Working Algorithm:** Uses a "Cell State" (a conveyor belt of long-term memory) and "Gates" (tiny neural networks) to decide what information to keep, forget, or add at each time step.
*   **The Math (LSTM Gates):**
    *   Forget Gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
    *   Input Gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
    *   Cell Update: $C_t = f_t * C_{t-1} + i_t * \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
    *   Output Gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
    *   New Hidden State: $h_t = o_t * \tanh(C_t)$

## 5. Transformers
The architecture behind modern LLMs (GPT, Claude). Solves the sequence problem by processing everything in parallel using Attention.

*   **Working Algorithm:** Each token looks at every other token in the sequence to understand context. It relies heavily on Queries, Keys, and Values, without any recurrent loops.
*   **The Math (Scaled Dot-Product Attention):**
    *   $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

## 6. Autoencoders (AE) & Variational Autoencoders (VAE)
Used for data compression, denoising, and generation.

*   **Working Algorithm:** An *Encoder* network compresses the input into a tiny "latent space" (bottleneck). A *Decoder* network tries to reconstruct the original input from that tiny latent space. VAEs force the latent space to follow a probability distribution (like a Gaussian curve) so we can sample from it to generate *new* things.
*   **The Math (VAE Loss - ELBO):**
    *   $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$
    *   (Reconstruction loss minus the Kullback-Leibler divergence which acts as a regularizer).

## 7. Generative Adversarial Networks (GANs)
Used for generating hyper-realistic images, deepfakes, and audio.

*   **Working Algorithm:** Two networks compete. The *Generator* tries to create fake data. The *Discriminator* tries to guess if the data is real or fake. They train each other until the Generator makes data indistinguishable from reality.
*   **The Math (Minimax Game Objective):**
    *   $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$

## 8. Diffusion Models (e.g., Midjourney, Stable Diffusion)
The SOTA for image generation, having largely replaced GANs.

*   **Working Algorithm:** 
    1. **Forward Process:** Slowly add random Gaussian noise to an image over 1,000 steps until it is pure static.
    2. **Reverse Process:** Train a neural network (usually a U-Net) to predict and subtract the noise, step-by-step, turning pure static into a beautiful image.
*   **The Math (Reverse Step):**
    *   $p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
    *   The network predicts the mean $\mu_\theta$ of the noise at step $t$.

## 9. Graph Neural Networks (GNN)
Used for data structured as graphs (social networks, molecular chemistry, knowledge graphs).

*   **Working Algorithm:** Nodes in a graph pass "messages" (feature vectors) to their neighbors. A node updates its own state by aggregating the messages from all its connected neighbors.
*   **The Math (Message Passing):**
    *   $h_v^{(k)} = \text{UPDATE}^{(k)}\left( h_v^{(k-1)}, \text{AGGREGATE}^{(k)}\left( \{ h_u^{(k-1)} : u \in \mathcal{N}(v) \} \right) \right)$
    *   Where $h_v$ is the feature of node $v$, and $\mathcal{N}(v)$ are its neighbors.

## 10. State Space Models (SSMs) & Mamba
The emerging rival to Transformers for long-sequence tasks.

*   **Working Algorithm:** Based on control theory. They map a 1D input sequence to a latent state space using continuous differential equations, which are then discretized to run on computers. They scale linearly $O(N)$ with sequence length instead of the Transformer's $O(N^2)$.
*   **The Math (Continuous Time):**
    *   $h'(t) = A h(t) + B x(t)$
    *   $y(t) = C h(t) + D x(t)$
    *   (Where $A, B, C, D$ are learned matrices).

## 11. Spiking Neural Networks (SNNs)
The closest mathematical representation of the biological brain.

*   **Working Algorithm:** Unlike MLPs which use continuous decimal numbers, SNNs use discrete "spikes" of electricity occurring over time. A neuron only fires an output if its internal electrical charge (membrane potential) crosses a specific threshold.
*   **The Math (Leaky Integrate-and-Fire Model):**
    *   $\tau_m \frac{dV(t)}{dt} = -[V(t) - V_{rest}] + R I(t)$
    *   If $V(t) > V_{threshold}$, emit a spike and reset $V(t) = V_{reset}$.

## 12. Liquid Neural Networks (LNNs)
Developed at MIT for continuous-time robotics and edge computing.

*   **Working Algorithm:** The weights of the network are not fixed after training. They are defined by differential equations that allow the weights to dynamically change *during inference* based on the input they are receiving. Highly robust to noise.
*   **The Math:**
    *   $\frac{dx(t)}{dt} = -f(x(t)) + I(t) + \sum_{j} w_{ji} \sigma(x_j(t))$
    *   The time constant and weights are state-dependent.

## 13. Restricted Boltzmann Machines (RBM) / Deep Belief Networks
Early, historically significant networks popularized in the 2000s before modern backpropagation GPUs took off.

*   **Working Algorithm:** A generative stochastic (random) network. It learns the probability distribution of its inputs using an energy-based model. It consists of a "visible" layer and a "hidden" layer, with no intra-layer connections (hence "Restricted").
*   **The Math (Energy Function):**
    *   $E(v, h) = - \sum_{i} a_i v_i - \sum_{j} b_j h_j - \sum_{i,j} v_i W_{ij} h_j$
    *   The network tries to find states with the lowest energy.

## 14. Self-Organizing Maps (SOMs) / Kohonen Networks
Unsupervised learning networks used for dimensionality reduction and clustering.

*   **Working Algorithm:** Nodes compete for the right to respond to input data. The winning node (and its neighbors) adjust their weights to look more like the input data, creating a topographical map of the data space.
*   **The Math (Weight Update for winning node $c$):**
    *   $W_v(s + 1) = W_v(s) + \theta(u, v, s) \cdot \alpha(s) \cdot [D(t) - W_v(s)]$
    *   Where $\theta$ is a neighborhood function (how close nodes are to the winner) and $\alpha$ is learning rate.

# N01: Tape-Based Reverse-Mode Automatic Differentiation

## 1. The Beginner's Analogy
When you calculate $(x + y) \times z$, your brain executes operations forward: first adding, then multiplying. 
To determine how much the final answer changes if you wiggle $x$ slightly ($\frac{\partial \text{output}}{\partial x}$):
- **Numerical Differentiation (Finite Differences):** You could run the entire calculation again with $x + 0.0001$. But if you have 100 million parameters (like a neural network), running 100 million separate forward passes takes weeks.
- **Symbolic Differentiation:** You could write out a massive algebraic equation, but the formulas explode exponentially in size ("expression swell").
- **Reverse-Mode Autodiff:** During the forward pass, you record every elementary operation onto an execution "tape" (a Directed Acyclic Graph). Then, starting from the final loss output with a gradient of $1.0$, you walk **backward** down the tape, using the chain rule to compute exact gradients for all 100 million parameters in a single backward pass!

This is the exact algorithmic engine that powers modern deep learning frameworks (PyTorch's `torch.autograd`, TensorFlow's `tf.GradientTape`, and JAX).

## 2. The Deep Architecture & Mechanics
Reverse-mode autodiff relies on four core pillars:
1. **Computational Graph (DAG):** Nodes represent scalar/tensor values; directed edges represent elementary mathematical operations.
2. **Dynamic Trace Recording (The Tape):** Each forward operation records its parent inputs and saves necessary intermediate values for the backward pass (e.g., saving input $x$ to compute $\frac{d}{dx}\text{relu}(x)$).
3. **Topological Ordering:** Gradients cannot be computed for a node until all of its downstream consumers have completed backpropagating their gradients. Nodes are ordered via Depth-First Search (DFS) post-order reversal.
4. **Gradient Accumulation & Vector-Jacobian Products (VJPs):**
   When a node's output branches to multiple operations, its gradient is the sum of gradients from all children:
   $$\bar{x} = \sum_{c \in \text{children}(x)} \bar{c} \cdot \frac{\partial c}{\partial x}$$

## 3. The Hardcore Mathematics

### Chain Rule in Multivariable Vector Calculus
Let scalar loss $\mathcal{L} = f(y_1, y_2, \dots, y_m)$ depend on intermediate variables $y_i$, which in turn depend on input parameter $x \in \mathbb{R}^n$.
The total derivative (adjoint $\bar{x} = \frac{\partial \mathcal{L}}{\partial x}$) is:
$$\bar{x} = \sum_{j=1}^m \bar{y}_j \frac{\partial y_j}{\partial x} = J_{y/x}^T \bar{y}$$
where $J_{y/x} = \left[ \frac{\partial y_j}{\partial x_i} \right]$ is the Jacobian matrix.
In reverse mode, rather than forming the full $m \times n$ Jacobian explicitly, we evaluate the **Vector-Jacobian Product (VJP)**:
$$\text{VJP}(v) = v^T J$$
which contracts the gradient vector directly with the local derivative in $O(n)$ time!

### Elementary Adjoint Rules
1. **Addition ($z = x + y$):**
   $$\bar{x} = \bar{z}, \quad \bar{y} = \bar{z}$$
2. **Elementwise Multiplication ($z = x \odot y$):**
   $$\bar{x} = \bar{z} \odot y, \quad \bar{y} = \bar{z} \odot x$$
3. **Matrix Multiplication ($Z = X Y$ where $X \in \mathbb{R}^{B \times D_{in}}, Y \in \mathbb{R}^{D_{in} \times D_{out}}$):**
   $$\bar{X} = \bar{Z} Y^T, \quad \bar{Y} = X^T \bar{Z}$$
4. **ReLU Activation ($z = \max(0, x)$):**
   $$\bar{x} = \bar{z} \odot \mathbb{I}(x > 0)$$
5. **Sigmoid Activation ($\sigma(x) = \frac{1}{1 + e^{-x}}$):**
   $$\bar{x} = \bar{z} \odot \sigma(x) \odot (1 - \sigma(x))$$

## 4. The Algorithm Step-by-Step
1. **Forward Pass:** Construct `Tensor` instances with data. As operations (`+`, `*`, `@`, activations) execute, record the input parents and local backward closures onto each resultant `Tensor`.
2. **Topological Sort:** Starting from the scalar loss node, perform post-order DFS to obtain a linear sequence of nodes such that each node appears before all its dependencies.
3. **Backward Pass:**
   - Initialize scalar loss gradient $\bar{\mathcal{L}} = 1.0$.
   - Iterate through the topologically sorted nodes in reverse.
   - For each node, execute its registered backward function to distribute its adjoint $\bar{z}$ to its parent inputs $\bar{x}$ and $\bar{y}$.
4. **Optimizer Update:** Iterate over model parameters and apply gradient descent:
   $$W \leftarrow W - \eta \bar{W}, \quad \bar{W} \leftarrow 0$$

## 5. Limitations & Failure Modes
- **Memory Overhead of Forward Retention:** Backward passes require keeping intermediate forward activations in memory (leading to GPU Out-of-Memory errors in large models, solved via activation checkpointing).
- **Control Flow Cycles:** Dynamic tapes cleanly handle loops and branching by unrolling them into a dynamic execution DAG, but in-place mutations overwrite required forward values and corrupt the backward pass.

## 6. Where This Fits
Cross-reference: This engine provides the foundational mechanism behind all deep neural network backpropagation in the sibling `Transformer` repository and replaces the hand-calculated parameter gradients in Module I (Optimization).

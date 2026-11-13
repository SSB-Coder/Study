# J12: Multi-Task & Transfer Learning (MMD & Shared Representations)

## 1. Intuition & The "Grand Analogy"
Imagine learning to play musical instruments. If you first master the violin, you learn to read sheet music, internalize rhythm, and coordinate fine finger pressure. When you subsequently learn to play the cello, you do not start from absolute zero (**Transfer Learning**). 
Furthermore, if an orchestra student practices both violin and cello simultaneously (**Multi-Task Learning**), their shared mental representation of acoustic harmony improves both skills faster than practicing either instrument in isolation.

However, if you take a violinist trained on classical music and force them to play electric guitar through heavy distortion, the sound physics have shifted (**Domain Shift**). **Maximum Mean Discrepancy (MMD)** provides a rigorous mathematical yardstick to measure the statistical distance between source and target domains.

---

## 2. Architectural & Mathematical Breakdown

### Multi-Task Learning (Hard Parameter Sharing, Caruana 1997)
An input $\mathbf{x}$ passes through a shared trunk representation $\mathbf{z} = g(\mathbf{x}; \mathbf{W}_{\text{shared}})$, which then branches into $T$ task-specific heads:
$$\hat{y}_t = h_t(\mathbf{z}; \mathbf{W}_t), \quad t \in \{1, \dots, T\}$$
Joint objective minimizes weighted sum of task losses:
$$\mathcal{L}_{\text{total}}(\mathbf{W}_{\text{shared}}, \{ \mathbf{W}_t \}) = \sum_{t=1}^T \lambda_t \mathcal{L}_t(\hat{y}_t, y_t)$$

---

## 3. Hardcore Mathematical Derivations

### Maximum Mean Discrepancy (MMD, Gretton et al., 2012)
Let $P$ and $Q$ be two probability distributions. MMD measures the distance between their mean embeddings in a Reproducing Kernel Hilbert Space $\mathcal{H}_k$:
$$\text{MMD}^2(\mathcal{F}, P, Q) \triangleq \left\| \mathbb{E}_{\mathbf{x} \sim P}[\phi(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[\phi(\mathbf{y})] \right\|_{\mathcal{H}}^2$$
Using the kernel trick $\langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle = k(\mathbf{x}, \mathbf{x}')$:
$$\text{MMD}^2(P, Q) = \mathbb{E}_{\mathbf{x}, \mathbf{x}' \sim P}[k(\mathbf{x}, \mathbf{x}')] - 2 \mathbb{E}_{\mathbf{x} \sim P, \mathbf{y} \sim Q}[k(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{\mathbf{y}, \mathbf{y}' \sim Q}[k(\mathbf{y}, \mathbf{y}')]$$
Given empirical samples $X = \{ \mathbf{x}_i \}_{i=1}^n$ and $Y = \{ \mathbf{y}_j \}_{j=1}^m$:
$$\widehat{\text{MMD}}^2(X, Y) = \frac{1}{n^2} \sum_{i=1}^n \sum_{i'=1}^n k(\mathbf{x}_i, \mathbf{x}_{i'}) - \frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m k(\mathbf{x}_i, \mathbf{y}_j) + \frac{1}{m^2} \sum_{j=1}^m \sum_{j'=1}^m k(\mathbf{y}_j, \mathbf{y}_{j'})$$
- If $P = Q$, $\text{MMD}(P, Q) = 0$.
- In Domain Adaptation (e.g. DAN / DDC), minimizing $\widehat{\text{MMD}}^2$ aligns the latent feature distributions of source and target domains.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Multi-Task Linear Trunk with Shared Gradients
Inputs: Multi-task dataset (X, Y_1, Y_2), learning rate eta
1. Initialize shared projection W_trunk in R^{h x d}
2. Initialize task heads W_1 in R^{1 x h}, W_2 in R^{1 x h}
3. For step = 1 to MaxSteps:
     z = relu(W_trunk * x)
     y1_pred = W_1 * z, y2_pred = W_2 * z
     loss1 = (y1_pred - y1)^2, loss2 = (y2_pred - y2)^2
     Backpropagate joint gradients:
       dL/dW_1 = dloss1/dW_1
       dL/dW_2 = dloss2/dW_2
       dL/dW_trunk = (dloss1/dz + dloss2/dz) * dz/dW_trunk
     Update all weights
```

---

## 5. Failure Modes & When It Breaks
1. **Negative Transfer:** If tasks are conflicting or uncorrelated, forcing them to share a common representation trunk harms performance on both tasks compared to training separate models.
2. **Task Dominance / Gradient Starvation:** If Task 1 has gradients $100\times$ larger than Task 2, the shared trunk optimizes exclusively for Task 1 (mitigated by GradNorm).

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Instruction Tuning is Massively Multi-Task Learning:** Models like FLAN and InstructGPT train a single transformer backbone simultaneously on thousands of disparate tasks (summarization, coding, translation, sentiment).
- **Domain Adaptation for Enterprise LLMs:** Fine-tuning base generalist models on specialized internal corporate wikis while preserving general reasoning.

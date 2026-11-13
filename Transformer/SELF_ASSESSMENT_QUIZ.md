# Master Self-Assessment Quiz & Diagnostic Guide

Test your mathematical, code-level, and architectural comprehension across all 17 phases before building your own models. Click on **"<details><summary>Check Answer</summary>"** under each question to reveal the full solution and code justification.

---

## Phase 0: Mathematical Foundations & Matrix Algebra

### Question 1: Matrix Multiplication Dimension Compatibility
You have an input batch matrix $X \in \mathbb{R}^{B \times N \times D}$ (where $B=2$ batch size, $N=4$ sequence length, $D=8$ embedding dimension) and a Projection Weight matrix $W_Q \in \mathbb{R}^{D \times K}$ (where $K=16$). 
1. What is the tensor shape of the resulting Query matrix $Q = X W_Q$?
2. Write the exact `torch.einsum` equation to calculate $Q$.

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Shape of $Q$: `(2, 4, 16)` or `[B, N, K]`.
2. PyTorch einsum notation:
   ```python
   Q = torch.einsum('b n d, d k -> b n k', X, W_Q)
   ```
*Explanation:* Matrix multiplication contracts the inner dimension $D$. $(B, N, D) \times (D, K) \rightarrow (B, N, K)$.
</details>

---

### Question 2: Softmax Backpropagation Derivative
In Scaled Dot-Product Attention, $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
Why is the Softmax score divided by $\sqrt{d_k}$? What happens to the gradients during backpropagation if we omit $\sqrt{d_k}$ for large head dimensions ($d_k = 128$)?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
For large $d_k$, the dot product $Q K^T$ grows large in magnitude ($\text{Var}(Q K^T) \approx d_k$). Large input values push the Softmax function into regions with extremely small gradients ($\frac{\partial \text{softmax}(z_i)}{\partial z_j} \approx 0$). This causes the **Vanishing Gradient Problem**, preventing lower layers from learning. Dividing by $\sqrt{d_k}$ normalizes the variance to $1.0$.
</details>

---

## Phase 4: Transformer & MoE Architecture Mechanics

### Question 3: Sparse Router Top-K Selection
In a Sparse Mixture of Experts (MoE) layer with 8 total experts, a Router network outputs logits $g \in \mathbb{R}^{B \times N \times 8}$.
If we select $top\_k = 2$, how do we mathematically shut off (zero out) the remaining 6 sleeping experts so they consume zero compute during the forward pass?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
We use `torch.topk(logits, k=2)` to find top expert indices and values, compute Softmax *only over those top 2 values*, and use `torch.gather` or sparse indexing to execute only those 2 expert sub-networks:
```python
top_logits, top_indices = torch.topk(logits, k=2, dim=-1)
routing_weights = torch.softmax(top_logits, dim=-1)
```
The remaining 6 experts are completely unreferenced in the forward execution loop, saving up to 75% of FLOPs.
</details>

---

## Phase 8 & 13: Triton Hardware Kernels & Distributed Systems

### Question 4: SRAM vs HBM Roofline Bottlenecks
Why is standard Scaled Dot-Product Attention $O(N^2)$ in VRAM memory overhead, and how does OpenAI Triton's FlashAttention kernel achieve $O(N)$ VRAM memory footprint without storing the $N \times N$ attention matrix?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
Standard Attention computes the full $N \times N$ matrix $S = Q K^T$ and writes it to GPU High Bandwidth Memory (HBM). For $N=64,000$, $S$ takes tens of Gigabytes of VRAM. FlashAttention tiles $Q, K, V$ into small $64 \times 64$ blocks in ultra-fast Streaming Multiprocessor (SM) SRAM cache, computes Softmax online using Online Normalization (running max and running sum reduction), and streams the result directly into Output HBM without ever writing the $N \times N$ attention matrix to HBM.
</details>

---

## Phase 15 & 16: Fine-Tuning, Quantization & RAG Ecosystem

### Question 5: LoRA Low-Rank Parameter Decomposition
In a 16-bit Linear layer with input dimension $d_{in} = 4096$ and output dimension $d_{out} = 4096$:
1. How many parameters exist in the base weight matrix $W_0$?
2. If we apply LoRA with rank $r = 8$, how many trainable parameters exist in adapters $A$ and $B$?
3. What is the parameter reduction percentage?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Base parameters: $4096 \times 4096 = 16,777,216$ parameters (~16.7M).
2. LoRA adapter parameters:
   * Matrix $B \in \mathbb{R}^{8 \times 4096} = 32,768$
   * Matrix $A \in \mathbb{R}^{4096 \times 8} = 32,768$
   * Total LoRA params = $32,768 + 32,768 = 65,536$ parameters.
3. Parameter reduction:
   $$\text{Reduction} = 1 - \frac{65,536}{16,777,216} \approx 99.61\% \text{ reduction}$$
We freeze 99.61% of the weights and only compute gradients for the 0.39% LoRA parameters.
</details>

---

### Question 6: HNSW Vector Search Routing Complexity
Why does Hierarchical Navigable Small World (HNSW) search scale at $O(\log N)$ time complexity compared to $O(N)$ for brute-force Cosine Similarity search over 1,000,000 embedding vectors?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
HNSW constructs a multi-layer graph. Top layers contain a sparse subset of vectors acting as high-speed "expressways." Search starts at the top layer, greedily walking towards nodes with higher Cosine Similarity to the query. Once a local maximum is reached, the search drops down to the next layer at that exact entry point, repeating down to Layer 0. This logarithmic graph traversal skips 99% of unrelated vectors.
</details>

---

## Phase 16: Universal Libraries, Agentic Graphs & Observability

### Question 7: Scikit-Learn ColumnTransformers & PyTorch AMP Mechanics
1. Why is wrapping feature scalers and imputers inside a Scikit-Learn `Pipeline` with `ColumnTransformer` strictly mandatory before evaluating model performance with Cross-Validation?
2. In PyTorch 2.x, why must we use `scaler.scale(loss).backward()` and `scaler.unscale_(optimizer)` before calling `torch.nn.utils.clip_grad_norm_()` when training with FP16 Automatic Mixed Precision?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Fitting scalers/imputers on the entire dataset prior to cross-validation causes **Data Leakage**—the validation folds influence the mean and standard deviation used to transform the training folds, artificially inflating cross-validation scores. A `Pipeline` ensures transformers are fitted *only* on the training splits of each fold.
2. In FP16 Mixed Precision, gradients are multiplied by a large scaling factor to prevent small gradient underflows. If `clip_grad_norm_()` is called before unscaling, it would compute the norm on scaled gradients, resulting in premature over-clipping or distorted gradient directions. `scaler.unscale_(optimizer)` restores true gradients before clipping.
</details>

---

### Question 8: LangGraph Cyclic State Machines vs. Observability Spans
1. What fundamental architectural limitation of linear DAGs (like basic LangChain chains) does **LangGraph** solve for multi-agent reasoning?
2. How do observability platforms like **Langfuse** and **LangSmith** calculate token cost and track multi-agent tool latency without slowing down real-time LLM inference?

<details>
<summary>Check Answer & Explanation</summary>

**Answer:**
1. Real agentic workflows require **cycles, loops, and conditional dynamic branching** (e.g., iteratively calling tools, evaluating intermediate errors, and re-prompting until completion) plus state persistence (`checkpointers`) for Human-In-The-Loop interrupts. Standard linear DAGs only support forward acyclic passes.
2. Observability frameworks use non-blocking asynchronous event callbacks or OpenTelemetry background flush threads. They intercept model tokens, time stamps, and prompt IDs in memory and ship telemetry payloads out-of-band to collectors, computing exact USD token pricing using cached rate cards with zero inference latency penalty.
</details>

---

## 🏆 Scoring & Diagnostic Results

* **7–8 Correct:** **Mastery Level.** You possess comprehensive mastery across foundational mathematics, PyTorch/CUDA execution, Hugging Face PEFT fine-tuning, high-throughput vLLM serving, LangGraph multi-agent systems, and Langfuse/LangSmith observability.
* **4–6 Correct:** **Intermediate Level.** Review [44_Complete_ML_DL_AI_Libraries_Master_Guide.md](Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md) and execute [44_Complete_ML_DL_AI_Libraries_Cookbook.py](Code/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Cookbook.py).
* **0–3 Correct:** **Foundational Level.** Return to [STUDY_ROADMAP.md](STUDY_ROADMAP.md) and start from Phase 0 math and Scikit-Learn / PyTorch fundamentals.

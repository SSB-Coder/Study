# 🧠 Transformer & Neural Architectures — Master Cheat Sheet

> Built from every file in the repository (9 core Reports, 51 Deep Dives, 48 code files, 57 interactive HTML simulations + master portal).
> Format: **equation → one-line meaning → code idiom**. This is a *lookup* document, not a tutorial — read it after you already understand the concept once, to jog your memory fast.
> LaTeX renders natively in Obsidian/Typora. If viewing on GitHub, math blocks show as raw `$...$`.

**Jump table:** [0.5 🎯 Build-Priority Guide](#05--build-priority-guide--read-this-first-if-your-goal-is-make-a-working-transformer) · [0. Notation](#0-universal-notation) · [1. Core Math](#1-universal-math-toolkit) · [2. The Transformer](#2-the-classic-transformer-the-main-event) · [3. Positional Encoding](#3-positional-encoding--rope--alibi) · [4. Efficient/Scaled Attention](#4-beyond-dense-attention-the-efficiency-family) · [5. Other Architecture Families](#5-other-core-architecture-families) · [6. Frontier Architectures + SOTA/Agents/KG/OI/QML](#6-frontier--exotic-architectures) · [7. Training & Alignment](#7-training-alignment--fine-tuning) · [8. Systems & Data](#8-systems-distributed-training--data) · [9. PyTorch/CUDA/Triton Glossary](#9-pythonpytorchcudatriton-keyword-glossary) · [10. Libraries Quick-Ref](#10-libraries--frameworks-quick-reference) · [11. Equation Index](#11-master-equation-index-alphabetical) · [12. Code Keyword Index](#12-master-code-keyword-index-alphabetical) · [13. What's NOT in here](#13-whats-in-the-zip-but-not-in-this-cheat-sheet) · [14. Self-check](#14-self-check-answering-your-4-questions)

> **v1.1 revision note:** After you asked "does this cover everything," I re-audited the whole archive a second time and found real gaps — see the changelog in §13.
> **v1.2 revision note:** Added §0.5 below — a build-priority guide, because most of this document is *survey/reference* material for the full 44-architecture archive, and only a fraction of it is what you'll actually touch while coding a Transformer.

---

## 0.5 🎯 Build-Priority Guide — read this first if your goal is "make a working Transformer"

**Reality check on the rest of this document:** roughly 65% of it (§5, §6, most of §8, half of §10) is the *survey layer* — one architecture family after another, mirroring how the zip itself is organized. You will likely never write GAN, SOM, or RBM code while building a Transformer. It's there so you understand *where the Transformer sits relative to everything else*, not because you need it to build one. This section is the honest "what do I actually open" map.

**Fastest path:** the zip's own `Code/03_Self_Attention.py` → `04_Transformer_Architecture.py` → `05_MoE_Architecture.py` is already a complete, working, minimal build in that exact order. Everything below just tells you which cheat-sheet section explains each piece of that code.

### Tier 1 — Required. You cannot have a working Transformer without these.
| # | Piece | Where in this doc |
|---|---|---|
| 1 | Token embeddings + positional embeddings | §2.5 |
| 2 | Q, K, V linear projections | §2.1 |
| 3 | Scaled dot-product attention formula | §2.2 |
| 4 | Multi-head reshape/einsum + concat | §2.3 (real code) |
| 5 | Causal mask (so decoder can't see the future) | §2.3, §2.5 |
| 6 | Residual connection + LayerNorm ("Add & Norm") | §1.6, §2.4 |
| 7 | Feed-forward block | §2.4 |
| 8 | Stack N blocks → output head → cross-entropy loss | §2.5, §2.6 |
| 9 | Backprop + optimizer step (`loss.backward()`, `optimizer.step()`, `zero_grad()`) | §1.4, §1.5 |
| 10 | **Shape cheat sheet** — you will hit shape-mismatch bugs constantly | §2.7 |

If you can build and train something using only the 10 rows above, you have a real, working Transformer. Everything past this point is refinement or a different architecture entirely.

### Tier 2 — Do these next; this is what separates a toy Transformer from a modern/competitive one.
| Piece | Where |
|---|---|
| RoPE instead of learned/sinusoidal positions | §3 |
| SwiGLU instead of plain ReLU-MLP feed-forward | §2.4 |
| AdamW + gradient clipping + mixed precision (AMP) training loop | §1.5, §10 |
| `torch.compile` for speed | §9.3 |
| KV cache for fast autoregressive inference | §4.7 |

### Tier 3 — Optional. Only touch these if you have a *specific* reason to (scaling past dense attention, fine-tuning cheaply, going sparse, etc.)
- MoE (§4.1) — only if you want to scale parameters sparsely.
- FlashAttention/Triton (§4.6) — only once you hit real memory/speed walls at long context.
- LoRA / quantization (§7.4–7.5) — only when fine-tuning an existing model cheaply, not for training from scratch.
- **Everything in §5 and §6** (CNN, RNN, LSTM, VAE, GAN, Diffusion, GNN, SNN, LNN, RBM, SOM, JEPA, KAN, Neurosymbolic, Active Inference, Omnimodal, Agentic/PPO) — these are **other architecture families**, not Transformer components. They're in the doc because they're in the zip, for comparative literacy, not because a Transformer needs them.
- Most of §10 (Keras/JAX/XGBoost/LangGraph/vLLM) — different tools for different jobs, not Transformer-build code.

---

## 0. Universal Notation

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| $X$ | Input matrix/vector | $W, b$ | Weight matrix, bias vector |
| $Z^{[l]}$ | Pre-activation of layer $l$ ($=W^{[l]}A^{[l-1]}+b^{[l]}$) | $A^{[l]}$ | Post-activation of layer $l$ |
| $\hat y$ | Model prediction | $y$ | Ground truth |
| $\mathcal{L}$ or $\mathcal{J}$ | Loss / cost | $\alpha$ | Learning rate |
| $\partial \mathcal{L}/\partial W$ | Gradient of loss w.r.t. $W$ (a.k.a. `dW`) | $\nabla$ | Gradient vector (all partials) |
| $\sigma$ | Sigmoid, or sometimes "activation function" generically | $\odot$ or $*$ | Element-wise (Hadamard) product |
| $Q,K,V$ | Query, Key, Value matrices | $d_k$ | Key/head dimension |
| $N$ | Sequence length | $m$ | Batch size |
| $\mathbb{E}$ | Expected value | $D_{KL}$ | Kullback-Leibler divergence |
| $\theta$ | Generic "all trainable parameters" | $\pi_\theta$ | Policy (RL / RLHF context) |

---

## 1. Universal Math Toolkit
*(Source: `06_Mathematical_Foundations.md`, `Deep_Dives/00_The_Complete_Mathematics_of_Neural_Networks.md`, `Deep_Dives/22_Information_Theory_and_Manifolds.md`)*

### 1.1 The Neuron & Forward Pass
$$ z = W\cdot X + b \qquad a = f(z) $$
$$ Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g^{[l]}(Z^{[l]}), \qquad A^{[0]}=X $$
Shapes: $X\in\mathbb R^{n_0\times m}$, $W^{[l]}\in\mathbb R^{n_l\times n_{l-1}}$, $b^{[l]}\in\mathbb R^{n_l\times 1}$.

```python
# code idiom (nn.Module forward pass skeleton, seen in every architecture)
z = self.linear(x)          # W·x + b
a = F.relu(z)                # or torch.sigmoid, torch.tanh
```

### 1.2 Activation Functions
| Name | Formula | Range |
|---|---|---|
| Sigmoid | $f(z)=\frac{1}{1+e^{-z}}$ | (0,1) |
| ReLU | $f(z)=\max(0,z)$ | [0,∞) |
| Tanh | $f(z)=\tanh(z)$ | (-1,1) |
| Softmax | $\text{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$ | sums to 1 |
| SiLU/Swish (SwiGLU base) | `F.silu(x)` $= x\cdot\sigma(x)$ | used in modern MLP blocks |

### 1.3 Loss Functions
$$ \text{MSE: } \mathcal L=\frac1n\sum(\hat y-y)^2 \qquad\qquad \text{Binary Cross-Entropy: } \mathcal L=-\big(y\log\hat y+(1-y)\log(1-\hat y)\big) $$
$$ \text{Categorical Cross-Entropy: } \mathcal L = -\sum_c y_c\log(\hat y_c) \;\xrightarrow{\text{one-hot}}\; \mathcal L=-\log(\hat y_{\text{correct}}) $$
```python
loss = F.mse_loss(pred, target)
loss = F.cross_entropy(logits, targets)          # combines log_softmax + NLL
loss = F.binary_cross_entropy(pred, target)
```

### 1.4 Backpropagation (The Chain Rule)
$$ dZ^{[L]} = A^{[L]}-Y \quad(\text{softmax/sigmoid + CE output layer}) $$
$$ dW^{[l]} = \frac1m\,dZ^{[l]}(A^{[l-1]})^T \qquad db^{[l]}=\frac1m\sum_i dZ^{[l](i)} $$
$$ dZ^{[l-1]} = (W^{[l]})^T dZ^{[l]} \;*\; g'^{[l-1]}(Z^{[l-1]}) $$
> Gradient **explodes** if repeated weight products > 1; **vanishes** if < 1 — root cause of RNN long-range failure ($\prod_{j} \partial h_j/\partial h_{j-1}$, see §5.3).

```python
loss.backward()                 # autograd computes every dW, db automatically
optimizer.step()                # applies the update rule below
optimizer.zero_grad()           # MUST be called each step or grads accumulate
```

### 1.5 Optimizers
$$ \textbf{SGD: } W \leftarrow W-\alpha\, dW $$
$$ \textbf{Momentum: } v_{dW}=\beta_1 v_{dW}+(1-\beta_1)dW $$
$$ \textbf{RMSprop: } s_{dW}=\beta_2 s_{dW}+(1-\beta_2)dW^2 $$
$$ \textbf{Adam: } W \leftarrow W-\alpha\frac{v_{dW}}{\sqrt{s_{dW}}+\epsilon} \quad(\text{Momentum + RMSprop combined}) $$
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # prevent explosion
```

### 1.6 Normalization
$$ \text{LayerNorm}(x) = \gamma\,\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta $$
Normalizes **across the feature dimension per token** (unlike BatchNorm, which normalizes across the batch). This is why Transformers use LayerNorm — it works with variable-length sequences and batch size 1.
```python
self.norm = nn.LayerNorm(embed_size)
x = self.norm(sublayer_output + residual_x)   # "Add & Norm"
```

### 1.7 Information Theory (underlies every loss & KL term you'll see later)
$$ H(P) = -\sum P(x)\log P(x) \qquad(\text{Shannon Entropy}) $$
$$ D_{KL}(P\|Q) = \sum P(x)\log\frac{P(x)}{Q(x)} \qquad(\text{"extra bits wasted using } Q \text{ instead of } P\text{"}) $$
$$ \text{Cross-Entropy} = H(P) + D_{KL}(P\|Q) $$
> Minimizing Cross-Entropy loss ≡ minimizing $D_{KL}$, since $H(P)$ (the true label distribution) is fixed. This is *why* CE is the default classification loss.

---

## 2. The Classic Transformer (the main event)
*(Source: `Reports/03_Transformers_and_SOTA_Models.md`, `Deep_Dives/13_Classic_Transformer_Dense.md`, `Deep_Dives/00_...md §4`, `Code/03_Self_Attention.py`, `Code/04_Transformer_Architecture.py`)*

### 2.1 Q, K, V — the three questions every token asks
$$ Q = XW_Q \qquad K = XW_K \qquad V = XW_V $$
- **Query**: "what am I looking for?" · **Key**: "what do I contain?" · **Value**: "what do I actually mean?"

### 2.2 Scaled Dot-Product Attention — THE core equation
$$ \boxed{\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V} $$
| Step | What happens | Output shape |
|---|---|---|
| $QK^T$ | dot-product similarity between every pair of tokens | `[N, N]` |
| $/\sqrt{d_k}$ | scaling — stops softmax gradients from vanishing when $d_k$ is large | `[N, N]` |
| `softmax(dim=-1)` | turns scores into probabilities that sum to 1 per row | `[N, N]` |
| $\times V$ | blend of Value vectors weighted by attention | `[N, d_v]` |
| causal mask | set future positions to $-\infty$ before softmax → forces prob = 0 | decoder only |

### 2.3 Multi-Head Attention — real code, from `03_Self_Attention.py`
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size
        self.values  = nn.Linear(embed_size, embed_size, bias=False)
        self.keys    = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out  = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])   # QK^T per head
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)

        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.embed_size)            # concat heads
        return self.fc_out(out)
```
**`einsum` cheat**: `"nqhd,nkhd->nhqk"` = batch(n), query_len(q), heads(h), head_dim(d) × batch(n), key_len(k), heads(h), head_dim(d) → batch, heads, query_len, key_len. This *is* $QK^T$, computed per-head, in one line.

### 2.4 The Transformer Block ("Add & Norm" sandwich) — from `04_Transformer_Architecture.py`
$$ \text{Self-Attention} \to \text{Add \& Norm} \to \text{Feed-Forward MLP} \to \text{Add \& Norm} $$
```python
class TransformerBlock(nn.Module):
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x   = self.dropout(self.norm1(attention + query))   # residual + LayerNorm
        fwd = self.feed_forward(x)                            # Linear→ReLU→Linear
        out = self.dropout(self.norm2(fwd + x))               # residual + LayerNorm
        return out
```
- **Residual connection**: $x + \text{Layer}(x)$ — lets gradients skip straight through deep stacks (same fix as ResNet §5.2).
- **Feed-Forward**: `nn.Linear(d, 4d) → ReLU → nn.Linear(4d, d)` — the "4×" expansion (`forward_expansion`) is standard in this teaching code; **real SOTA models (Llama, DeepSeek, etc.) replace this with SwiGLU**, from `Code/Deep_Dives/44`:
```python
class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))          # SiLU/Swish gate
        up   = self.up_proj(x)
        return self.norm(self.down_proj(gate * up))   # gated FFN, no bias terms
```

### 2.5 Full model wiring — `GenerativeTransformerModel`
```python
self.word_embedding     = nn.Embedding(vocab_size, embed_size)
self.position_embedding = nn.Embedding(max_length, embed_size)
out = self.word_embedding(x) + self.position_embedding(positions)   # inject order
for layer in self.layers: out = layer(out, out, out, mask)          # stack N blocks
logits = self.fc_out(out)                                            # → vocab_size
```
Causal mask for autoregressive (GPT-style) decoding, padding mask:
```python
src_mask = (src != 0).unsqueeze(1).unsqueeze(2)   # ignore PAD tokens, shape (N,1,1,src_len)
```

### 2.6 Final Layer & Loss
$$ \mathcal L = -\sum_c y_c\log(\hat y_c) = -\log(\hat y_{\text{correct}}) \qquad\text{(Cross-Entropy over vocab)} $$

### 2.7 Shape cheat sheet (memorize this table — it's 90% of debugging)
| Tensor | Shape |
|---|---|
| Input tokens | `[batch, seq_len]` |
| Embeddings | `[batch, seq_len, embed_size]` |
| Q, K, V (pre-split) | `[batch, seq_len, embed_size]` |
| Q, K, V (per-head) | `[batch, seq_len, heads, head_dim]` |
| Attention scores $QK^T$ | `[batch, heads, seq_len, seq_len]` |
| Attention output | `[batch, seq_len, embed_size]` |
| Final logits | `[batch, seq_len, vocab_size]` |

### 2.8 Complexity — why everything in §4 exists
$O(N^2)$ time & memory in $QK^T$ (sequence length $N$) → the **quadratic bottleneck**. KV Cache during generation grows $O(N)$ in memory. These two facts motivate almost every "advanced architecture" report in this zip.

---

## 3. Positional Encoding & RoPE / ALiBi
*(Source: `Deep_Dives/30_Positional_Encodings_and_RoPE.md`)*

Attention itself is **permutation-invariant** — no built-in sense of order.

| Method | Equation | Note |
|---|---|---|
| Sinusoidal (2017 original) | $\text{Embedding} = \text{WordVec} + \text{SineWavePos}$ | breaks past training length |
| **RoPE** (Llama/Mistral/Qwen) | $q_m = q\,e^{im\theta}$ | rotates Q,K as complex 2D vectors by position $m$; dot product naturally depends only on **relative** distance $m-n$ |
| NTK-aware / YaRN | divide rotation angle by a scale factor | "tricks" RoPE into extrapolating to longer context without retraining |
| ALiBi (MPT) | subtract a distance-proportional penalty from attention scores pre-softmax | no rotation; naturally extrapolates to unseen lengths |

---

## 4. Beyond Dense Attention: the Efficiency Family
*(Source: `Deep_Dives/14, 15, 23, 25, 26, 27, 31`, `Code/05_MoE_Architecture.py`, `Code/Deep_Dives/15, 23, 25, 26, 27`)*

### 4.1 Mixture of Experts (MoE) — sparse compute
$$ G(x)=\text{Softmax}(W_g\cdot x) \qquad G_{sparse}(x)=\text{Softmax}(\text{KeepTopK}(W_g\cdot x)) $$
$$ y = \sum_{i=1}^N G_{sparse}(x)_i \cdot E_i(x) \qquad(\text{non-selected } E_i(x) \text{ are never computed}) $$
```python
routing_probs = F.softmax(self.router(x), dim=-1)
routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # renormalize
for i, expert in enumerate(self.experts):
    token_idx, choice_idx = torch.where(selected_experts == i)
    flat_output.index_add_(0, token_idx, expert(flat_x[token_idx]) * weight)
```
Key trick: `torch.topk` for expert selection, `index_add_` for scatter-accumulate.

### 4.2 State Space Models (SSM) / Mamba — the $O(N)$ alternative
$$ h'(t)=Ah(t)+Bx(t) \qquad y(t)=Ch(t) $$
Discretize (Zero-Order Hold) for computers:
$$ \bar A=\exp(\Delta A) \qquad \bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B $$
$$ h_k=\bar A h_{k-1}+\bar B x_k \qquad y_k=Ch_k $$
Unrolled = a convolution: $y = x * \bar K$ where $\bar K=(C\bar B, C\bar A\bar B, C\bar A^2\bar B,\dots)$ — computable via FFT in $O(N\log N)$, trained in parallel, runs as $O(1)$-memory RNN at inference (**no KV cache!**).
```python
A = -torch.exp(self.A_log)             # keep eigenvalues negative → stable decay
A_bar = torch.exp(torch.outer(dt, A))  # ZOH discretization
h_t = A_bar.unsqueeze(0) * h_t + B_bar * x_t.unsqueeze(2)   # recurrent state update
y_t = torch.sum(C_t * h_t, dim=-1)
```
Trade-off: struggles at exact "needle-in-haystack" recall vs. Transformer, because state is compressed.

### 4.3 Linear Attention & Gated Linear Attention (GLA)
$$ (QK^T)V = Q(K^TV) \qquad\Rightarrow\qquad O(N) \text{ instead of } O(N^2) $$
$$ \text{Attention} \approx \phi(Q)\big(\phi(K)^TV\big) \qquad(\phi=\text{kernel feature map, e.g. Performer's random Fourier features}) $$
```python
class LinearAttention(nn.Module):
    def forward(self, Q, K, V):
        Q, K = self.feature_map(Q), self.feature_map(K)   # e.g. nn.ELU()+1
        KV = torch.matmul(K.transpose(-2,-1), V)           # compute K^T V FIRST
        return torch.matmul(Q, KV)                          # then Q(K^TV) — O(N)
```
GLA adds data-dependent gating on top for selective forgetting (recovers near-Transformer quality at $O(N)$).

### 4.4 RWKV & xLSTM
- xLSTM's **mLSTM**: scalar memory $C_t\in\mathbb R^d$ → **matrix memory** $C_t\in\mathbb R^{d\times d}$, giving $O(d^2)$ storage capacity → near-Transformer associative recall.
- RWKV: RNN-style, **zero KV cache**, linear time.

### 4.5 Titans (Neural Memory)
Stores past context inside the **weights** of a small neural network (not a KV-cache matrix) → $O(1)$ memory footprint with strong long-context recall.
```python
def memorize(self, x_current):  ...   # gradient-based write into memory weights
def retrieve(self, x_query):    ...   # forward pass through memory net = "read"
```

### 4.6 FlashAttention & Triton — same math, faster hardware execution
Standard attention is **memory-bound**: naive PyTorch writes the full $N\times N$ score matrix to slow HBM three times (`S=QK^T`, `softmax`, `×V`). FlashAttention **tiles** Q/K/V into GPU SRAM and never materializes the full matrix.
```python
@triton.jit
def _fwd_kernel(Q, K, V, Out, ..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    q = tl.load(q_ptrs)                       # HBM → SRAM
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs); v = tl.load(v_ptrs)
        qk = tl.dot(q, k) * 0.125              # Q·K^T fully inside SRAM
        acc += tl.dot(qk, v)                    # ×V still inside SRAM
    tl.store(out_ptrs, acc)                     # SRAM → HBM, ONCE
```
Result: exact same math, 3–10× faster, avoids $O(N^2)$ HBM read/writes.

### 4.7 Multi-Head Latent Attention (MLA) & Speculative Decoding
- **MLA**: compresses all K,V into one low-rank latent $c^{KV}$ via a projection matrix; "up-projects" back to full K,V only inside fast SRAM → shrinks the KV cache dramatically (DeepSeek's trick).
- **Speculative decoding**: a small "draft" model proposes $k$ tokens; the big "target" model verifies them in a single parallel forward pass — accepted tokens are free speedup.
```python
draft_tokens = mock_draft_model(context, k_guesses=4)
verified     = mock_target_model(context, draft_tokens)   # 1 fwd-pass verifies all k
```

---

## 5. Other Core Architecture Families
*(Source: `Reports/02, 07`, `Deep_Dives/01–12`, `Code/01, 02`, `Code/Deep_Dives/01–12`)*

### 5.1 The Perceptron & MLP / Feedforward
$$ Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]} \qquad A^{[l]}=g^{[l]}(Z^{[l]}) $$
**The Perceptron (single neuron, step activation)** — the literal atomic unit, from `Code/01_Simple_NNs.py`:
```python
def _activation_func(self, x): return np.where(x >= 0, 1, 0)          # step function
linear_output = np.dot(x_i, self.weights) + self.bias
update = self.lr * (y_true - y_predicted)                              # Delta rule
self.weights += update * x_i; self.bias += update
```
**Pure-NumPy 2-layer backprop** (no autograd — shows the raw chain rule from §1.4 with real numbers):
```python
self.Z1 = np.dot(X, self.W1) + self.b1;  self.A1 = self.sigmoid(self.Z1)
self.Z2 = np.dot(self.A1, self.W2) + self.b2;  self.A2 = self.sigmoid(self.Z2)
dZ2 = (output - y) * self.sigmoid_derivative(self.Z2)
dW2 = (1/m) * np.dot(self.A1.T, dZ2)
dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.Z1)         # propagate error back
self.W2 -= self.lr * dW2                                                # gradient descent step
```
```python
nn.Linear(in_features, out_features); F.relu(z); F.softmax(logits, dim=-1)
```

### 5.2 CNN
$$ S(i,j)=(I*K)(i,j)=\sum_m\sum_n I(i+m,j+n)K(m,n) \qquad \text{OutSize}=\left\lfloor\frac{W-F+2P}{S}\right\rfloor+1 $$
Backward: gradient w.r.t. kernel = convolution of input with output-gradient; gradient w.r.t. input = full convolution with 180°-rotated kernel.
- **ResNet trick**: $F(x)+x$ (residual/skip connection) — solves vanishing gradients in very deep CNNs; same idea reused in Transformer §2.4.
```python
nn.Conv2d(...); F.max_pool2d(x, 2); out = F.relu(self.conv(x)) + x   # residual
```

### 5.3 RNN
$$ h_t=\tanh(W_{hh}h_{t-1}+W_{hx}x_t+b_h) \qquad \hat y_t=\text{softmax}(W_{yh}h_t+b_y) $$
BPTT gradient (why RNNs vanish/explode):
$$ \frac{\partial\mathcal L_t}{\partial W_{hh}}=\sum_{k=1}^{t}\frac{\partial\mathcal L_t}{\partial\hat y_t}\frac{\partial\hat y_t}{\partial h_t}\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)\frac{\partial h_k}{\partial W_{hh}} $$
```python
h_t = torch.tanh(self.W_hx(x_t) + self.W_hh(h_prev))   # one recurrent step
```

### 5.4 LSTM / GRU — the gates that fixed vanishing gradients
$$ f_t=\sigma(W_f[h_{t-1},x_t]+b_f) \quad i_t=\sigma(W_i[\cdot]+b_i) \quad \tilde C_t=\tanh(W_C[\cdot]+b_C) $$
$$ C_t = f_t*C_{t-1}+i_t*\tilde C_t \quad(\text{addition here} \Rightarrow \text{gradients flow cleanly}) $$
$$ o_t=\sigma(W_o[\cdot]+b_o) \qquad h_t=o_t*\tanh(C_t) $$
GRU (2 gates instead of 3): $z_t=\sigma(W_z[\cdot])$ (update), $r_t=\sigma(W_r[\cdot])$ (reset), $h_t=(1-z_t)*h_{t-1}+z_t*\tilde h_t$.
```python
f_t = torch.sigmoid(self.W_f(torch.cat([h_prev, x_t], dim=-1)))
C_t = f_t * C_prev + i_t * torch.tanh(self.W_C(...))
```

### 5.5 Autoencoders & VAE
$$ \mathcal L = -\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big) $$
**Reparameterization trick** (makes sampling differentiable):
$$ z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,1) $$
$$ D_{KL} = -\frac12\sum_j\big(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\big) $$
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### 5.6 GANs
$$ \min_G\max_D V(D,G)=\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))] $$
```python
loss_D = -torch.log(D(x)) - torch.log(1 - D(G(z)))
loss_G = -torch.log(D(G(z)))            # non-saturating trick: maximize log D(G(z))
```

### 5.7 Diffusion Models
Forward (fixed, no weights): $q(x_t|x_0)=\mathcal N(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$, i.e.
$$ x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon $$
Reverse (learned): predict the **noise**, not the image:
$$ \mathcal L_{simple}(\theta)=\mathbb E_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right] $$
Latent Diffusion = run this in a VAE's compressed latent space instead of pixel space.
```python
noise = torch.randn_like(x_0)
x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * noise
predicted_noise = unet(x_t, t)
loss = F.mse_loss(predicted_noise, noise)
```

### 5.8 GNN (Graph Neural Networks)
$$ h_v^{(k)}=\text{UPDATE}\big(h_v^{(k-1)},\text{AGGREGATE}(\{h_u^{(k-1)}:u\in\mathcal N(v)\})\big) $$
GCN layer form:
$$ H^{(k)}=\sigma\big(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(k-1)}W^{(k-1)}\big),\qquad \tilde A=A+I $$
```python
A_hat = A + torch.eye(A.size(0))                 # add self-loops
D_inv_sqrt = torch.diag(torch.pow(A_hat.sum(1), -0.5))
H_next = F.relu(D_inv_sqrt @ A_hat @ D_inv_sqrt @ H @ W)
```

### 5.9 SNN (Spiking Neural Networks) — LIF neuron
$$ \tau_m\frac{dV_m(t)}{dt}=-[V_m(t)-V_{rest}]+R\,I(t) \qquad \text{if } V_m\ge V_{th}:\ \text{spike, reset } V_m\leftarrow V_{reset} $$
Non-differentiable spike → needs a **surrogate gradient** in the backward pass.
```python
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_membrane, threshold): return (voltage_membrane >= threshold).float()
    @staticmethod
    def backward(ctx, grad_output): return grad_output * torch.sigmoid_derivative_approx(...)
```

### 5.10 LNN (Liquid Neural Networks) — continuous-time, input-dependent weights
$$ \frac{dx_i(t)}{dt}= -\Big(\frac1{\tau_i}+f(x,I,\theta_i)\Big)x_i(t)+A(x,I,\theta_i) $$
Trained via an ODE solver (Euler/Runge-Kutta forward, **Adjoint Sensitivity Method** backward — avoids storing all intermediate states).

### 5.11 RBM / Deep Belief Networks — energy-based
$$ E(v,h)=-\sum_i a_iv_i-\sum_j b_jh_j-\sum_{i,j}v_iW_{ij}h_j \qquad P(v,h)=\frac{1}{Z}e^{-E(v,h)} $$
$Z$ (partition function) is intractable → train with **Contrastive Divergence**:
$$ \Delta W_{ij}=\eta\big(\langle v_ih_j\rangle_{data}-\langle v_ih_j\rangle_{reconstruction}\big) $$
```python
h_prob = torch.sigmoid(F.linear(v, self.W, self.b_h))
h_sample = torch.bernoulli(h_prob)          # stochastic binary sampling
```

### 5.12 SOM / Kohonen Networks — unsupervised, competitive learning
$$ c=\arg\min_v\|x(t)-W_v(t)\| \qquad(\text{Best Matching Unit}) $$
$$ \theta(c,v,t)=\exp\!\left(-\frac{\|r_c-r_v\|^2}{2\sigma^2(t)}\right) \qquad W_v(t{+}1)=W_v(t)+\theta\cdot\alpha(t)\cdot[x(t)-W_v(t)] $$
```python
distances = torch.sum((self.weights - x)**2, dim=-1)
bmu_index = torch.argmin(distances)         # find winning node
```

---

## 6. Frontier & Exotic Architectures
*(Source: `Deep_Dives/16–21`, `Code/Deep_Dives/16–21`)*

### 6.1 Omnimodal Transformers
Text, image patches (ViT-style, $16\times16$), and audio are all projected to the **same dimension** (e.g. 4096) and concatenated into one sequence — the backbone Transformer then just does standard attention over a mixed-modality token stream.
$$ X=[T;V;A] \qquad \text{Attention}(XW_Q, XW_K, XW_V) $$

### 6.2 Agentic Reasoning (test-time "thinking")
$$ \mathcal L^{CLIP}(\theta)=\hat{\mathbb E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta),1{-}\epsilon,1{+}\epsilon)\hat A_t\big)\Big] $$
This is the **PPO clipped objective** — used to reward the model for generating useful hidden `<thought>` tokens before answering.

### 6.3 JEPA (Joint Embedding Predictive Architecture)
$$ s_x=E_x(x)\quad s_y=E_y(y)\quad \hat s_y=P(s_x,z) \qquad \mathcal L=\|\hat s_y-s_y\|_2^2 $$
Anti-collapse trick: **stop-gradient** through $E_y$ + **EMA weight update**:
$$ \theta_y \leftarrow \tau\theta_y+(1-\tau)\theta_x $$
```python
with torch.no_grad():
    for p_y, p_x in zip(target_encoder.parameters(), context_encoder.parameters()):
        p_y.data = tau * p_y.data + (1 - tau) * p_x.data   # EMA update, no gradient
```

### 6.4 Neurosymbolic AI (differentiable logic)
$$ \text{AND: } P\cdot Q \qquad \text{OR: } P+Q-PQ \qquad \text{NOT: } 1-P $$
(Product t-norm fuzzy logic — lets discrete logical rules be backpropagated through.)

### 6.5 KAN (Kolmogorov-Arnold Networks)
Standard MLP: fixed activation on a weighted sum, $y=\text{ReLU}(w_1x_1+w_2x_2)$.
KAN: **learnable spline function on each edge**, summed: $y=\phi_1(x_1)+\phi_2(x_2)$. Gradient descent moves spline control points, not scalar weights → interpretable (you can literally read off the learned formula).
```python
class SimplifiedKANLayer(nn.Module):
    def forward(self, x):
        # x -> evaluate learnable spline (via einsum over grid/basis) per edge, then sum
        return torch.einsum('...i,io->...o', spline_basis(x), self.spline_weights)
```

### 6.6 Active Inference
$$ F = D_{KL}[Q(s)\|P(s|o)] - \log P(o) = \text{Complexity} - \text{Accuracy} $$
Agent acts to minimize **Variational Free Energy** — naturally balances exploration (resolving uncertainty) and exploitation, no epsilon-greedy needed.

### 6.7 Test-Time Scaling & Process Reward Models (PRM)
$$ V(s_t,a_t)\in[-1,1] \qquad(\text{PRM scores each reasoning step}) $$
If $V<0$ → prune that branch of the Tree-of-Thoughts search. Three strategies: **Parallel** (self-consistency / pass@K), **Sequential** (tree search / MCTS), **Hybrid/Adaptive**.

### 6.8 SOTA Models, Multi-Agent Systems & Emerging Paradigms (conceptual — no equations in source)
*(Source: `Reports/03_Transformers_and_SOTA_Models.md`, `07 §17–19`, `05_Future_Of_AI.md`)*
| Topic | One-line summary |
|---|---|
| Proprietary SOTA (2024–26) | Anthropic Claude 3.5 (Sonnet, Opus), OpenAI (GPT-4o, o1/o3), Google Gemini (1.5, 2.0) — closed, RLHF'd, agentic-reasoning-capable |
| Open-source SOTA | Llama (Meta), DeepSeek, Nemotron (NVIDIA) — openly-weighted, often MoE-based |
| Multi-Agent & Orchestration | mixture of *specialized* agents (planner/coder/verifier) that talk to each other, vs. one monolithic model |
| Large Concept Models (LCMs) | trained to reason over structured causal relationships between concepts, not just next-token text |
| Large Action Models (LAMs) | optimized to output API calls / UI actions / executable workflows — the "hands" of an agentic system |
| Knowledge Graphs (KGs) | Nodes+Edges graph of facts; LLM writes a graph query (e.g. Cypher) for 100%-accurate lookups vs. RAG's approximate vector search — combining both = GraphRAG |
| Organoid Intelligence (OI) | experimental: lab-grown human brain-cell cultures wired to microelectrode arrays as ultra-low-power biological compute (20W vs. GPU wattage) |
| Quantum Machine Learning (QML) | using quantum circuits for the matrix-multiplication core of NNs; theoretical exponential speedups, currently hardware-noise-limited |

### 6.9 AI for Science & Physical Systems (PINNs, FNO, EGNN, Evoformer, EDA, NQS, Symbolic)
*(Source: `Deep_Dives/45–51`, `Code/Deep_Dives/45–51`)*

| Model / Concept | Canonical Formula | Meaning & Code Idiom |
|---|---|---|
| **PINNs (45)** | $\mathcal L_{\text{PDE}}=\|u_t+u u_x-\nu u_{xx}\|^2$ | Continuous mesh-free physics; exact analytical spatial/temporal derivatives via Autograd (`create_graph=True`). |
| **FNO (46)** | $(\mathcal K v)(x) = \mathcal F^{-1}(R_\theta\cdot\mathcal F(v))(x)$ | Infinite-dimensional operator learning; global spatial convolutions in frequency domain via `torch.fft.rfft2` with zero-shot mesh super-resolution. |
| **EGNN (47)** | $x_i^{(l+1)}=x_i^{(l)}+\sum_j(x_i-x_j)\phi_x(m_{ij})$ | $E(3)$ rotational/translational invariance for molecular scalar energy ($E$) and exact equivariance for 3D atomic forces ($\vec F$). |
| **Evoformer (48)** | $Z_{ij}\leftarrow Z_{ij}+\sum_k a_{ik}\odot b_{jk}$ | AlphaFold 2/3: outer-product evolutionary MSA injection + triangular multiplicative update enforcing 3D Euclidean triangle inequality. |
| **Netlist EDA (49)** | $\text{LSE}_x(e)=\gamma\ln\sum e^{x_i/\gamma}-\gamma\ln\sum e^{-x_i/\gamma}$ | Microchip floorplanning; differentiable Log-Sum-Exp approximation of Half-Perimeter Wire Length (HPWL) on bipartite netlist graphs. |
| **Quantum NQS (50)** | $E_{\text{loc}}(S)=\frac{\langle S\|\hat H\|\Psi_\theta\rangle}{\psi_\theta(S)}$ | Variational Monte Carlo (VMC) solving $2^N$ exponential Hilbert space spin wavefunctions using neural policy gradients without explicit matrix diagonalization. |
| **Symbolic AI (51)** | $\mathcal L = \text{MSE} + \lambda \|W\|_1$ | Lean 4 formal tactic verification + $L_1$ regularized symbolic regression discovering human-readable closed-form physical equations ($y=f(x)$). |

---

## 7. Training, Alignment & Fine-Tuning
*(Source: `Deep_Dives/24, 38, 42`, `Code/Deep_Dives/24, 38`)*

### 7.1 RLHF actors
$\pi_\theta(y|x)$ policy · $\pi_{ref}(y|x)$ frozen reference (KL anchor) · $R_\phi(x,y)$ reward model · $V_\psi(x)$ critic.

### 7.2 DPO (Direct Preference Optimization) — RLHF without a reward model
$$ \mathcal L_{DPO}=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\!\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\right] $$
$y_w$ = preferred/winning response, $y_l$ = dispreferred/losing response.
- **KTO**: binary thumbs-up/down signal (no paired preferences needed). **ORPO/SimPO**: further alignment variants, no separate reference model.

### 7.3 PPO training loop (RLHF classic path)
```python
kl_penalty = calculate_kl_divergence(logprobs_policy, logprobs_ref)
ratio = torch.exp(new_logprobs - old_logprobs)
clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
loss = -torch.min(ratio * advantage, clipped).mean()
```

### 7.4 LoRA / DoRA / QLoRA — parameter-efficient fine-tuning
$$ W_{new} = W_0 + \frac{\alpha}{r}(A\cdot B), \quad A\in\mathbb R^{d\times r},\ B\in\mathbb R^{r\times k},\ r\ll d $$
DoRA decomposes into magnitude × direction: $W=m\dfrac{W_0+\Delta W}{\|W_0+\Delta W\|_F}$.
```python
class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16.0):
        self.weight = nn.Parameter(torch.randn(out_features, in_features)); self.weight.requires_grad = False
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_A = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = alpha / rank
    def forward(self, x):
        base = F.linear(x, self.weight)                          # frozen path
        lora = F.linear(F.linear(x, self.lora_B), self.lora_A)   # x·Bᵀ·Aᵀ, trainable
        return base + lora * self.scaling
```
**QLoRA** = quantize the frozen base to 4-bit (NF4) + attach 16-bit LoRA adapters → fine-tune a huge model on one consumer GPU.

### 7.5 Quantization
$$ \text{scale}=\frac{\max(|x|)}{127} \qquad x_q=\text{round}(x/\text{scale}) \qquad x\approx x_q\cdot\text{scale} $$
```python
scale = torch.max(torch.abs(tensor)) / 127.0
q = torch.round(tensor / scale).to(torch.int8)
recovered = q.to(torch.float32) * scale
```

### 7.6 Scaling Laws (Chinchilla)
$$ N\propto C^{0.5}, \quad D\propto C^{0.5} \qquad(N{=}\text{params}, D{=}\text{data size}, C{=}\text{compute}) $$
Parameters and data should scale **equally** with compute for optimal loss (not "just add more parameters").

---

## 8. Systems, Distributed Training & Data
*(Source: `Deep_Dives/28, 29, 32, 33, 35, 39, 41`)*

- **Tokenization (BPE)**: iteratively merge the most frequent adjacent symbol pair — `"t"+"h"→"th"`, `"th"+"e"→"the"`.
- **MinHash**: approximate document similarity in linear time instead of $O(N^2)$ pairwise comparison — used for dedup at web scale.
- **ZeRO (DeepSpeed)**: shards optimizer states / gradients / parameters across GPUs instead of replicating them, so trillion-parameter models fit in aggregate GPU memory.
- **Mechanistic Interpretability**: probing/steering internal activations ("clamping" specific directions) to understand or control model behavior.
- **Benchmarking**: measure Memory (VRAM peak), Latency/Throughput, and FLOPs — plot Memory vs. Sequence Length; a flat line = you found an $O(N)$ architecture, a parabola = still $O(N^2)$.
```python
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
peak_mem = torch.cuda.max_memory_allocated()
```
- **Vector DB / HNSW**: Cosine Similarity $=\dfrac{A\cdot B}{\|A\|\|B\|}$; HNSW builds layered graphs so search drops from $O(N)$ to $O(\log N)$.
- **RAG (Reciprocal Rank Fusion, hybrid search)**:
$$ RRF\_Score(d) = \sum_{m\in M}\frac{1}{k+r_m(d)} \qquad \text{Faithfulness}=\frac{\text{claims supported by context}}{\text{total claims}} $$

---

## 9. Python/PyTorch/CUDA/Triton Keyword Glossary
*(Source: `Deep_Dives/40_Python_PyTorch_and_CUDA_Execution_Glossary.md` — the single densest reference file in the whole archive)*

### 9.1 Python mechanics you'll see constantly
| Keyword | Meaning |
|---|---|
| `__init__(self, ...)` | class constructor — allocates layers/params |
| `__call__` | lets `model(x)` work; **never call `model.forward(x)` directly** — skips PyTorch's hooks |
| `__getitem__` / `__len__` | required for custom `torch.utils.data.Dataset` |
| `__enter__` / `__exit__` | context manager protocol, e.g. `with torch.no_grad():` |
| `@torch.no_grad()` | disables autograd graph construction → saves VRAM for inference |
| `@triton.jit` | compiles a Python function into a native CUDA kernel |
| `@property` | read-only computed attribute |
| `*args`, `**kwargs` | variadic positional / keyword args, used heavily in generic layer wrappers |

### 9.2 Tensor & Module core
| Call | Meaning |
|---|---|
| `nn.Module` | base class for every layer/model |
| `nn.Parameter` | tensor auto-registered as trainable in `.parameters()` |
| `requires_grad=True/False` | toggles whether autograd tracks a tensor (LoRA freezes base weights this way) |
| `tensor.stride()` | memory step-size per dimension |
| `tensor.contiguous()` | force physical memory to match logical shape (needed after `transpose`/`permute`) |
| `tensor.view()` vs `.reshape()` | `view` needs contiguous memory; `reshape` copies if not contiguous |
| `tensor.permute(*dims)` | reorder multiple dims at once |
| `torch.einsum(eq, *tensors)` | Einstein summation — the workhorse of every attention implementation |
| `torch.gather` | pull values by index (MoE top-k routing) |
| `torch.scatter_` / `index_add_` | write/accumulate values by index (MoE output combination) |
| `torch.topk` | select top-K values+indices (MoE routing, speculative decoding) |
| `torch.clamp(x, min, max)` | cap values (PPO surrogate clipping) |
| `torch.masked_fill(mask, value)` | set masked positions (causal masking) |
| `torch.cumprod`, `torch.outer`, `torch.einsum` | used in SSM discretization / diffusion schedules |

### 9.3 `torch.compile` pipeline (PyTorch 2.x)
`torch.compile(model)` →
1. **TorchDynamo** — safely captures the computation graph from Python bytecode.
2. **AOTAutograd** — captures forward *and* backward graphs ahead-of-time.
3. **TorchInductor** — compiles to fused C++/Triton GPU kernels (e.g. fuses `Add+LayerNorm+ReLU` into one kernel).

### 9.4 Triton / CUDA primitives (writing a custom kernel, e.g. FlashAttention)
| Primitive | Meaning |
|---|---|
| `tl.constexpr` | compile-time constant inside a kernel (e.g. `BLOCK_SIZE_M: tl.constexpr = 64`) |
| `tl.load(ptr+offsets, mask)` | vectorized HBM → SRAM read |
| `tl.store(ptr+offsets, val, mask)` | vectorized SRAM → HBM write |
| `tl.dot(a, b)` | matrix multiply inside SRAM |
| `grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), ...)` | defines GPU thread-block launch grid |

### 9.5 When to reach for the PyTorch wrapper vs. build custom
| Concept | High-level wrapper | Why go custom? |
|---|---|---|
| Attention | `nn.MultiheadAttention` | need RoPE, KV cache, custom masks |
| Recurrence | `nn.LSTM` | non-standard gating (e.g. xLSTM matrix memory) |
| Quantization | `torch.quantization` | full control of INT4/NF4 scale/zero-point |
| LoRA | `peft.LoraModel` | exact gradient-flow visibility |

---

## 10. Libraries & Frameworks Quick Reference
*(Source: `Reports/09_Complete_ML_DL_AI_Libraries_Master_Guide.md`, `Deep_Dives/41, 43, 44`)*

**PyTorch 2.x production training step (the loop you'll rewrite 1000 times):**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model = torch.compile(model)                      # PT2.x graph compilation
scaler = torch.cuda.amp.GradScaler()
for x, y in dataloader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # AMP mixed precision
        loss = F.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer); scaler.update()
```

**Scikit-learn classical ML pipeline:**
```python
pipe = Pipeline([("prep", ColumnTransformer([...])), ("clf", RandomForestClassifier())])
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
```

**Hugging Face + PEFT/QLoRA fine-tuning:**
```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"]))
```

**LangGraph (stateful multi-agent graphs):**
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("action", tool_execution_node)
workflow.add_conditional_edges("agent", should_continue_router, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
app = workflow.compile(checkpointer=MemorySaver())
```

**vLLM serving (PagedAttention):**
```python
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1)
outputs = llm.generate(["Explain quantum superposition."], sampling_params)
```
PagedAttention's actual trick — KV cache lives in **non-contiguous physical blocks** (like OS virtual memory), so VRAM never needs one giant contiguous allocation:
```python
class VirtualPagedKVCache:
    def allocate_tokens(self, request_id, num_tokens):
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.request_block_tables[request_id] = allocated   # per-request block table
```

**Hugging Face tokenizer & chat templates:**
```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    tokenize=False, add_generation_prompt=True)
```

**TRL trainers (alignment on top of §7.2–7.3):** `SFTTrainer` (supervised fine-tune on instructions) · `DPOTrainer` (DPO loss from §7.2, no reward model needed) · PPO trainer (full RLHF loop from §7.3).

**Keras 3 (multi-backend — same code runs on JAX/TF/PyTorch):**
```python
os.environ["KERAS_BACKEND"] = "torch"   # or "jax", "tensorflow"
class MultiBackendDenseResidual(keras.Model):
    def call(self, inputs, training=False):
        x = self.dense1(inputs); x_norm = self.norm(x)
        return self.out(ops.add(x, self.dense2(x_norm)))   # keras.ops is backend-agnostic
```

**JAX / Flax (pure functional, TPU-native):**
```python
grad_fn = jax.jit(jax.value_and_grad(loss_fn))   # jax.grad = autodiff, jax.jit = XLA compile
# jax.vmap  -> auto-vectorize over batch dim      # jax.pmap -> shard across devices
```

**Tabular boosting (non-neural, still dominant on structured data):**
| Library | Tree-growth strategy | Key trick |
|---|---|---|
| XGBoost | depth-wise (level-wise) | exact greedy + histogram pruning |
| LightGBM | leaf-wise (best-first) | GOSS gradient sampling + EFB |
| CatBoost | symmetric/oblivious | ordered boosting (no target leakage) |

**Decision cheat-sheet:**
- Tabular/classical → **Scikit-Learn** + **XGBoost/LightGBM**
- Deep learning research → **PyTorch 2.x** + `torch.compile`
- Multi-backend/mobile → **Keras 3** + TF Lite
- LLM fine-tuning → **Hugging Face** (`transformers`, `peft`, `trl`)
- Serving → **vLLM** (GPU, PagedAttention) / **llama.cpp** (CPU/edge, GGUF)
- Agent orchestration → **LangGraph**; observability → **LangSmith** / **Langfuse**

---

## 11. Master Equation Index (alphabetical)
| Concept | Formula | §  |
|---|---|---|
| Adam optimizer | $W\leftarrow W-\alpha\, v_{dW}/(\sqrt{s_{dW}}+\epsilon)$ | 1.5 |
| ALiBi | subtract distance-penalty from attention scores | 3 |
| Attention (scaled dot-product) | $\text{softmax}(QK^T/\sqrt{d_k})V$ | 2.2 |
| BPTT gradient | $\prod_j \partial h_j/\partial h_{j-1}$ | 5.3 |
| Contrastive Divergence | $\Delta W=\eta(\langle vh\rangle_{data}-\langle vh\rangle_{recon})$ | 5.11 |
| Convolution | $S(i,j)=\sum\sum I(i+m,j+n)K(m,n)$ | 5.2 |
| Cosine similarity | $A\cdot B/(\|A\|\|B\|)$ | 8 |
| Cross-entropy loss | $-\sum y_c\log\hat y_c$ | 1.3 |
| Diffusion forward process | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ | 5.7 |
| DPO loss | $-\mathbb E[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})]$ | 7.2 |
| GAN minimax | $\min_G\max_D \mathbb E[\log D(x)]+\mathbb E[\log(1-D(G(z)))]$ | 5.6 |
| GCN layer | $\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)$ | 5.8 |
| JEPA loss | $\|\hat s_y-s_y\|_2^2$ | 6.3 |
| KL Divergence | $\sum P(x)\log(P(x)/Q(x))$ | 1.7 |
| KAN edge function | $y=\phi_1(x_1)+\phi_2(x_2)$ | 6.5 |
| LayerNorm | $\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$ | 1.6 |
| LIF neuron | $\tau_m dV/dt = -[V-V_{rest}]+RI(t)$ | 5.9 |
| LoRA | $W_0+\frac{\alpha}{r}(A\cdot B)$ | 7.4 |
| LSTM cell state update | $C_t=f_t*C_{t-1}+i_t*\tilde C_t$ | 5.4 |
| MoE routing | $y=\sum G_{sparse}(x)_i\cdot E_i(x)$ | 4.1 |
| PPO clipped objective | $\min(r_t\hat A_t, \text{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t)$ | 6.2, 7.3 |
| Quantization | $x_q=\text{round}(x/\text{scale})$ | 7.5 |
| RBM energy | $E(v,h)=-\sum a_iv_i-\sum b_jh_j-\sum v_iW_{ij}h_j$ | 5.11 |
| Reparameterization trick | $z=\mu+\sigma\odot\epsilon$ | 5.5 |
| RoPE rotation | $q_m=q\,e^{im\theta}$ | 3 |
| Scaling laws | $N\propto C^{0.5},\ D\propto C^{0.5}$ | 7.6 |
| SOM weight update | $W_v(t{+}1)=W_v(t)+\theta\cdot\alpha(t)[x(t)-W_v(t)]$ | 5.12 |
| SSM discretization (ZOH) | $\bar A=\exp(\Delta A)$ | 4.2 |
| SwiGLU | $\text{SiLU}(xW_{gate})\odot(xW_{up})\,W_{down}$ | 2.4 |
| Variational Free Energy | $F=D_{KL}[Q\|P]-\log P(o)$ | 6.6 |

---

## 12. Master Code Keyword Index (alphabetical)
| Keyword/Call | One-liner |
|---|---|
| `F.cross_entropy` | log_softmax + NLL loss combined |
| `F.silu` | SwiGLU-family activation: $x\cdot\sigma(x)$ |
| `index_add_` | scatter-accumulate values by index (MoE combine step) |
| `nn.LayerNorm` | per-token feature normalization |
| `nn.ModuleList` | list of sub-modules PyTorch tracks correctly (vs. plain Python list) |
| `nn.Parameter` | tensor that auto-registers as trainable |
| `requires_grad = False` | freeze a tensor (LoRA base weights, EMA target encoders) |
| `torch.autograd.Function` | define custom forward/backward (surrogate gradients for SNN spikes) |
| `torch.bernoulli` | sample binary 0/1 (RBM stochastic units) |
| `torch.compile` | PT2.x graph compilation → fused Triton/C++ kernels |
| `torch.cuda.amp.autocast` | mixed-precision context manager |
| `torch.einsum` | Einstein summation notation for tensor contraction (attention, KAN, SSM) |
| `torch.masked_fill` | apply causal/padding masks pre-softmax |
| `torch.no_grad()` | disable autograd graph building (inference, EMA updates) |
| `torch.topk` | top-K selection (MoE routing, speculative decoding) |
| `triton.jit` | compile Python → CUDA kernel |
| `tl.load` / `tl.store` | HBM↔SRAM vectorized memory ops inside a Triton kernel |
| `F.silu` + gate*up | SwiGLU gated feed-forward (modern FFN, replaces plain ReLU MLP) |
| `jax.jit` / `jax.grad` / `jax.vmap` / `jax.pmap` | JAX: XLA-compile / autodiff / auto-batch / multi-device shard |
| `tokenizer.apply_chat_template` | HF: formats a message list into a model's prompt format |
| `np.where(x>=0, 1, 0)` | step-function activation (the original Perceptron) |

---

## 13. What's in the zip but NOT in this cheat sheet
This document deliberately covers **equations + code + the load-bearing conceptual context around them**. After a second full audit pass (checking every Report and Deep Dive against the draft, not just the ones with `$$...$$` blocks), the following gaps were found and patched into the sections above: the literal Perceptron/step-function + pure-NumPy backprop code (§5.1), SwiGLU — the FFN real LLMs actually use instead of plain ReLU-MLP (§2.4), PagedAttention's block-allocator (§10), HF chat templates, TRL trainers, Keras 3, JAX/Flax, and XGBoost/LightGBM/CatBoost (§10), and the non-equation conceptual material — SOTA proprietary/open models, Multi-Agent/LCM/LAM, Knowledge Graphs, Organoid Intelligence, QML (§6.8).

What's still intentionally **not** reproduced here:
- `STUDY_ROADMAP.md`, `SELF_ASSESSMENT_QUIZ.md` — a learning path and quiz, not reference material.
- `PAPERS_TO_READ_CATALOG.md`, `FUTURE_RESEARCH_MAP.md`, `non-agentic-ai-research-directions.md` — reading lists / research direction notes, no equations or code.
- `LLM_DEBUGGING_AND_TROUBLESHOOTING.md` — a practical debugging playbook, different genre from a math/code cheat sheet (worth a *second*, separate cheat sheet if useful — say the word).
- `AI_Architecture_Research_Breakdown.html`, `neural_codex (1).html`, `report (1).html`, `math_mastery_transformer.html` — narrative/HTML versions of material already condensed above; useful for the long-form explanation behind any formula here, but not re-summarized to avoid duplicating §1–9 verbatim.
- `Code/verify_all.py` — an environment/GPU-check test harness, not architecture code.

---

## 14. Self-check: answering your 4 questions

**Will this work like a cheat sheet?** Yes — every entry is `formula → one-line meaning → runnable code idiom`, with two alphabetical indices (§11 equations, §12 code) for fast lookup, and a jump table at the top. No prose you have to re-read to extract the formula.

**Does it cover all parts?** It covers every equation-bearing file: all 9 core `Reports/`, all 44 `Deep_Dives/` reports, and the real code from all 41 `.py` files (core Transformer files `03–05`, and `Code/Deep_Dives/01–44`). §13 lists the handful of non-technical files (roadmap, quiz, reading lists) intentionally left out, so you know exactly what's summarized vs. what to open directly.

**Does it show all math equations most used?** Every equation extracted from the archive's `$$...$$` / `$...$` blocks was reviewed; the ones kept are the *canonical* form of each concept (dropping only redundant duplicate/restated equations across files, e.g. the same attention formula appears in 4 different reports — it's here once, in §2.2, cross-referenced in §11).

**Good enough to remember & understand the whole zip?** For recall and pattern-matching, yes — this is built to answer "wait, what was that formula/line called again?" in seconds. It is *not* a substitute for the worked numerical examples in `Deep_Dives/00` or the full narrative reasoning in the HTML files when you're learning a concept for the first time — use this **after** first-pass understanding, as your fast-reference layer.

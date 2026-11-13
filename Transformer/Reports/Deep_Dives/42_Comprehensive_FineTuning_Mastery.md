# Deep Dive 42: Comprehensive Fine-Tuning & PEFT Mastery

Fine-tuning adapts pre-trained base models to specialized tasks, specific output formats, or custom domain styles. This report details the full taxonomy of fine-tuning, parameter-efficient adapters, preference optimization, and training compute optimizations.

---

## 1. The Fine-Tuning Taxonomy

```
                      ┌─────────────────────────────────────────┐
                      │        FINE-TUNING METHODOLOGY MAP      │
                      └─────────────────────────────────────────┘
  [Pre-Trained Base Model]
          │
          ├──► Supervised Fine-Tuning (SFT) ──► Teaches formatting, domain tone & tool use
          │
          └──► Preference Alignment ───────► Aligns outputs with human values & instruction fidelity
                     ├──► RLHF (PPO)        (Policy + Reward + Value + Ref Model)
                     ├──► DPO               (Implicit reward via log-ratio loss)
                     ├──► KTO               (Binary positive/negative signal tuning)
                     └──► ORPO / SimPO      (Monolithic reference-free preference loss)
```

---

## 2. Preference Alignment Paradigms

### 2.1 RLHF with PPO (Proximal Policy Optimization)
Uses 4 active neural networks in VRAM:
1.  **Actor/Policy Model:** $\pi_\theta(y|x)$ (Model being trained).
2.  **Reference Model:** $\pi_{ref}(y|x)$ (Frozen base model, enforces KL divergence penalty so policy doesn't collapse).
3.  **Reward Model:** $R_\phi(x, y)$ (Scores completion quality).
4.  **Critic/Value Model:** $V_\psi(x)$ (Predicts expected total reward for advantage calculation).

### 2.2 DPO (Direct Preference Optimization)
Eliminates the Reward Model and Critic Model entirely. Mathematically proves that the policy model itself can act as its own implicit reward model:
$$ \mathcal{L}_{DPO}(\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right] $$
where $y_w$ is the winning (preferred) response and $y_l$ is the losing (dispreferred) response.

### 2.3 Emerging Alignment: KTO, ORPO & SimPO
*   **KTO (Kahneman-Tversky Optimization):** Works with binary signals (Yes/No thumbs up/down) instead of paired preferences ($y_w, y_l$). Based on prospect theory.
*   **ORPO (Odds Ratio Preference Optimization):** Combines SFT cross-entropy loss with an odds-ratio penalty in a single training step, eliminating the need for a reference model.
*   **SimPO (Simple Preference Optimization):** Reference-free alignment method that directly uses average log likelihood as implicit reward with a target length penalty margin.

---

## 3. PEFT (Parameter-Efficient Fine-Tuning) Variants

```
Method             Trainable Parameters    VRAM Reduction     Mechanism
-----------------------------------------------------------------------------------------------------
Full Fine-Tuning   100%                    0%                 Updates all weights W
LoRA               0.01% - 0.1%            60% - 75%          Injected low-rank matrices A @ B
QLoRA              0.01% - 0.1%            85% - 90%          LoRA on 4-bit NF4 quantized base model
DoRA               0.01% - 0.2%            60% - 75%          Decomposes W into Magnitude + Direction
Prefix Tuning      0.1%                    70%                Prepends trainable KV vectors to layers
```

### 3.1 LoRA & DoRA Mechanics
*   **LoRA (Low-Rank Adaptation):** $W_{new} = W_0 + \frac{\alpha}{r} (A \cdot B)$, where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ with rank $r \ll d$.
*   **DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposes weight matrix into magnitude vector $m$ and directional matrix $V$:
    $$ W = m \frac{V}{||V||_F} = m \frac{W_0 + \Delta W}{||W_0 + \Delta W||_F} $$
    Improves learning stability by matching full fine-tuning direction/magnitude update dynamics.

---

## 4. Hardware Compute & Memory Optimizations

```
Optimization Technique     Memory Reduction     Mechanism
-------------------------------------------------------------------------------------------
Gradient Accumulation      Enables large batch  Splits batch into micro-batches; accumulates 
                           sizes on 1 GPU       gradients before optimizer step
Gradient Checkpointing    30% - 50% VRAM       Discards intermediate activations during forward 
                           savings              pass; recomputes them during backward pass
FlashAttention-2 / 3       50% - 80% VRAM       Fuses attention computation into SRAM blocks;
                           speedup 2-4x         avoids $O(N^2)$ HBM allocations
Mixed Precision (BF16)    50% Memory           Uses 16-bit Bfloat16 for activations & weights;
                           2x Speedup           retains FP32 range without overflow
```

### 4.1 Quantization Formats & Conversion Tools
*   **NF4 (NormalFloat 4):** Quantization data type optimized for normally distributed neural network weights (used in QLoRA).
*   **AWQ (Activation-aware Weight Quantization):** Protects the top 1% salient weight channels based on activation magnitudes.
*   **GPTQ:** One-shot post-training quantization based on second-order Taylor expansion error compensation.
*   **GGUF:** Binary file format designed for fast execution on CPUs and consumer GPUs via `llama.cpp`.

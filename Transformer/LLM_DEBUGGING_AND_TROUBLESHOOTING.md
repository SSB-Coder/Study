# Real-World ML & LLM Debugging Playbook

When building neural networks, training transformers, or deploying LLMs, errors will happen. This operational playbook provides a diagnostic decision tree and precise code fixes for the top real-world ML engineering failures.

---

## 1. Memory & Hardware Failures

### 1.1 CUDA Out Of Memory (OOM) Errors

#### Symptom
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB...`

#### Diagnostic Decision Tree
```
                    [ CUDA Out of Memory Error ]
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
[ Occurs in Batch 0 / Init ]                 [ Occurs Mid-Epoch / Step N ]
         │                                               │
 ┌───────┴───────┐                              ┌────────┴────────┐
 ▼               ▼                              ▼                 ▼
Model too big   Batch size too large         Memory Leak       KV Cache / Seq 
for 1 GPU       for VRAM                     (Tracking tensor  Length Spikes
                                              with history)
```

#### Code Fixes
1.  **Enable Gradient Checkpointing:**
    ```python
    model.gradient_checkpointing_enable()  # Trades 20% compute time for 60% VRAM reduction
    ```
2.  **Use Mixed Precision (`torch.cuda.amp` or Bfloat16):**
    ```python
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
    ```
3.  **Implement Gradient Accumulation:**
    ```python
    # Instead of batch_size=64, run micro_batch_size=8 over 8 steps
    optimizer.zero_grad()
    for micro_step in range(accumulation_steps):
        outputs = model(inputs_micro)
        loss = loss_fn(outputs, targets) / accumulation_steps
        loss.backward()  # Accumulates gradients
    optimizer.step()
    ```
4.  **Fix Python Memory Leaks:** Avoid appending loss tensors directly to a list:
    *   ❌ **WRONG:** `losses.append(loss)` (Holds computation graph in VRAM forever).
    *   ✅ **CORRECT:** `losses.append(loss.item())` or `losses.append(loss.detach())`.

---

## 2. Numerical Instability & Gradient Explosions

### 2.1 NaN / Inf Loss Explosion

#### Symptom
Loss outputs `nan` or `inf` during step 500 of training. Model weights turn into `nan`.

#### Causes & Fixes
1.  **Exploding Gradients:** Clip gradient norm before optimizer step:
    ```python
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ```
2.  **Softmax Logarithm Underflow:** Taking `torch.log(x)` where $x=0$ yields $-\infty$.
    *   Add a small $\epsilon$ stability constant: `torch.log(x + 1e-8)`.
3.  **FP16 Underflow/Overflow:** FP16 has max value 65,504. Switching to **Bfloat16** (same exponent bits as FP32) prevents overflow without precision tuning.
4.  **Learning Rate Too High:** Use a Learning Rate Warmup schedule:
    ```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    ```

---

## 3. Layer Alignment & Matrix Shape Errors

### 3.1 Dimension Mismatch in Attention / Linear Layers

#### Symptom
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 256x512)`

#### Diagnostic Checklist
1.  Check inner dimensions: Matrix multiplication $(A \times B) \cdot (B \times C)$ requires inner dimension $B$ to match exactly.
2.  Check stride contiguity after `permute()` or `transpose()`:
    *   ❌ **WRONG:** `x.permute(0, 2, 1, 3).view(batch, seq, hidden)`
    *   ✅ **CORRECT:** `x.permute(0, 2, 1, 3).contiguous().view(batch, seq, hidden)`
3.  Debug shapes inline with `print(x.shape)` or using PyTorch autograd shape hooks.

---

## 4. Fine-Tuning & Adapter Issues

### 4.1 LoRA Rank Collapse / Zero Learning

#### Symptom
LoRA fine-tuning completes, but model performance is identical to base model or outputs total gibberish.

#### Fixes
1.  **Check Initialization of Matrix A & B:**
    *   Matrix $A$ must be initialized with Gaussian noise (`torch.randn`).
    *   Matrix $B$ must be initialized to **zeros** (`torch.zeros`) so that at step 0, $\Delta W = B \cdot A = 0$.
2.  **Verify Target Modules:** Ensure LoRA is applied to projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`), not just the feed-forward network.
3.  **Check LoRA Scaling Constant:** Ensure $\alpha / r$ is set properly (e.g., $r=8, \alpha=16$).

---

## 5. Tokenizer & Generation Bugs

### 5.1 Model Repetition Loop ("The Endless Echo")

#### Symptom
Model repeats the same sentence over and over: *"The answer is yes. The answer is yes. The answer is yes."*

#### Fixes
1.  **Apply Repetition Penalty during sampling:**
    ```python
    repetition_penalty = 1.2  # Penalizes previously generated token IDs
    ```
2.  **Temperature & Top-P (Nucleus) Tuning:** Set `temperature=0.7`, `top_p=0.9`. Avoid deterministic `temperature=0.0` for open-ended text.
3.  **Check End-Of-Sequence (EOS) Token:** Ensure the tokenizer's `eos_token_id` is passed to the generation loop so generation stops when the model predicts EOS.

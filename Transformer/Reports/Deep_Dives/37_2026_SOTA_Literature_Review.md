# Deep Dive: 2026 SOTA Literature Review & Architectural Mapping

The goal of this final report is to connect the massive web of mathematical theories you have learned in the previous 36 Deep Dives directly to the State-of-the-Art (SOTA) frontier models of 2024–2026. 

When you read a research paper or technical report for a new frontier model, they will no longer explain *how* Attention works. They assume you know the theory. Below is a mapping of modern top-tier foundation models and the specific architectural blueprints they rely on.

---

## 1. OpenAI: o1 / o3 & GPT-4o Series
*Released: 2024–2025*

**The Architecture:** Natively Omnimodal Tokenization + Test-Time Compute Scaling (Reasoning Chains & PRMs)
**The Deep Dives to Study:**
*   [16_Transformer_Omnimodal.md](16_Transformer_Omnimodal.md)
*   [17_Transformer_Agentic_Reasoning.md](17_Transformer_Agentic_Reasoning.md)
*   [36_Test_Time_Scaling_and_PRMs.md](36_Test_Time_Scaling_and_PRMs.md)

**Literature Review Synthesis:**
Prior models (like early GPT-4) used a pipelined approach: a text Transformer handed off audio to a separate Whisper model and vision to a separate CLIP vision encoder. GPT-4o introduced end-to-end **native omnimodality**, projecting audio waveforms, pixels, and text tokens directly into the same latent embedding space inside the Transformer (DD16).
Furthermore, the OpenAI o1 and o3 family unlocked the **test-time compute scaling frontier** (DD36). Rather than emitting immediate next tokens, these models are trained with large-scale reinforcement learning over chain-of-thought trajectories. They dynamically allocate variable test-time compute to plan, backtrack, evaluate candidate reasoning steps, and self-correct prior to producing the final response.

---

## 2. Anthropic: Claude 3.5 Sonnet, Claude 3.5 Opus & Computer Use
*Released: 2024–2025*

**The Architecture:** Hybrid Reasoning + Agentic Tool Use & Prompt Caching
**The Deep Dives to Study:**
*   [17_Transformer_Agentic_Reasoning.md](17_Transformer_Agentic_Reasoning.md)
*   [36_Test_Time_Scaling_and_PRMs.md](36_Test_Time_Scaling_and_PRMs.md)
*   [24_Scaling_Laws_and_RLHF.md](24_Scaling_Laws_and_RLHF.md)

**Literature Review Synthesis:**
Anthropic's Claude 3.5 generation established benchmarks in reasoning, coding, and multi-step tool use. The architecture incorporates native **Computer Use** (DD17), structuring the model's action space to inspect GUI screenshots, calculate coordinate deltas, and interact with software interfaces.
To make long multi-turn interactions practical, Claude leverages hardware-aware **Prompt Caching**, preserving precomputed KV-caches across API requests and cutting time-to-first-token latency. On safety and alignment (DD24), Anthropic utilizes Constitutional AI and RLHF with direct preference tuning, ensuring safety guardrails without compromising reasoning depth.

---

## 3. xAI: Grok 2 and Grok 3
*Released: 2024–2025*

**The Architecture:** Massive Sparse Mixture of Experts (MoE) + Colossus Scale Infrastructure
**The Deep Dives to Study:**
*   [14_Transformer_MoE_Sparse.md](14_Transformer_MoE_Sparse.md)
*   [32_Distributed_Training_and_ZeRO.md](32_Distributed_Training_and_ZeRO.md)

**Literature Review Synthesis:**
Trained on the Colossus cluster (100,000+ liquid-cooled NVIDIA H100 GPUs), Grok represents extreme brute-force infrastructure scaling. Grok models rely on the Sparse Mixture of Experts architecture (DD14). Rather than routing every token through a massive monolithic dense network, top-$k$ routers dynamically activate only a subset of specialized expert parameters per token, maximizing total capacity while containing inference FLOPs. Sharding model states, gradients, and optimizer memory at this magnitude requires deep distributed optimizations like DeepSpeed ZeRO-3 (DD32) and Megatron tensor/pipeline parallelism across high-speed InfiniBand/NVLink fabrics.

---

## 4. Frontier Innovators: Gemini, DeepSeek & Titans

### Gemini 1.5 & 2.0 (Google DeepMind)
*   **Architecture:** Sparse MoE + Multi-Million-Token Context via Ring Attention & Speculative Decoding
*   **Deep Dives:** [14_Transformer_MoE_Sparse.md](14_Transformer_MoE_Sparse.md), [31_Speculative_Decoding_and_MLA.md](31_Speculative_Decoding_and_MLA.md)
*   **Synthesis:** Google focuses on massive context windows (1M–2M+ tokens) through distributed Ring Attention and Grouped-Query Attention (GQA). Inference is accelerated via Speculative Decoding (DD31), where a lightweight draft model generates candidate tokens verified in parallel by the target model.

### DeepSeek-V3 & DeepSeek-R1 (DeepSeek AI)
*   **Architecture:** Multi-Head Latent Attention (MLA) + DeepSeekMoE + Pure RL Reasoning
*   **Deep Dives:** [14_Transformer_MoE_Sparse.md](14_Transformer_MoE_Sparse.md), [31_Speculative_Decoding_and_MLA.md](31_Speculative_Decoding_and_MLA.md), [36_Test_Time_Scaling_and_PRMs.md](36_Test_Time_Scaling_and_PRMs.md)
*   **Synthesis:** DeepSeek-V3 introduced Multi-Head Latent Attention (MLA), compressing the Key-Value cache into low-dimensional latent vectors to drastically slash serving memory. DeepSeekMoE uses fine-grained expert segmentation (256 routed experts with shared isolation experts). DeepSeek-R1 demonstrated that advanced reasoning chains can emerge purely through large-scale reinforcement learning (GRPO) without initial human-supervised fine-tuning.

### Titans Neural Memory (Google Research)
*   **Architecture:** Test-Time Gradient Descent (Long-Term Associative Memory)
*   **Deep Dives:** [27_Titans_Neural_Memory.md](27_Titans_Neural_Memory.md)
*   **Synthesis:** As context lengths expand, the KV cache footprint grows monotonically. The Titans architecture (Behrouz et al., 2024) replaces or augments the static KV cache with an associative neural memory module that updates its internal weights at inference time via gradient descent, enabling bounded-memory long-term context retention.

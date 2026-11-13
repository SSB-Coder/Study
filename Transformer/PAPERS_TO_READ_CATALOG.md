# Primary Literature Reading Catalog: 30 Landmark AI & LLM Papers (2017–2026)

To achieve true AI mastery, reading primary research papers is mandatory. This catalog indexes the 30 most influential research papers that shaped modern AI, detailing their arXiv identifiers, core mathematical innovations, and direct mappings to the Deep Dive reports in this repository.

---

## 1. The Foundational Architecture Epoch (2017–2021)

### 1. Attention Is All You Need (Vaswani et al., 2017)
*   **ArXiv:** `1706.03762`
*   **Core Contribution:** Replaced Recurrent and Convolutional networks with Multi-Head Scaled Dot-Product Self-Attention. Introduced positional encodings.
*   **Matching Report:** [13_Classic_Transformer_Dense.md](Reports/Deep_Dives/13_Classic_Transformer_Dense.md)

### 2. Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)
*   **ArXiv:** `2005.14165`
*   **Core Contribution:** Demonstrated that scaling autoregressive LLMs to 175B parameters enables zero-shot and few-shot in-context learning without fine-tuning.
*   **Matching Report:** [13_Classic_Transformer_Dense.md](Reports/Deep_Dives/13_Classic_Transformer_Dense.md)

### 3. An Image is Worth 16x16 Words: Vision Transformers (Dosovitskiy et al., 2020)
*   **ArXiv:** `2010.11929`
*   **Core Contribution:** Applied standard Transformer encoders directly to image patch sequences, surpassing CNNs for vision.
*   **Matching Report:** [16_Transformer_Omnimodal.md](Reports/Deep_Dives/16_Transformer_Omnimodal.md)

### 4. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
*   **ArXiv:** `2106.09685`
*   **Core Contribution:** Frozen base model weight matrices + injected trainable low-rank decomposition matrices ($W_0 + B \cdot A$), reducing fine-tuning trainable parameters by 99%.
*   **Matching Report:** [38_Quantization_and_LoRA.md](Reports/Deep_Dives/38_Quantization_and_LoRA.md)

---

## 2. Systems, Alignment & Scaling Epoch (2022–2023)

### 5. FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
*   **ArXiv:** `2205.14135`
*   **Core Contribution:** IO-aware attention algorithm using SRAM tiling and online Softmax normalization to reduce VRAM memory access from $O(N^2)$ to $O(N)$.
*   **Matching Report:** [23_Systems_FlashAttention_and_Triton.md](Reports/Deep_Dives/23_Systems_FlashAttention_and_Triton.md)

### 6. Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)
*   **ArXiv:** `2203.15556`
*   **Core Contribution:** Proved that for compute-optimal training, dataset token count must scale equally with parameter count (20 tokens per parameter).
*   **Matching Report:** [24_Scaling_Laws_and_RLHF.md](Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md)

### 7. ZeRO: Memory Optimizations for Training Trillion Parameter Models (Rajbhandari et al., 2020/2022)
*   **ArXiv:** `1910.02054`
*   **Core Contribution:** Distributed memory sharding across GPUs for Optimizer States (ZeRO-1), Gradients (ZeRO-2), and Weights (ZeRO-3).
*   **Matching Report:** [32_Distributed_Training_and_ZeRO.md](Reports/Deep_Dives/32_Distributed_Training_and_ZeRO.md)

### 8. LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
*   **ArXiv:** `2302.13971`
*   **Core Contribution:** Established modern open-weight LLM standard: Pre-normalization (RMSNorm), SwiGLU activations, and Rotary Position Embeddings (RoPE).
*   **Matching Report:** [30_Positional_Encodings_and_RoPE.md](Reports/Deep_Dives/30_Positional_Encodings_and_RoPE.md)

### 9. Direct Preference Optimization / DPO (Rafailov et al., 2023)
*   **ArXiv:** `2305.18290`
*   **Core Contribution:** Proved RLHF loss can be solved directly via policy model log-ratios without needing a separate Reward Model.
*   **Matching Report:** [24_Scaling_Laws_and_RLHF.md](Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md)

### 10. Mamba: Linear-Time Sequence Modeling with Selective SSMs (Gu & Dao, 2023)
*   **ArXiv:** `2312.00752`
*   **Core Contribution:** Introduced input-dependent selection mechanisms into continuous-time State Space Models, achieving $O(N)$ linear sequence scaling.
*   **Matching Report:** [15_Transformer_SSM_Mamba_Hybrids.md](Reports/Deep_Dives/15_Transformer_SSM_Mamba_Hybrids.md)

---

## 3. Advanced Modern & Frontier Epoch (2024–2026)

### 11. KAN: Kolmogorov-Arnold Networks (Liu et al., 2024)
*   **ArXiv:** `2404.19756`
*   **Core Contribution:** Replaced fixed node activation functions with learnable B-spline activation functions on network edges.
*   **Matching Report:** [20_KAN_Kolmogorov_Arnold.md](Reports/Deep_Dives/20_KAN_Kolmogorov_Arnold.md)

### 12. DeepSeek-V3 & DeepSeek-R1 Technical Reports (DeepSeek AI, 2024–2025)
*   **ArXiv:** `2412.19437` & `2501.12948`
*   **Core Contribution:** Multi-head Latent Attention (MLA), DeepSeekMoE fine-grained expert routing, and pure RL-driven reasoning emergence without preliminary SFT.
*   **Matching Reports:** [14_Transformer_MoE_Sparse.md](Reports/Deep_Dives/14_Transformer_MoE_Sparse.md) & [31_Speculative_Decoding_and_MLA.md](Reports/Deep_Dives/31_Speculative_Decoding_and_MLA.md)

### 13. Inference-Time Compute Scaling & Process Rewards (Snell et al. / Lightman et al., 2023–2024)
*   **ArXiv:** `2408.03314` & `2305.20050`
*   **Core Contribution:** Proved that allocating compute at inference time via Tree-of-Thoughts and Process Reward Models (PRMs) beats training larger models.
*   **Matching Report:** [36_Test_Time_Scaling_and_PRMs.md](Reports/Deep_Dives/36_Test_Time_Scaling_and_PRMs.md)

---

## 4. Systems, Frameworks & Serving Architecture Epoch

### 14. Scikit-learn: Machine Learning in Python (Pedregosa et al., 2011)
*   **JMLR:** `12(85):2825-2830`
*   **Core Contribution:** Established unified Estimator/Transformer/Pipeline API philosophy that became the global standard for tabular ML.
*   **Matching Report:** [44_Complete_ML_DL_AI_Libraries_Master_Guide.md](Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md)

### 15. PyTorch: An Imperative Style, High-Performance Deep Learning Library (Paszke et al., 2019)
*   **NeurIPS:** `32`
*   **Core Contribution:** Dynamic tape-based reverse-mode automatic differentiation (Autograd) and seamless CUDA hardware tensor execution.
*   **Matching Reports:** [40_Python_PyTorch_and_CUDA_Execution_Glossary.md](Reports/Deep_Dives/40_Python_PyTorch_and_CUDA_Execution_Glossary.md) & [44_Complete_ML_DL_AI_Libraries_Master_Guide.md](Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md)

### 16. Hugging Face Transformers: State-of-the-Art Natural Language Processing (Wolf et al., 2020)
*   **EMNLP:** `2020.emnlp-demos.6`
*   **Core Contribution:** Unified model architecture abstractions (`AutoModel`, `AutoTokenizer`) enabling rapid community democratization of pretrained foundation models.
*   **Matching Reports:** [43_Modern_AI_Frameworks_Encyclopedia.md](Reports/Deep_Dives/43_Modern_AI_Frameworks_Encyclopedia.md) & [44_Complete_ML_DL_AI_Libraries_Master_Guide.md](Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md)

### 17. Efficient Memory Management for Large Language Model Serving with PagedAttention / vLLM (Kwon et al., 2023)
*   **SOSP:** `23` / **ArXiv:** `2309.06180`
*   **Core Contribution:** PagedAttention algorithm inspired by OS virtual memory paging, eliminating KV-cache VRAM fragmentation and enabling high-throughput continuous batching.
*   **Matching Report:** [44_Complete_ML_DL_AI_Libraries_Master_Guide.md](Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md)

---

## 5. Universal Neural Computation & AI for Science Epoch

### 18. Auto-Encoding Variational Bayes (Kingma & Welling, 2013)
*   **ArXiv:** `1312.6114`
*   **Core Contribution:** Formulated the Variational Autoencoder (VAE) and the reparameterization trick, allowing backpropagation through stochastic latent variables by optimizing the ELBO.
*   **Matching Report:** [05_Autoencoders_VAE.md](Reports/Deep_Dives/05_Autoencoders_VAE.md)

### 19. Denoising Diffusion Probabilistic Models / DDPM (Ho et al., 2020)
*   **ArXiv:** `2006.11239`
*   **Core Contribution:** Formulated generative diffusion as reversing a parameterized Langevin Markov chain, proving mathematical equivalence to score-based generative modeling.
*   **Matching Report:** [07_Diffusion_Models.md](Reports/Deep_Dives/07_Diffusion_Models.md)

### 20. Highly Accurate Protein Structure Prediction with AlphaFold (Jumper et al., 2021)
*   **Nature:** `596, 583–589`
*   **Core Contribution:** The Evoformer architecture, introducing triangular multiplicative attention updates and Invariant Point Attention (IPA) for exact 3D atomic structure realization.
*   **Matching Report:** [48_Evoformer_and_AlphaFold_Architectures.md](Reports/Deep_Dives/48_Evoformer_and_AlphaFold_Architectures.md)

### 21. Physics-Informed Neural Networks / PINNs (Raissi et al., 2019)
*   **Journal:** *Journal of Computational Physics*, `378:686–707`
*   **Core Contribution:** Formulated PDE residuals directly inside neural loss functions using autograd derivatives, solving forward and inverse non-linear Navier-Stokes equations.
*   **Matching Report:** [45_PINNs_Physics_Informed_Neural_Networks.md](Reports/Deep_Dives/45_PINNs_Physics_Informed_Neural_Networks.md)

### 22. Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2020)
*   **ArXiv:** `2010.08895`
*   **Core Contribution:** Formulated neural operator learning in infinite-dimensional function spaces via spectral convolutions in Fourier space, achieving zero-shot super-resolution.
*   **Matching Report:** [46_Fourier_Neural_Operators_FNO.md](Reports/Deep_Dives/46_Fourier_Neural_Operators_FNO.md)

### 23. E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
*   **ArXiv:** `2102.09844`
*   **Core Contribution:** Guaranteed exact $E(3)$ rotational and translational equivariance over 3D particle coordinate updates without relying on computationally expensive spherical harmonics.
*   **Matching Report:** [47_Equivariant_GNNs_and_Molecular_Dynamics.md](Reports/Deep_Dives/47_Equivariant_GNNs_and_Molecular_Dynamics.md)

### 24. Towards Monosemanticity: Decomposing Language Models with Dictionary Learning (Bricken et al., 2023)
*   **Publication:** Anthropic Research
*   **Core Contribution:** Used overcomplete Sparse Autoencoders (SAEs) on transformer residual streams to disentangle polysemantic representations into clean, monosemantic concept directions.
*   **Matching Report:** [33_Mechanistic_Interpretability.md](Reports/Deep_Dives/33_Mechanistic_Interpretability.md)

### 25. Efficient and Robust Approximate Nearest Neighbor Search Using HNSW (Malkov & Yashunin, 2018)
*   **TPAMI / ArXiv:** `1603.09320`
*   **Core Contribution:** Multi-layer skip-graph hierarchy (Hierarchical Navigable Small World) delivering logarithmic complexity nearest-neighbor search across dense vector spaces.
*   **Matching Report:** [39_Vector_Databases_and_HNSW.md](Reports/Deep_Dives/39_Vector_Databases_and_HNSW.md)

---

## 6. Emerging Recurrent, Associative & Hybrid Frontiers

### 26. RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
*   **ArXiv:** `2104.09864`
*   **Core Contribution:** Rotary Position Embedding (RoPE) incorporating relative distance information via 2D rotational block diagonal matrices, now universal across modern LLMs.
*   **Matching Report:** [30_Positional_Encodings_and_RoPE.md](Reports/Deep_Dives/30_Positional_Encodings_and_RoPE.md)

### 27. Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
*   **ArXiv:** `2211.17192`
*   **Core Contribution:** Accelerated autoregressive generation 2x–3x without distribution drift by having a small draft model propose tokens validated in parallel by the target model.
*   **Matching Report:** [31_Speculative_Decoding_and_MLA.md](Reports/Deep_Dives/31_Speculative_Decoding_and_MLA.md)

### 28. xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)
*   **ArXiv:** `2405.04517`
*   **Core Contribution:** Extended LSTM with exponential gating and matrix-valued associative memory ($sLSTM$ and $mLSTM$), overcoming Transformer sequence length bottlenecks.
*   **Matching Report:** [26_RWKV_and_xLSTM.md](Reports/Deep_Dives/26_RWKV_and_xLSTM.md)

### 29. Titans: Learning to Memorize at Test Time (Behrouz et al., 2024)
*   **ArXiv:** `2412.20320`
*   **Core Contribution:** Neural long-term memory module trained via test-time gradient descent with surprise-gated associative storage, eliminating KV-cache growth.
*   **Matching Report:** [27_Titans_Neural_Memory.md](Reports/Deep_Dives/27_Titans_Neural_Memory.md)

### 30. A Path Towards Autonomous Machine Intelligence / I-JEPA (LeCun, 2022; Assran et al., 2023)
*   **ArXiv:** `2301.08243`
*   **Core Contribution:** Joint Embedding Predictive Architecture (JEPA) predicting latent representations rather than raw pixels/tokens, preventing representation collapse via Exponential Moving Average (EMA) targets.
*   **Matching Report:** [18_JEPA_Predictive_Architecture.md](Reports/Deep_Dives/18_JEPA_Predictive_Architecture.md)

---

## 7. Reading Strategy & Sequence

1. Read the matching Deep Dive report first to grasp the intuition, math, and PyTorch code.
2. Open the corresponding arXiv paper and read the **Abstract**, **Architecture Diagram**, and **Methodology Section**.
3. Compare the paper's mathematical notation with our custom PyTorch implementation in `Code/Deep_Dives/`.

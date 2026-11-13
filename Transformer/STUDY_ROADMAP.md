# The Neural Architecture Masterclass: Study Roadmap

Welcome to the ultimate repository for understanding Artificial Intelligence, Machine Learning, and Large Language Models from the absolute atomic scale to the modern SOTA (State of the Art). 

This repository contains two main folders:
1.  **`Reports/`**: Deep mathematical and theoretical explanations.
2.  **`Code/`**: Educational, explicitly-written PyTorch implementations.

This is a massive amount of information. If you try to jump straight to modern frontier architectures, the math will overwhelm you. **Follow this roadmap strictly in order.**

---

## Phase 0: Mathematical Foundations & Interactive Visualizers
*Goal: Understand the mathematical language of Neural Networks before looking at any Python code.*

1.  **Read:** [[Reports/Deep_Dives/00_The_Complete_Mathematics_of_Neural_Networks.md]] (The master glossary, backpropagation step-by-step numbers, and Transformer matrix math).
2.  **Study Interactive Web Guides & Visualizers:**
    *   [[math_mastery_transformer.html]] (Interactive web guide to Linear Algebra, Calculus, and Probability).
    *   [[AI_Architecture_Research_Breakdown.html]] (Interactive architectural breakdown & visual guide across neural network families).
    *   [[neural_codex (1).html]] (Interactive neural network codex and parameter flow map).
    *   [[report (1).html]] (Interactive theoretical synthesis and architecture reference dashboard).
    *   [[Sims/index.html]] (Master interactive simulation portal across 57 standalone architectures).

---

## Phase 1: The High-Level Overview
*Goal: Understand the history, the terminology, and the absolute basics of how a computer "learns" before looking at any code.*

1.  **Read:** [[Reports/01_Foundations_AI_ML_DL.md]] (What is AI vs ML vs DL? What are Embeddings?)
2.  **Read:** [[Reports/02_Evolution_of_Neural_Networks.md]] (The history from Perceptrons to Transformers).
3.  **Read:** [[Reports/03_Transformers_and_SOTA_Models.md]] (High-level architecture overview of modern Transformer models).
4.  **Read:** [[Reports/06_Mathematical_Foundations.md]] (Crucial. Do not skip this. Understand how gradients flow and what the Adam Optimizer does).
5.  **Read:** [[Reports/07_All_Neural_Network_Types.md]] (A quick glossary of the architectures so you know the map of the territory).
6.  **Read:** [[Reports/04_Limitations_and_Improvements.md]] (Key bottlenecks of early neural nets and how modern techniques solve them).
7.  **Review the Matrix:** [[Reports/08_Architecture_Pros_and_Cons.md]] (Keep this open as a cheat sheet to understand *why* each architecture was invented and what bottlenecks it solves).
8.  **Read:** [[Reports/05_Future_Of_AI.md]] (High-level trajectory from current LLMs toward AGI and multi-agent systems).

---

## Phase 2: Building the Fundamentals (Theory + Code)
*Goal: Learn the classic architectures that ruled the 2010s. For each step, read the Deep Dive report first, then immediately open the matching Python file and trace the math.*

*   **Warmup Code (Starter scripts in root `Code/` directory):**
    *   [[Code/01_Simple_NNs.py]] (Starter feedforward perceptron & multi-layer network)
    *   [[Code/02_CNN_RNN.py]] (Starter 2D Convolution & unrolled RNN loop)

1.  **The Multi-Layer Perceptron (MLP)**
    *   **Theory:** [[Reports/Deep_Dives/01_MLP_Feedforward.md]]
    *   **Code:** [[Code/Deep_Dives/01_MLP_Feedforward.py]]
    *   *Task:* Ensure you understand $XW^T + b$ and how ReLU activations work.
2.  **Convolutional Neural Networks (CNN)**
    *   **Theory:** [[Reports/Deep_Dives/02_CNN_Convolutional.md]]
    *   **Code:** [[Code/Deep_Dives/02_CNN_Convolutional.py]]
    *   *Task:* Understand how flattening a 2D spatial map connects to a 1D MLP.
3.  **Recurrent Neural Networks (RNN) & Memory**
    *   **Theory:** [[Reports/Deep_Dives/03_RNN_Recurrent.md]]
    *   **Code:** [[Code/Deep_Dives/03_RNN_Recurrent.py]]
    *   *Task:* Trace the "unrolled loop" in the code. Understand the Vanishing Gradient problem.
4.  **LSTM & GRU (Fixing the RNN)**
    *   **Theory:** [[Reports/Deep_Dives/04_LSTM_GRU.md]]
    *   **Code:** [[Code/Deep_Dives/04_LSTM_GRU.py]]
    *   *Task:* Identify the Forget, Input, and Output gates in the PyTorch code.

---

## Phase 3: The Generative Era
*Goal: Understand how AI generates *new* things (images, deepfakes) instead of just classifying them.*

1.  **Variational Autoencoders (VAE)**
    *   **Theory:** [[Reports/Deep_Dives/05_Autoencoders_VAE.md]]
    *   **Code:** [[Code/Deep_Dives/05_Autoencoders_VAE.py]]
    *   *Task:* Study the "Reparameterization Trick" in the code. This is a critical statistical concept.
2.  **Generative Adversarial Networks (GAN)**
    *   **Theory:** [[Reports/Deep_Dives/06_GANs.md]]
    *   **Code:** [[Code/Deep_Dives/06_GANs.py]]
    *   *Task:* Look at the mock training loop in the code to see how the Forger and Detective fight each other mathematically.
3.  **Diffusion Models (Midjourney / Stable Diffusion)**
    *   **Theory:** [[Reports/Deep_Dives/07_Diffusion_Models.md]]
    *   **Code:** [[Code/Deep_Dives/07_Diffusion_Models.py]]
    *   *Task:* Understand that the U-Net is just predicting the *noise*, not the actual image.

---

## Phase 4: Modern LLMs & The SOTA (The Path to Frontier Architectures)
*Goal: Understand the massive linguistic brains running the modern AI revolution.*

*   **Warmup Code (Starter scripts in root `Code/` directory):**
    *   [[Code/03_Self_Attention.py]] (Minimal standalone dot-product self-attention)
    *   [[Code/04_Transformer_Architecture.py]] (Minimal PyTorch single-head/multi-head Transformer block)
    *   [[Code/05_MoE_Architecture.py]] (Minimal top-k routing and Mixture of Experts implementation)

1.  **The Classic Dense Transformer**
    *   **Theory:** [[Reports/Deep_Dives/13_Classic_Transformer_Dense.md]]
    *   **Code:** [[Code/Deep_Dives/13_Classic_Transformer_Dense.py]]
    *   *Task:* Master the Scaled Dot-Product Attention equation. This is the most important algorithm of the decade.
2.  **Sparse Mixture of Experts (MoE)**
    *   **Theory:** [[Reports/Deep_Dives/14_Transformer_MoE_Sparse.md]]
    *   **Code:** [[Code/Deep_Dives/14_Transformer_MoE_Sparse.py]]
    *   *Task:* See how the Router code uses `topk()` to selectively shut off parts of the network to save compute.
3.  **State Space Models (SSM / Mamba)**
    *   **Theory:** [[Reports/Deep_Dives/15_Transformer_SSM_Mamba_Hybrids.md]]
    *   **Code:** [[Code/Deep_Dives/15_Transformer_SSM_Mamba_Hybrids.py]]
    *   *Task:* Understand how we defeat the $O(N^2)$ Transformer memory wall by returning to recurrent "notebook" mathematics.
4.  **Omnimodal Architectures (Gemini)**
    *   **Theory:** [[Reports/Deep_Dives/16_Transformer_Omnimodal.md]]
    *   **Code:** [[Code/Deep_Dives/16_Transformer_Omnimodal.py]]
5.  **Agentic Reasoning (Claude 3.5 / OpenAI o1/o3)**
    *   **Theory:** [[Reports/Deep_Dives/17_Transformer_Agentic_Reasoning.md]]
    *   **Code:** [[Code/Deep_Dives/17_Transformer_Agentic_Reasoning.py]]

---

## Phase 5: Exotic, Biological & Scientific Networks (Optional Deep Nerd Tier)
*Goal: Explore fringe, specialized, scientific, and highly experimental architectures.*

1.  **Graph Neural Networks (GNN)**
    *   **Theory:** [[Reports/Deep_Dives/08_GNN_Graph.md]]
    *   **Code:** [[Code/Deep_Dives/08_GNN_Graph.py]]
2.  **Spiking Neural Networks (SNN)** (Biological brain emulation)
    *   **Theory:** [[Reports/Deep_Dives/09_SNN_Spiking.md]]
    *   **Code:** [[Code/Deep_Dives/09_SNN_Spiking.py]]
3.  **Liquid Neural Networks (LNN)** (Continuous-time dynamic robotics)
    *   **Theory:** [[Reports/Deep_Dives/10_LNN_Liquid.md]]
    *   **Code:** [[Code/Deep_Dives/10_LNN_Liquid.py]]
4.  **Restricted Boltzmann Machines (RBM)**
    *   **Theory:** [[Reports/Deep_Dives/11_RBM_DeepBelief.md]]
    *   **Code:** [[Code/Deep_Dives/11_RBM_DeepBelief.py]]
5.  **Self-Organizing Maps (SOM)**
    *   **Theory:** [[Reports/Deep_Dives/12_SOM_Kohonen.md]]
    *   **Code:** [[Code/Deep_Dives/12_SOM_Kohonen.py]]
6.  **Non-Agentic Scientific AI Research Survey** - [[non-agentic-ai-research-directions.md]] (Comprehensive survey covering DeepMind's AlphaFold 2/3, GNoME crystal discovery, GraphCast weather forecasting, Arc's Evo 2 genomics, Fourier Neural Operators for physics, and Meta's V-JEPA world models).

---

## Phase 6: The Path to AGI (The Frontier)
*Goal: Look past current LLMs and explore the bleeding edge of intelligence design.*

1.  **JEPA (Joint Embedding Predictive Architecture)**
    *   **Theory:** [[Reports/Deep_Dives/18_JEPA_Predictive_Architecture.md]]
    *   **Code:** [[Code/Deep_Dives/18_JEPA_Predictive_Architecture.py]]
    *   *Task:* Understand how EMA prevents "information collapse" without using contrastive pairs.
2.  **Neurosymbolic AI**
    *   **Theory:** [[Reports/Deep_Dives/19_Neurosymbolic_AI_Hybrids.md]]
    *   **Code:** [[Code/Deep_Dives/19_Neurosymbolic_AI_Hybrids.py]]
    *   *Task:* Trace how logic rules (AND/OR) are compiled into differentiable fuzzy math for backprop.
3.  **Kolmogorov-Arnold Networks (KAN)**
    *   **Theory:** [[Reports/Deep_Dives/20_KAN_Kolmogorov_Arnold.md]]
    *   **Code:** [[Code/Deep_Dives/20_KAN_Kolmogorov_Arnold.py]]
    *   *Task:* Examine the code and verify that there are ZERO activation functions on the nodes, only on the edges!
4.  **Active Inference (Free Energy Principle)**
    *   **Theory:** [[Reports/Deep_Dives/21_Active_Inference.md]]
    *   **Code:** [[Code/Deep_Dives/21_Active_Inference.py]]
    *   *Task:* Contrast this with standard RL. Notice how the agent optimizes for Accuracy minus Complexity, not for a "Reward."

---

## Phase 7: The Researcher's Toolkit (Math, Systems, Evals)
*Goal: Understand the graduate-level mathematics and hardware engineering required to actually build and train new models.*

1.  **Information Theory & Manifolds**
    *   **Theory:** [[Reports/Deep_Dives/22_Information_Theory_and_Manifolds.md]]
    *   *Task:* Understand why we use Cross-Entropy and how latent space geometry works.
2.  **Hardware, FlashAttention & Triton**
    *   **Theory:** [[Reports/Deep_Dives/23_Systems_FlashAttention_and_Triton.md]]
    *   **Code:** [[Code/Deep_Dives/23_Triton_FlashAttention.py]]
    *   *Task:* Learn the Roofline Model and trace the custom SRAM memory kernel.
3.  **Scaling Laws & Alignment (RLHF/DPO)**
    *   **Theory:** [[Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md]]
    *   **Code:** [[Code/Deep_Dives/24_RLHF_PPO_Training_Loop.py]]
    *   *Task:* Understand the Chinchilla optimal compute limits and trace the PPO reinforcement loop.
4.  **Emergence of Intelligence & Evals**
    *   **Theory:** [[Reports/Deep_Dives/28_The_Emergence_of_Intelligence_and_Evaluations.md]]
    *   *Task:* Understand the philosophical "Compression" hypothesis and how we benchmark LLMs (MMLU, HumanEval).

---

## Phase 8: The Linear & Hybrid Frontier
*Goal: Explore the exact 2024 architecture competitors to the standard Transformer.*

1.  **Linear Attention and GLA**
    *   **Theory:** [[Reports/Deep_Dives/25_Linear_Attention_and_GLA.md]]
    *   **Code:** [[Code/Deep_Dives/25_Linear_Attention_and_GLA.py]]
    *   *Task:* Understand the associative property trick that achieves $O(N)$ speed.
2.  **RWKV and xLSTM**
    *   **Theory:** [[Reports/Deep_Dives/26_RWKV_and_xLSTM.md]]
    *   **Code:** [[Code/Deep_Dives/26_RWKV_and_xLSTM.py]]
    *   *Task:* Look at the code for xLSTM and trace how the 1D memory was expanded into a full $O(d^2)$ Matrix Memory.
3.  **Titans Neural Memory**
    *   **Theory:** [[Reports/Deep_Dives/27_Titans_Neural_Memory.md]]
    *   **Code:** [[Code/Deep_Dives/27_Titans_Neural_Memory.py]]
    *   *Task:* See how a neural network performs gradient descent *on itself* at test-time to memorize new inputs.

---

## Phase 9: Core Infrastructure (Engineering & Deployment)
*Goal: Understand the massive engineering pipelines required to train and serve models at scale.*

1.  **Tokenization & Data Curation**
    *   **Theory:** [[Reports/Deep_Dives/29_Tokenization_and_Data_Curation.md]]
    *   **Code:** [[Code/Deep_Dives/29_Dataset_Curation_MinHash.py]]
    *   *Task:* Learn why BPE fails spelling tests, and trace the MinHash deduplication pipeline.
2.  **Positional Encodings**
    *   **Theory:** [[Reports/Deep_Dives/30_Positional_Encodings_and_RoPE.md]]
    *   *Task:* Understand how RoPE rotates embeddings to keep track of word order and extrapolate context.
3.  **Inference Acceleration**
    *   **Theory:** [[Reports/Deep_Dives/31_Speculative_Decoding_and_MLA.md]]
    *   **Code:** [[Code/Deep_Dives/31_Speculative_Decoding_and_MLA.py]]
    *   *Task:* Run the speculative decoding script to see how a draft model achieves 3x speedups.
4.  **Distributed Training**
    *   **Theory:** [[Reports/Deep_Dives/32_Distributed_Training_and_ZeRO.md]]
    *   **Code:** [[Code/Deep_Dives/32_DeepSpeed_ZeRO3_NCCL.py]]
    *   *Task:* Learn how DeepSpeed ZeRO-3 config slices model weights across 1,000 GPUs via NCCL.
5.  **Mechanistic Interpretability**
    *   **Theory:** [[Reports/Deep_Dives/33_Mechanistic_Interpretability.md]]
    *   *Task:* Understand the Superposition Hypothesis and how Sparse Autoencoders find the "Golden Gate Bridge" neuron.

---

## Phase 10: The Engineering Benchmarks & SFT
*Goal: Bridge the gap into actual Python implementation for tool use and profiling.*

1.  **Agentic Tool Use Fine-Tuning**
    *   **Code:** [[Code/Deep_Dives/34_Agentic_Tool_Use_FineTuning.py]]
    *   *Task:* See how Supervised Fine-Tuning trains a model to emit `<tool_call>` tags and wait for an external API.
2.  **Benchmarking Efficiency**
    *   **Theory:** [[Reports/Deep_Dives/35_Benchmarking_Efficiency.md]]
    *   **Code:** [[Code/Deep_Dives/35_Benchmarking_Efficiency.py]]
    *   *Task:* Run this PyTorch Profiler script to measure Latency, FLOPs, and Memory footprint without training an LLM.

---

## Phase 11: Frontier Test-Time Scaling & SOTA Review
*Goal: Understand the final paradigm shift of 2024–2026 and how it maps to actual commercial models.*

1.  **Test-Time Compute Scaling & PRMs**
    *   **Theory:** [[Reports/Deep_Dives/36_Test_Time_Scaling_and_PRMs.md]]
    *   **Code:** [[Code/Deep_Dives/36_Test_Time_Scaling_and_PRMs.py]]
    *   *Task:* Run the code to see how a model uses a Tree-of-Thoughts and a Process Reward Model to scale compute during inference.
2.  **The 2026 SOTA Literature Review**
    *   **Theory:** [[Reports/Deep_Dives/37_2026_SOTA_Literature_Review.md]]
    *   *Task:* Read how the architectures of modern frontier models (OpenAI o1/o3, Claude 3.5 Sonnet, DeepSeek-V3/R1, and Grok) map directly to the mathematics you have learned in the previous 36 Deep Dives.

---

## Phase 12: The Final Deployment Pillars
*Goal: The absolute final step. How to actually compress and deploy a 70B model in the real world.*

1.  **Quantization & LoRA (Fine-Tuning)**
    *   **Theory:** [[Reports/Deep_Dives/38_Quantization_and_LoRA.md]]
    *   **Code:** [[Code/Deep_Dives/38_Quantization_and_LoRA.py]]
    *   *Task:* Understand how a massive weight matrix is compressed into 4-bit, and trace the math of Low-Rank Adaptation ($W + A \times B$).
2.  **Vector Databases & Advanced RAG (HNSW)**
    *   **Theory:** [[Reports/Deep_Dives/39_Vector_Databases_and_HNSW.md]]
    *   **Code:** [[Code/Deep_Dives/39_Vector_Databases_and_HNSW.py]]
    *   *Task:* See how the Hierarchical Navigable Small World (HNSW) graph algorithm allows a vector database to search a billion documents in milliseconds.

---

## Phase 13: Programming Mastery & System Operations
*Goal: Master the complete code syntax, execution model, framework landscape, and enterprise software stack.*

1.  **Python, PyTorch & CUDA Execution Glossary**
    *   **Theory:** [[Reports/Deep_Dives/40_Python_PyTorch_and_CUDA_Execution_Glossary.md]]
    *   *Task:* Study the syntax reference for every Python keyword, PyTorch tensor method, autograd hook, and Triton GPU execution primitive used in this codebase.
2.  **Advanced RAG Architecture & Retrieval Ecosystem**
    *   **Theory:** [[Reports/Deep_Dives/41_Advanced_RAG_Architecture_and_Ecosystem.md]]
    *   *Task:* Master Naive vs Advanced vs GraphRAG, HyDE, Reranking, Multi-Query expansion, and RAGAS evaluation metrics.
3.  **Comprehensive Fine-Tuning & PEFT Mastery**
    *   **Theory:** [[Reports/Deep_Dives/42_Comprehensive_FineTuning_Mastery.md]]
    *   *Task:* Learn the exact mechanics of SFT, Alignment (RLHF, DPO, KTO, ORPO, SimPO), PEFT (LoRA, QLoRA, DoRA), and compute optimizations.
4.  **Modern AI Frameworks Encyclopedia**
    *   **Theory:** [[Reports/Deep_Dives/43_Modern_AI_Frameworks_Encyclopedia.md]]
    *   *Task:* Master the full software stack: TensorFlow/Keras 3, JAX/Flax, LangChain/LCEL, LangGraph, LlamaIndex, CrewAI, DSPy, Hugging Face, vLLM, llama.cpp, and Classical ML.
5.  **Complete ML, DL, AI & Transformer Libraries Master Guide & Cookbook**
    *   **Theory:** [[Reports/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Master_Guide.md]] & [[Reports/09_Complete_ML_DL_AI_Libraries_Master_Guide.md]]
    *   **Code:** [[Code/Deep_Dives/44_Complete_ML_DL_AI_Libraries_Cookbook.py]]
    *   *Task:* Master Scikit-Learn Pipelines, PyTorch 2.x Autograd & AMP, Keras 3 Multi-Backend, Hugging Face Tokenizers & PEFT LoRA, vLLM PagedAttention, LangGraph ReAct agents, and LangSmith/Langfuse observability tracing.

---

## Phase 14: Interactive Testing & Environment Verification
*Goal: Automatically test your local setup and diagnose your readiness before building projects.*

1.  **Automated Codebase Verification Suite**
    *   **Code:** [[Code/verify_all.py]]
    *   *Task:* Run `python Code/verify_all.py` in your terminal to verify Python/PyTorch/CUDA/Triton dependencies and run a dry-run test across all 42 deep-dive scripts.
2.  **Master Self-Assessment Quiz & Diagnostic Guide**
    *   **Guide:** [[SELF_ASSESSMENT_QUIZ.md]]
    *   *Task:* Complete the self-diagnostic quiz questions across all phases and check your math, code, and architectural understanding.

---

## Phase 15: Research Catalog & Real-World Debugging Playbook
*Goal: Connect theory to landmark primary literature and master operational ML troubleshooting.*

1.  **Primary Literature Reading Catalog**
    *   **Catalog:** [[PAPERS_TO_READ_CATALOG.md]]
    *   *Task:* Study the top 30 landmark research papers from 2017 to 2026 (Vaswani, LoRA, FlashAttention, Chinchilla, DPO, Mamba, DeepSeek-V3/R1).
2.  **Real-World ML & LLM Debugging Playbook**
    *   **Playbook:** [[LLM_DEBUGGING_AND_TROUBLESHOOTING.md]]
    *   *Task:* Master operational solutions for CUDA OOM, NaN loss, gradient explosions, LoRA rank collapse, and tokenizer generation bugs.
3.  **Master Lookup Reference & Formulas**
    *   **Cheat Sheet:** [[Transformer_Cheat_Sheet.md]]

---

## Phase 16: Universal Neural Computation & AI for Science
*Goal: Move beyond text and language models into the continuous physical universe—aerodynamics, fluid dynamics, chemistry, structural biology, microchips, quantum mechanics, and symbolic mathematics.*

1.  **Physics-Informed Neural Networks (PINNs) & Fluid Turbulence**
    *   **Theory:** [[Reports/Deep_Dives/45_PINNs_Physics_Informed_Neural_Networks.md]]
    *   **Code:** [[Code/Deep_Dives/45_PINNs_Navier_Stokes_Turbulence.py]]
    *   *Task:* Master exact continuous derivatives via PyTorch Autograd and solve the non-linear Burgers' equation while discovering unknown physical constants.
2.  **Fourier Neural Operators (FNO) & Infinite-Dimensional Operators**
    *   **Theory:** [[Reports/Deep_Dives/46_Fourier_Neural_Operators_FNO.md]]
    *   **Code:** [[Code/Deep_Dives/46_Fourier_Neural_Operator_2D.py]]
    *   *Task:* Understand spectral convolutions in Fourier space and verify zero-shot super-resolution across varying mesh resolutions.
3.  **Equivariant Graph Neural Networks (EGNN) & Molecular Dynamics**
    *   **Theory:** [[Reports/Deep_Dives/47_Equivariant_GNNs_and_Molecular_Dynamics.md]]
    *   **Code:** [[Code/Deep_Dives/47_SE3_Equivariant_Molecular_GNN.py]]
    *   *Task:* Prove $E(3)$ rotational invariance of molecular scalar energy and exact rotational equivariance of 3D atomic forces.
4.  **The Evoformer & Structural Biology (AlphaFold 2/3)**
    *   **Theory:** [[Reports/Deep_Dives/48_Evoformer_and_AlphaFold_Architectures.md]]
    *   **Code:** [[Code/Deep_Dives/48_Triangular_Attention_Evoformer.py]]
    *   *Task:* Trace the outer-product mean co-evolution projection and enforce the 3D triangle inequality via triangular multiplicative edge updates.
5.  **AI for Chip Design & Electronic Design Automation (EDA)**
    *   **Theory:** [[Reports/Deep_Dives/49_AI_for_Chip_Design_and_Floorplanning.md]]
    *   **Code:** [[Code/Deep_Dives/49_GNN_Netlist_Floorplanning.py]]
    *   *Task:* Model circuit netlists as bipartite graphs and optimize silicon die floorplans using differentiable Log-Sum-Exp wirelength (HPWL).
6.  **Neural Network Quantum States (NQS) & Variational Monte Carlo**
    *   **Theory:** [[Reports/Deep_Dives/50_Neural_Network_Quantum_States_VMC.md]]
    *   **Code:** [[Code/Deep_Dives/50_Neural_Wavefunction_Quantum_Ising.py]]
    *   *Task:* Overcome the $2^N$ Hilbert space dimensionality wall using a neural wavefunction and optimize ground state energy via MCMC sampling.
7.  **Automated Theorem Proving, Lean 4 & Symbolic Discovery**
    *   **Theory:** [[Reports/Deep_Dives/51_Automated_Theorem_Proving_and_Lean4.md]]
    *   **Code:** [[Code/Deep_Dives/51_Symbolic_Regression_Kan_Physics.py]]
    *   *Task:* Explore formal mathematical kernels and extract closed-form human-readable physics equations from neural weights using $L_1$ Lasso sparsity.

---

## Phase 17: Your Personal Research Vectors
Once you have mastered the code in Phases 0-16, it is time to build your own systems.
*   **Read:** [[FUTURE_RESEARCH_MAP.md]] (Located in the root directory).
This document will give you your personalized curriculum for the heavy math, hardware engineering (CUDA/Triton), and non-silicon (Organoid Intelligence) research needed to actually build AGI.

---

## How to Study the Code
Do not just read the `.py` files. 
1.  Ensure you have Python and PyTorch installed on your machine.
2.  Run the files in your terminal (e.g., `python Code/Deep_Dives/13_Classic_Transformer_Dense.py`).
3.  Watch the `__main__` block execute the dummy data. 
4.  **The Challenge:** Try modifying the `batch_size`, `embed_dim`, or `seq_length` variables in the `__main__` block. If the code still runs without throwing a Tensor shape error, you have successfully understood how the matrix dimensions flow through the architecture!

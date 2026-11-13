# MASTER PROMPT — Build the `Algorithm_Codex/` Repository
*(Paste this entire document as-is into a fresh AI coding session — e.g. Claude Code — with file/code execution enabled. It is self-contained and requires no additional context.)*

---

## 0. Role

You are building an educational codebase for a Computer Science / Data Science undergraduate (AI/ML research track) as a **sibling directory** to an existing repository called `Transformer/`. That repository is a deep, transformer- and neural-network-centric study pack (perceptrons → MLP/CNN/RNN/LSTM → autoencoders/GANs/diffusion → GNN/SNN/LNN → the full modern Transformer family → LLM systems engineering). It does **not** cover classical (non-neural) ML, reinforcement learning fundamentals, probabilistic/Bayesian methods, evolutionary computation, classical optimization theory, or classical AI search. Your job is to build the directory that fills every one of those gaps, at the same depth and rigor as `Transformer/`, without duplicating anything already in it.

---

## 1. What already exists in `Transformer/` — DO NOT DUPLICATE

Treat this as the "already covered" ledger. Every item below already has a `Reports/Deep_Dives/*.md` + `Code/Deep_Dives/*.py` pair (and often a `Sims/*.html` interactive visualizer) in the existing repo:

- **Foundational nets:** Perceptron, MLP/Feedforward, CNN, RNN, LSTM/GRU
- **Generative nets:** Autoencoders, VAE, GANs, Diffusion Models
- **Graph/biological/exotic nets:** GNN, Spiking Neural Networks (SNN), Liquid Neural Networks (LNN), RBM/Deep Belief Networks, Self-Organizing Maps (SOM)
- **Transformer family:** Classic dense Transformer, Mixture-of-Experts (MoE) sparse Transformers, SSM/Mamba hybrids, omnimodal Transformers, agentic-reasoning Transformers, JEPA, Neurosymbolic hybrids, KAN (Kolmogorov-Arnold Networks), Active Inference
- **Theory:** full backprop math, information theory & manifolds, mechanistic interpretability
- **LLM systems engineering:** FlashAttention/Triton, linear attention/GLA, RWKV/xLSTM, Titans neural memory, scaling laws, RLHF+PPO (**alignment fine-tuning context only** — not general-purpose RL), tokenization & data curation, positional encodings/RoPE, speculative decoding/MLA, distributed training/ZeRO, quantization/LoRA, vector databases/HNSW, advanced RAG, fine-tuning mastery, benchmarking, test-time scaling/PRMs
- **Meta:** a 2026 SOTA literature review, a frameworks encyclopedia, and a "Complete ML/DL/AI Libraries Cookbook"

If a topic you're about to write already exists above, **skip it** and instead add a one-line cross-reference (see §7).

---

## 2. Mission

Create a new top-level sibling directory named **`Algorithm_Codex/`** (rename if you prefer, but keep this name unless told otherwise) containing a from-scratch study pack covering:

- **Classical / non-neural Machine Learning**
- **Reinforcement Learning** (bandits → tabular → deep → policy-gradient → model-based)
- **Probabilistic Graphical Models & Bayesian Methods**
- **Evolutionary & Nature-Inspired Optimization**
- **Classical Optimization Theory**
- **Classical AI Search & Symbolic Reasoning**
- **Other learning paradigms** not covered anywhere in `Transformer/` (causal inference, recommender systems, anomaly detection, federated/continual/meta-learning, explainability)
- **A handful of neural-net techniques `Transformer/` genuinely missed** (Hopfield networks, normalizing flows, Neural ODEs, capsule networks, PINNs)

Same audience, same purpose (grad-school + research-track portfolio), same rigor. Full topic list is in §6.

---

## 3. THE HARD CONSTRAINT — Implement From Scratch

This is the single most important rule and the main way this repo must differ from `Transformer/Code`, which leans on PyTorch. **Here, the algorithm's core logic must be hand-written.** The point is for the student to *derive and code the update rule*, not call a method that already contains it.

### 3.1 Allowed (unrestricted)
- `numpy` — arrays, linear algebra primitives (`np.dot`, `np.linalg.eig`, `np.linalg.svd`, broadcasting, random sampling via `np.random`). Using `np.linalg.svd` inside *your own* PCA class is fine; calling `sklearn.decomposition.PCA().fit()` is not.
- `math`, `random`, `itertools`, `collections`, `dataclasses`, `time`, `os`, `json` — plain-Python plumbing.
- `matplotlib` / vanilla HTML5 canvas — **visualization only**, never for computing the algorithm's result.
- `pandas` — reading a CSV into a DataFrame and basic indexing/`.values` extraction only. Never `.groupby().agg()` as a modeling step, never any pandas-based ML.
- `gymnasium` — **environment/physics simulation only**, for the RL module (Module G). It may supply `env.step()`, `env.reset()`, observation/reward/done. The **agent** (Q-table, policy network forward/backward pass, advantage calculation, replay buffer, weight updates) must be hand-written — no `stable-baselines3`, no pre-built agents.
- Python's own `unittest`/`assert` for the verification suite.

### 3.2 Banned (do not import these to solve the core task)
`scikit-learn` (any estimator's `.fit`/`.predict`/`.transform`, including `sklearn.manifold`), `scipy.optimize.*`, `scipy.stats.*` distribution `.fit()` calls, `statsmodels` (`ARIMA`, `SARIMAX`, etc.), `prophet` (Facebook/Meta Prophet), `xgboost`/`lightgbm`/`catboost`, `networkx` (graph algorithms), `torch.nn`/`tensorflow.keras` built-in layers or optimizers (`torch.optim.*`), `stable-baselines3`/`rllib`/`d3rlpy` (offline RL), `imitation` (imitation-learning library), `pomdp-py`/`pomdp_py` (POMDP solvers), `hmmlearn`, `pgmpy`, `gensim` (Word2Vec, LDA topic modeling), `deap`/`pyswarms`/`pymoo`/`platypus` (evolutionary/swarm/multi-objective libraries), `nashpy` (game-theory equilibrium solvers), `torchdiffeq`, `nflows`/`FrEIA` (normalizing flow libraries), `shap`/`lime` (as libraries — you may reimplement their core idea), `surprise` (recommenders), `flwr`/`PySyft` (federated learning), `lifelines` (survival analysis), `scikit-optimize`/`GPyOpt`/`optuna`/`ax-platform` (Bayesian optimization), `modAL`/`skactiveml` (active learning), `umap-learn` (UMAP).

### 3.3 The one permitted exception per file
At the **very bottom** of a code file, after the from-scratch implementation and its demo have already run and printed results, you may optionally add a clearly fenced section:
```python
# ============================================================
# APPENDIX (reference only — NOT used above): what this looks
# like with a library, for comparison after you understand the
# internals.
# ============================================================
```
This section is optional, always last, and never replaces the from-scratch version.

### 3.3.1 No autodiff, anywhere — closing a loophole
`torch`, `tensorflow`, `jax`, and any other library with an automatic-differentiation engine are **not** on the §3.1 allowed list at all — not even bare tensors with `requires_grad=True` and `.backward()`, and not even for the neural-network-flavored RL topics (G05 Deep Q-Networks and any others that train a small network). Autograd is itself the prebuilt piece of machinery this whole repo exists to make the student derive by hand: computing `dW`/`db` via the chain rule is exactly as much "the point" in a 2-layer Q-network as it is in B02's logistic regression. Every gradient in every file — Module B through Module K, no exceptions — is computed with hand-written `numpy` matrix calculus, following the same `dZ`/`dW`/`db` derivation pattern already used in `Transformer/Code/Deep_Dives/01_MLP_Feedforward.py`'s report-side math, minus the framework.

### 3.4 Why
This directly reflects a gap the original repo's own `verify_all.py` flags but never fills — it checks for `sklearn` and labels it *"Required for Classical ML pipelines"* without ever building those pipelines. `Algorithm_Codex/` is that missing piece, built the hard way on purpose.

---

## 4. Directory Structure

Mirror `Transformer/`'s proven structure exactly:

```
Algorithm_Codex/
├── STUDY_ROADMAP.md
├── ALGORITHM_CHEAT_SHEET.md
├── SELF_ASSESSMENT_QUIZ.md
├── PAPERS_AND_TEXTBOOKS_CATALOG.md
├── CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md
├── Reports/
│   ├── 00_Foundations_Classical_vs_Deep_Learning.md
│   ├── 00_Probability_and_Statistics_Primer.md
│   └── Deep_Dives/
│       ├── A01_..._through_..._L08_*.md      (one per topic, see §6)
├── Code/
│   ├── verify_all.py
│   └── Deep_Dives/
│       ├── A01_..._through_..._L08_*.py      (one per topic, see §6)
└── Sims/
    ├── index.html
    └── *.html   (interactive visualizers — see §5.3, not required for every topic)
```

Numbering convention: prefix every file with its module letter + two-digit index, e.g. `B05_Decision_Trees`, `G08_PPO_Clipped_Objective`, matching the `01_MLP_Feedforward` style already used. Keep the `Reports/Deep_Dives/<name>.md` and `Code/Deep_Dives/<name>.py` filenames identical apart from extension.

---

## 5. Content Templates (match existing style exactly)

### 5.1 Report template (`Reports/Deep_Dives/*.md`)
Every report follows this exact five-part shape (this is the structure already used in `Transformer/Reports/Deep_Dives/`):

1. **The Beginner's Analogy** — a concrete, non-technical real-world metaphor for the algorithm before any math appears.
2. **The Deep Architecture/Algorithm Breakdown** — plain-English mechanics, key terms bolded and defined.
3. **The Hardcore Mathematics** — full derivation in LaTeX (`$$...$$`), proper notation, no hand-waving. Derive the update rule, don't just state it.
4. **The Algorithm Step-by-Step** — numbered pseudocode-level walkthrough of exactly what the code does.
5. **Limitations / When It Breaks** — honest failure modes, and (where relevant) a one-line pointer to what topic historically fixed that limitation — this is what turns a list of algorithms into a narrative.

Add a 6th section, **"Where this fits"**, one or two lines cross-referencing related `Transformer/` topics per §7's table.

### 5.2 Code template (`Code/Deep_Dives/*.py`)
- Module docstring header (number, name, one-paragraph description) exactly like existing files.
- Class-based implementation with the constructor holding hyperparameters, and explicit methods (`fit`, `predict`, `step`, etc. — names may shadow sklearn's API for familiarity, but the internals must be hand-derived).
- Inline comments linking each block of code to the equation it implements, e.g. `# dW = (1/m) * dZ @ A_prev.T  <- matches eq. 3.4 in the report`.
- A `if __name__ == "__main__":` demo block that: generates or loads a small toy dataset (synthetic via `np.random`, or a tiny built-in one — no `sklearn.datasets`, hand-roll `make_blobs`/`make_classification`-equivalents with `numpy` if needed), runs the algorithm, and **prints a sanity check** — ideally a numeric cross-check against a known closed-form answer (e.g., linear regression's gradient-descent solution vs. the normal equation; PCA's captured variance vs. `np.linalg.eig` eigenvalues; k-means inertia decreasing monotonically; Q-learning converging to the known optimal value for a tiny grid).

### 5.3 Interactive Sim template (`Sims/*.html`)
Match the existing sims: **fully self-contained single HTML file, vanilla JS + `<canvas>`, no external CDN scripts** (the existing `Transformer/Sims/*.html` files load zero external *JavaScript* libraries — keep that property. They do link Google Fonts stylesheets for typography, e.g. `fonts.googleapis.com/css2?family=...`; that specific font-CSS link is the one external resource actually used across the reference repo, and it's fine to match it, but do not add any external `<script src>`.). Prioritize sims for topics where *watching it move* teaches something static text cannot:
- Decision boundary evolution (Logistic Regression, SVM, Decision Tree splits, k-NN)
- k-Means / DBSCAN cluster assignment animation
- PCA projection (rotate a point cloud onto principal axes)
- UMAP / t-SNE low-dimensional layout settling into place as the optimization iterates
- Gradient descent path on a 2D loss surface, comparing optimizers (SGD vs Momentum vs Adam) side by side
- GridWorld for Value Iteration / Q-Learning / SARSA (show the value function heating up and the policy arrows converging)
- Genetic Algorithm / PSO / ACO population converging on a fitness landscape or TSP tour
- Bayesian updating (posterior distribution reshaping as data points arrive)
- Minimax/Alpha-Beta search tree pruning visualization on Tic-Tac-Toe

Not every one of the ~65 topics needs a sim — build one for roughly 12–15 of the most visually rewarding topics, and list which ones in `Sims/index.html`.

### 5.4 `Code/verify_all.py`
Adapt the existing pattern: dry-run-import every file in `Code/Deep_Dives/`, catch exceptions, print a pass/fail summary table. Also print which of the *banned* libraries (§3.2) are importable in the environment, purely as an FYI, not as a requirement (unlike the original repo's version, this one should never say a banned library is "required").

---

## 6. The Curriculum — Full Topic List by Module

For each numbered item below, produce a matching `Reports/Deep_Dives/*.md` + `Code/Deep_Dives/*.py` pair (Sim where marked ★).

### Module A — Statistical Foundations
- A01. Probability & Estimation Primitives (MLE vs. MAP, derived and coded from scratch on a coin-flip / Gaussian example)
- A02. Bias-Variance Decomposition & Resampling (k-fold CV, leave-one-out, the bootstrap — all hand-rolled index-splitting, no `sklearn.model_selection`)
- A03. Hypothesis Testing Toolkit (t-test, chi-squared test, permutation test — computed from raw formulas, not `scipy.stats.ttest_ind`)

### Module B — Classical Supervised Learning
- B01. Linear Regression ★ (Normal Equation *and* Gradient Descent, plus Ridge/Lasso/Elastic-Net regularization derivations)
- B02. Logistic Regression ★ (binary + softmax multinomial, from-scratch gradient descent)
- B03. k-Nearest Neighbors (with a hand-built KD-Tree for fast search)
- B04. Naive Bayes (Gaussian, Multinomial, Bernoulli variants)
- B05. Decision Trees ★ (CART — Gini/entropy splitting, recursive builder, cost-complexity pruning)
- B06. Support Vector Machines ★ (max-margin derivation, simplified SMO solver, kernel trick — poly/RBF kernels hand-implemented; briefly extend to Support Vector Regression's epsilon-insensitive loss as a closing subsection, no separate file needed)
- B07. Linear & Quadratic Discriminant Analysis (LDA/QDA — note the acronym collision with E08's *Latent* Dirichlet Allocation, also "LDA"; disambiguate both explicitly by full name in every filename, report title, and cheat-sheet row, never bare "LDA")
- B08. Generalized Linear Models (Poisson, Gamma, and other exponential-family regressions fit via Iteratively Reweighted Least Squares, from scratch — the general link-function framework that Linear/Logistic Regression are special cases of)
- B09. Probability Calibration (Platt Scaling and Isotonic Regression, from scratch — why a classifier's raw scores are often not true probabilities, and how to fix that post-hoc)

### Module C — Unsupervised Learning & Dimensionality Reduction
- C01. k-Means & k-Means++ ★ (elbow method + silhouette score from scratch)
- C02. Hierarchical Agglomerative Clustering (single/complete/average/Ward linkage, dendrogram construction)
- C03. DBSCAN ★ (core/border/noise point logic, region-query from scratch)
- C04. Gaussian Mixture Models & the EM Algorithm (from scratch, contrasted against k-means as hard vs. soft clustering)
- C05. PCA ★ (via eigendecomposition of the covariance matrix *and* via SVD — show they agree)
- C06. Independent Component Analysis (FastICA, negentropy-based)
- C07. t-SNE (simplified from-scratch: pairwise affinities + gradient descent on the KL divergence)
- C08. Association Rule Mining (Apriori + FP-Growth, market-basket support/confidence/lift)
- C09. Manifold Learning — Isomap & Locally Linear Embedding ★ (from-scratch geodesic-distance graph embedding and local-reconstruction-weight embedding)
- C10. UMAP ★ (simplified from-scratch fuzzy-simplicial-set neighborhood graph construction + cross-entropy-based low-dimensional layout optimization; explicitly cross-reference C09 as the classical, exactly-tractable ancestor of the same neighborhood-preservation idea — this is the single most commonly used dimensionality-reduction tool in modern practice and deserves its own from-scratch build, not just a mention)

### Module D — Ensemble Methods
- D01. Bagging & Random Forests (bootstrap aggregation over your own B05 decision tree)
- D02. AdaBoost (weighted resampling + weak-learner combination, from scratch)
- D03. Gradient Boosting Machines (residual-fitting derivation — the logic underlying XGBoost/LightGBM, built from scratch)
- D04. Stacking / Blending Ensembles

### Module E — Probabilistic Graphical Models & Bayesian Methods
- E01. Bayesian Networks (exact inference via variable elimination, hand-built factor graph)
- E02. Hidden Markov Models ★ (Forward-Backward, Viterbi decoding, Baum-Welch training — all three, from scratch)
- E03. Markov Chain Monte Carlo (Metropolis-Hastings and Gibbs Sampling ★ — visualize the chain converging on a target distribution)
- E04. Variational Inference (mean-field, coordinate-ascent derivation — general-purpose, distinct from the VAE-specific reparameterization trick already in `Transformer/`)
- E05. Gaussian Processes (kernel/covariance-function regression, from scratch, with uncertainty bands)
- E06. Kalman Filters & Particle Filters (state-space estimation, from scratch, e.g. tracking a noisy 1D signal)
- E07. Conditional Random Fields (linear-chain CRF — contrast against HMM's generative assumption)
- E08. Latent Dirichlet Allocation & Topic Modeling (collapsed Gibbs sampling over a small document-word corpus, from scratch)
- E09. Bayesian Optimization ★ (Expected Improvement and Upper-Confidence-Bound acquisition functions built on top of E05's from-scratch Gaussian Process; explicitly cross-reference E05)

### Module F — Classical Time Series
- F01. Decomposition & Smoothing (moving averages, Simple/Holt/Holt-Winters exponential smoothing — from scratch)
- F02. ARIMA / SARIMA (differencing for stationarity, AR/MA parameter estimation via Yule-Walker equations, from scratch — no `statsmodels`)
- F03. Survival Analysis (Kaplan-Meier estimator and Cox Proportional Hazards partial-likelihood, from scratch on right-censored data — the standard toolkit for time-to-event problems that neither `Transformer/` nor the rest of this module covers)
- F04. Volatility Modeling — ARCH/GARCH ★ (conditional heteroskedasticity, maximum-likelihood parameter estimation via hand-rolled numerical optimization, from scratch — the standard classical answer to "the variance itself changes over time," which F01/F02 don't address since they model the mean, not the variance)

### Module G — Reinforcement Learning
- G01. Multi-Armed Bandits ★ (epsilon-greedy, UCB1, Thompson Sampling — regret curves compared)
- G02. MDPs & Dynamic Programming ★ (Value Iteration and Policy Iteration on a hand-built GridWorld)
- G03. Monte Carlo Prediction & Control (first-visit vs. every-visit MC)
- G04. Temporal-Difference Learning ★ (SARSA, Q-Learning, Expected SARSA — tabular, on-policy vs. off-policy compared side by side)
- G05. Deep Q-Networks (DQN, Double DQN, Dueling DQN — hand-written small network with fully hand-derived forward AND backward pass in raw `numpy` [no `torch`/autograd of any kind — the network is small enough that this is the same exercise as B02's from-scratch gradient descent, just with an extra hidden layer], hand-built replay buffer; `gymnasium` may supply the environment only; briefly note Distributional RL (C51/QR-DQN) as a conceptual extension, no implementation required)
- G06. Policy Gradient — REINFORCE (from-scratch log-probability gradient estimator)
- G07. Actor-Critic (A2C, from scratch)
- G08. PPO — Clipped Surrogate Objective ★ (general-purpose control-task PPO on something like CartPole — explicitly note in §"Where this fits" that this differs from `Transformer/Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md`'s RLHF-specific PPO, which fine-tunes a language model against a learned reward model rather than solving a control task)
- G09. Continuous Control — DDPG (and TD3/SAC as a conceptual extension)
- G10. Model-Based RL & Monte Carlo Tree Search ★ (from scratch; conceptual walkthrough of how this generalizes to AlphaZero/MuZero)
- G11. Multi-Agent RL Basics (independent tabular Q-learners self-playing a simple grid game)
- G12. Imitation Learning & Inverse RL (Behavioral Cloning from expert trajectories, and a simplified Maximum-Entropy IRL that recovers a reward function, from scratch)
- G13. Offline & Hierarchical RL (Batch/Offline Q-learning from a fixed, frozen dataset with importance-weighting; the Options framework explained conceptually and demonstrated with a hand-built two-level policy — from scratch)
- G14. Partially Observable MDPs (POMDPs) (belief-state updates and a belief-space value function on a small hand-built POMDP, from scratch; cross-reference E06's Kalman/Particle filters)

### Module H — Evolutionary & Nature-Inspired Optimization
- H01. Genetic Algorithms ★ (selection/crossover/mutation solving the Traveling Salesman Problem)
- H02. Genetic Programming (tree-structured program evolution, minimal symbolic-regression example)
- H03. Evolution Strategies (simplified CMA-ES from scratch)
- H04. Particle Swarm Optimization ★
- H05. Ant Colony Optimization ★ (TSP via pheromone trails)
- H06. Simulated Annealing
- H07. Differential Evolution
- H08. NEAT — NeuroEvolution of Augmenting Topologies (conceptual + simplified from-scratch topology mutation)
- H09. Multi-Objective Evolutionary Optimization — NSGA-II ★ (non-dominated sorting, crowding distance, and Pareto-front construction, from scratch — the standard answer to "what if I have two conflicting objectives" that nothing else in this module addresses)

### Module I — Classical Optimization Theory
- I01. Gradient Descent Optimizer Family ★ (derive and hand-implement Momentum, Nesterov, AdaGrad, RMSProp, and Adam's update rules yourself — not `torch.optim.Adam`; visualize each on the same loss surface)
- I02. Newton's Method & Quasi-Newton (BFGS/L-BFGS from scratch)
- I03. Constrained Optimization (Lagrange multipliers and KKT conditions, worked from-scratch numeric examples)
- I04. Convex Optimization & Duality (basic theory + a from-scratch projected-gradient example)
- I05. Linear Programming — the Simplex Method (from scratch)

### Module J — Other Learning Paradigms
- J01. Semi-Supervised Learning (self-training and label propagation, from scratch)
- J02. Self-Supervised Contrastive Learning (derive and code the InfoNCE/SimCLR loss yourself on a tiny hand-built encoder)
- J03. Meta-Learning — MAML (from scratch on a toy sine-wave few-shot regression task)
- J04. Federated Learning — FedAvg (from-scratch simulation of multiple clients training locally and averaging weights)
- J05. Online Learning & Concept Drift (online Perceptron updates; ADWIN drift-detection concept)
- J06. Continual / Lifelong Learning — Elastic Weight Consolidation (from scratch, demonstrating and then mitigating catastrophic forgetting)
- J07. Causal Inference (structural causal models, propensity score matching, difference-in-differences — from scratch)
- J08. Anomaly Detection (Isolation Forest and statistical z-score/IQR methods, from scratch)
- J09. Recommender Systems ★ (user/item collaborative filtering and matrix factorization via SGD, from scratch)
- J10. Explainable AI (permutation feature importance and Shapley values computed by brute-force coalition enumeration on a small feature set, from scratch; LIME's local-surrogate idea reimplemented minimally)
- J11. Active Learning (uncertainty sampling and query-by-committee on a pool of unlabeled data, from scratch — how a model picks which examples are worth labeling next)
- J12. Multi-Task & Transfer Learning (hard parameter sharing on a hand-built shared-encoder MLP with multiple output heads, plus classical feature-based transfer, from scratch — distinct from the fine-tuning/PEFT-based transfer already covered in `Transformer/Reports/Deep_Dives/42_Comprehensive_FineTuning_Mastery.md`)

### Module K — Neural-Net Techniques Missing From `Transformer/`
- K01. Hopfield Networks & Modern Hopfield Networks (energy-based associative memory — explicitly cross-reference the mathematical link to self-attention)
- K02. Normalizing Flows (RealNVP-style affine coupling layers, from-scratch invertible transform + log-det-Jacobian)
- K03. Neural ODEs (hand-written Euler and RK4 solvers driving a tiny learned derivative function — no `torchdiffeq`)
- K04. Capsule Networks (dynamic routing-by-agreement, from scratch)
- K05. Physics-Informed Neural Networks (PINNs) (residual of a known ODE/PDE as part of the loss, from scratch)
- K06. Classical Word Embeddings — Word2Vec (Skip-gram with Negative Sampling, hand-derived and hand-coded, on a tiny toy corpus; explicitly cross-reference this as the shallow, non-contextual ancestor of the embedding lookup tables that every `Transformer/` model begins with)
- K07. Reservoir Computing — Echo State Networks (a large, fixed, randomly-connected recurrent reservoir with only the linear readout layer trained via least squares, from scratch — the "don't train the recurrence at all" counterpoint to every other recurrent architecture in `Transformer/` and this repo, where the reservoir's echo-state property is the entire mechanism; cross-reference `Transformer/Reports/Deep_Dives/10_LNN_Liquid.md` since LNNs are the trainable, continuous-time descendant of this same fixed-reservoir intuition)

### Module L — Classical AI Search & Symbolic Reasoning
- L01. Uninformed Search (BFS, DFS, Iterative Deepening on a state graph)
- L02. Informed Search — A* ★ (heuristic design, from scratch on a grid/maze)
- L03. Adversarial Search — Minimax & Alpha-Beta Pruning ★ (Tic-Tac-Toe or Connect-4, from scratch; note the direct bridge to G10's MCTS)
- L04. Constraint Satisfaction Problems (backtracking + AC-3 arc-consistency, from scratch on N-Queens/Sudoku)
- L05. Fuzzy Logic Systems (membership functions, Mamdani inference, from scratch)
- L06. Rule-Based Expert Systems (forward-chaining and backward-chaining mini inference engine, from scratch)
- L07. Classical Graph Algorithms for ML ★ (Dijkstra, PageRank via power iteration, and spectral clustering — all from scratch, no `networkx`; explicitly cross-reference GNNs in `Transformer/`)
- L08. Game Theory & Multi-Agent Equilibria (Nash equilibrium via best-response iteration / fictitious play, and the minimax theorem for zero-sum games, from scratch — generalizes L03's two-player zero-sum minimax to general-sum games; cross-reference G11's Multi-Agent RL)

**Total: 94 topics across 12 modules** (A:3, B:9, C:10, D:4, E:9, F:4, G:14, H:9, I:5, J:12, K:7, L:8 — count this yourself against §6 before starting; earlier drafts of this prompt under-counted this as "~79" and, in §10, as "~65").

---

## 7. Required Cross-Reference Table

Build `CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md` as a table like this (expand it — this is a starter, not exhaustive):

| `Algorithm_Codex/` topic | Related `Transformer/` topic | The connection |
|---|---|---|
| G08 PPO (control) | `Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md` | Same clipped-objective algorithm, applied to reward-model alignment instead of a control task |
| K01 Hopfield Networks | `Sims/00_Self_Attention_Mechanism.html` | Modern continuous Hopfield networks are mathematically a generalized form of self-attention |
| C04 EM Algorithm / GMM | `Reports/Deep_Dives/05_Autoencoders_VAE.md` | The VAE's reparameterization trick is a differentiable relaxation of the same latent-variable idea EM optimizes exactly |
| L07 PageRank / Spectral Clustering | `Reports/Deep_Dives/08_GNN_Graph.md` | GNN message passing is a learned generalization of these fixed, hand-derived graph propagation rules |
| I01 Adam from scratch | `Reports/Deep_Dives/00_The_Complete_Mathematics_of_Neural_Networks.md` | That report uses Adam as a black box optimizer; this module derives what it's actually doing |
| B06 SVM kernel trick | `Reports/Deep_Dives/20_KAN_Kolmogorov_Arnold.md` | Both replace a fixed linear boundary with a richer learned/kernel-induced function space |
| E02 HMM | `Reports/Deep_Dives/03_RNN_Recurrent.md` | HMMs are the classical, exactly-tractable ancestor of RNN-style sequence modeling |
| K06 Word2Vec (Skip-gram) | `Reports/Deep_Dives/29_Tokenization_and_Data_Curation.md` | Word2Vec is the shallow, non-contextual ancestor of the embedding lookup tables every Transformer begins with |
| E09 Bayesian Optimization | `Reports/Deep_Dives/24_Scaling_Laws_and_RLHF.md` | Scaling-laws work treats compute/hyperparameter trade-offs empirically; Bayesian Optimization is the principled search procedure that could tune them |
| G14 POMDPs (belief-state update) | `Reports/Deep_Dives/03_RNN_Recurrent.md` | A hand-derived belief-state recurrence is the classical analogue of what an RNN's hidden state learns implicitly and differentiably |
| G12 Imitation Learning | `Reports/Deep_Dives/17_Transformer_Agentic_Reasoning.md` | Behavioral Cloning is supervised imitation of expert action sequences — the same "learn from demonstrations" idea underlying agentic SFT, minus the language-model scale |
| C09 Isomap / LLE | `Reports/Deep_Dives/05_Autoencoders_VAE.md` | Both learn a low-dimensional manifold; Isomap/LLE fix the embedding by hand-derived geometry, autoencoders learn it end-to-end |

---

## 8. Supporting Meta-Files

- **`STUDY_ROADMAP.md`** — same phased structure as `Transformer/STUDY_ROADMAP.md`: group Modules A→L into ordered phases (e.g. Phase 0: Stats foundations → Phase 1: Classical supervised/unsupervised → Phase 2: Ensembles & PGMs → Phase 3: Optimization theory → Phase 4: Evolutionary computation → Phase 5: RL end-to-end → Phase 6: everything else), each step linking its Report + Code pair and stating a concrete "Task" the reader must complete before moving on.
- **`ALGORITHM_CHEAT_SHEET.md`** — one dense table: algorithm | category | time complexity | space complexity | when to use | when it fails | closest neural analogue in `Transformer/` (if any).
- **`SELF_ASSESSMENT_QUIZ.md`** — mirror `Transformer/SELF_ASSESSMENT_QUIZ.md`'s format; at least one question per module.
- **`PAPERS_AND_TEXTBOOKS_CATALOG.md`** — mirror `Transformer/PAPERS_TO_READ_CATALOG.md`; for classical ML/RL this should lean on textbooks as much as papers (Sutton & Barto for RL, Bishop's *Pattern Recognition and Machine Learning*, Murphy's *Probabilistic ML*, Hastie/Tibshirani/Friedman's *Elements of Statistical Learning*), plus the original papers for each named algorithm (Vapnik for SVM, Breiman for Random Forests, Watkins for Q-learning, Rechenberg/Schwefel for Evolution Strategies, Holland for Genetic Algorithms, etc.).

---

## 9. Acceptance Checklist (verify before considering this done)

- [ ] Every topic in §6 has both a `Reports/Deep_Dives/*.md` and a `Code/Deep_Dives/*.py` file, correctly numbered and named.
- [ ] No file in `Code/Deep_Dives/` imports anything from §3.2's banned list, or `torch`/`tensorflow`/`jax`/any autodiff engine (§3.3.1), to solve its core task (a library may appear only inside an §3.3-style trailing appendix, if at all).
- [ ] Every `.py` file runs standalone via `python <file>.py` and prints a numeric sanity check, not just "done."
- [ ] `Code/verify_all.py` runs every file and reports a clean pass count.
- [ ] Every report follows the 5-part template in §5.1, including the math derivation section with real LaTeX.
- [ ] At least 12 topics have a working, self-contained, dependency-free `Sims/*.html` visualizer.
- [ ] `STUDY_ROADMAP.md`, `ALGORITHM_CHEAT_SHEET.md`, `SELF_ASSESSMENT_QUIZ.md`, `PAPERS_AND_TEXTBOOKS_CATALOG.md`, and `CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md` all exist and are populated, not stubs.
- [ ] Nothing here re-implements a topic that already exists in `Transformer/` (check against §1 before writing each file).

---

## 10. Suggested Delivery Order

Given the size (94 topics × 2 files + metafiles + sims), build and deliver in this batch order so early batches are independently useful:
1. Meta-files skeleton (`STUDY_ROADMAP.md` with all modules listed, even before content exists) + Module A + Module B
2. Module C + Module D
3. Module I (optimization theory underpins everything above — good to have early) + Module E
4. Module G (Reinforcement Learning — largest module)
5. Module H + Module F
6. Module J + Module K
7. Module L
8. Sims batch (12–15 visualizers), `verify_all.py`, cheat sheet, quiz, papers catalog, cross-reference table — final polish pass.

Confirm each batch runs cleanly (`verify_all.py`-style check) before moving to the next.

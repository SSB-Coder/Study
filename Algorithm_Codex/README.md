# Algorithm Codex: Foundations of Machine Learning, Optimization & Symbolic Intelligence

> **A rigorous, from-scratch algorithmic encyclopedia and companion repository to [`Transformer`](../Transformer).**
>
> 174 canonical algorithms implemented from pure first principles in NumPy with matching hardcore mathematical derivations, algorithmic step-by-step walkthroughs, failure-mode pathology analyses, and interactive HTML5 visualizers. Zero third-party machine learning libraries.

---

## 🏛️ Repository Overview & Architecture

Modern AI didn't begin with the Transformer attention mechanism. Every neural advance—from self-attention to RLHF, PPO, DPO, test-time MCTS reasoning, and high-dimensional vector retrieval—rests upon a bedrock of classical statistical theory, convex optimization, dynamic programming, and search algorithms developed over seven decades.

`Algorithm_Codex` is built to mirror the depth, rigor, and clarity of the `Transformer` project. It contains:

- **174 Deep Dive Mathematical Reports** (`Reports/Deep_Dives/`): 5-part structure covering intuitive analogies, architectural breakdowns, rigorous LaTeX proofs, algorithmic walkthroughs, failure modes, and modern Transformer/LLM cross-references.
- **174 Pure NumPy Implementations** (`Code/Deep_Dives/`): Object-oriented, dependency-free implementations equipped with comprehensive numerical sanity checks and assertions.
- **2 Foundational Primer Reports** (`Reports/`): Classical vs. Deep Learning taxonomy and a comprehensive Probability & Statistics primer.
- **Interactive Simulation Lab** (`Sims/`): 34 standalone, zero-dependency HTML5 Canvas visualizers linked through an interactive gallery (`Sims/index.html`).
- **Master Verification Test Suite** (`Code/verify_all.py`): Automated test harness testing all 174 algorithm files end-to-end (**100% PASS RATE: 174/174**).
- **5 Comprehensive Meta-Guides**:
  - [`STUDY_ROADMAP.md`](STUDY_ROADMAP.md): Tiered learning curriculum spanning Foundations to Mastery across 26 modules.
  - [`ALGORITHM_CHEAT_SHEET.md`](ALGORITHM_CHEAT_SHEET.md): Dense 174-row reference matrix with time/space complexities, assumptions, and modern LLM relevance.
  - [`SELF_ASSESSMENT_QUIZ.md`](SELF_ASSESSMENT_QUIZ.md): Comprehensive diagnostic conceptual, mathematical, and coding questions with detailed solutions.
  - [`PAPERS_AND_TEXTBOOKS_CATALOG.md`](PAPERS_AND_TEXTBOOKS_CATALOG.md): 90+ seminal textbook and paper citations.
  - [`CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md`](CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md): Exhaustive conceptual mapping linking classical techniques to modern LLM mechanisms.

---

## 📊 Complete 26-Module Taxonomy (A through Z: 174 Topics)

```
Algorithm_Codex/
├── Reports/
│   ├── 00_Foundations_Classical_vs_Deep_Learning.md
│   ├── 00_Probability_and_Statistics_Primer.md
│   └── Deep_Dives/ (174 detailed .md reports)
├── Code/
│   ├── verify_all.py (Master Test Suite)
│   └── Deep_Dives/ (174 self-testing pure NumPy .py scripts)
├── Sims/ (34 Canvas HTML visualizers + index.html hub)
├── STUDY_ROADMAP.md
├── ALGORITHM_CHEAT_SHEET.md
├── SELF_ASSESSMENT_QUIZ.md
├── PAPERS_AND_TEXTBOOKS_CATALOG.md
└── CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md
```

### Module Breakdown

| Module | Name | Topics | Status | Key Highlights |
| :---: | :--- | :---: | :---: | :--- |
| **A** | **Statistical Foundations** | 3 | ✅ 3/3 Pass | Probability Primitives, Bias-Variance, Hypothesis Testing |
| **B** | **Classical Supervised Learning** | 9 | ✅ 9/9 Pass | Linear/Logistic Reg., k-NN, Naive Bayes, Trees, SVM, LDA/QDA, GLM, Calibration |
| **C** | **Unsupervised & Dimensionality** | 10 | ✅ 10/10 Pass | k-Means, Hierarchical, DBSCAN, GMM/EM, PCA, ICA, t-SNE, Apriori, Isomap/LLE, UMAP |
| **D** | **Ensemble Methods** | 4 | ✅ 4/4 Pass | Random Forests, AdaBoost, Gradient Boosting (GBM), Stacking/Blending |
| **E** | **Graphical & Bayesian Models** | 9 | ✅ 9/9 Pass | Bayes Nets, HMM, MCMC, Variational Inference, GP, Kalman/Particle, CRF, LDA, BayesOpt |
| **F** | **Time Series & Sequential Models**| 4 | ✅ 4/4 Pass | Holt-Winters Decomposition, ARMA/ARIMA, Survival Analysis, Dynamic Time Warping |
| **G** | **Reinforcement Learning** | 14 | ✅ 14/14 Pass | Bandits, MDP/DP, MC Control, TD/SARSA/Q, DQN, REINFORCE, A2C, PPO, DDPG, Dyna-Q, MARL, MaxEnt IRL, CQL, POMDP |
| **H** | **Evolutionary & Metaheuristics** | 9 | ✅ 9/9 Pass | GA, Genetic Prog., CMA-ES, Differential Evolution, PSO, Annealing, ACO, NEAT, NSGA-II |
| **I** | **Classical Optimization Theory**| 5 | ✅ 5/5 Pass | Gradient Descent Family, Newton/BFGS, KKT Conditions, Convex Duality, Simplex LP |
| **J** | **Other Learning Paradigms** | 12 | ✅ 12/12 Pass | SSL, InfoNCE, MAML, FedAvg, ADWIN, EWC, Causal Inf., Isolation Forest, RecSys, SHAP/LIME, Active Learning, Multi-Task |
| **K** | **Specialized Neural Architectures**| 7 | ✅ 7/7 Pass | Hopfield, Normalizing Flows, Neural ODE, Capsule Networks, PINNs, Word2Vec SGNS, ESN |
| **L** | **AI Search & Symbolic Reasoning** | 8 | ✅ 8/8 Pass | BFS/DFS/UCS, A* Search, Minimax Alpha-Beta, CSP/AC-3, Fuzzy Logic, Expert Systems, Dijkstra/PageRank, Game Theory |
| **M** | **High-Dim Vector Search & Streaming** | 4 | ✅ 4/4 Pass | HNSW, Locality-Sensitive Hashing, Space Partitioning (k-d/Ball), Streaming Sketches |
| **N** | **Computational Graph & Autodiff** | 3 | ✅ 3/3 Pass | Reverse-Mode Tape DAG, Forward Dual Numbers (JVP), Symbolic Trees & Simplification |
| **O** | **Optimal Transport & Geometry** | 3 | ✅ 3/3 Pass | Earth Mover's Distance, Sinkhorn Entropic Regularization, Information Geometry & FIM |
| **P** | **Physics-Based Sampling & Diffusion**| 3 | ✅ 3/3 Pass | Hamiltonian Monte Carlo Leapfrog, Langevin SDE & MALA, Energy Models & Score Matching |
| **Q** | **Multi-Agent Systems & Mechanisms** | 3 | ✅ 3/3 Pass | Value Decomposition & QMIX, MADDPG CTDE, Algorithmic Mechanism Design & VCG Auctions |
| **R** | **Quantum Computing** | 8 | ✅ 8/8 Pass | Statevector Simulator, Quantum Gates, Teleportation, Deutsch-Jozsa, Grover, Shor QFT, VQE, QPE |
| **S** | **Computational Physics & Mechanics** | 6 | ✅ 6/6 Pass | Barnes-Hut N-Body, Velocity Verlet, Ising Model, Lattice Boltzmann Fluid, Split-Operator Schrödinger, Orbital Mechanics |
| **T** | **Computational Biology** | 6 | ✅ 6/6 Pass | Needleman-Wunsch/Smith-Waterman, BWT & FM-Index, Molecular Fingerprints, Force Fields, Coalescent, Gray-Scott Reaction-Diffusion |
| **U** | **Signal Processing & Control Theory** | 6 | ✅ 6/6 Pass | Radix-2 FFT, FIR/IIR Butterworth Filters, PID with Anti-Windup, State-Space LQR via CARE, SPICE MNA, Hamming/Reed-Solomon |
| **V** | **Numerical Linear Algebra** | 5 | ✅ 5/5 Pass | QR Householder/Gram-Schmidt, LU Partial Pivoting, Shifted QR Eigenvalues, Gauss-Legendre Quadrature, Brent-Dekker Root Finder |
| **W** | **EDA & Engineering Optimization** | 14 | ✅ 14/14 Pass | Lee Maze, Steiner Tree, SA Placement, Floorplanning, Min-Cut, Quine-McCluskey, BDD, STA, DPLL SAT, FEM Truss, SIMP, Engine Cycles, Kriging |
| **X** | **Robotics & Computer Vision** | 11 | ✅ 11/11 Pass | RRT, RRT*, PRM, EKF, UKF, EKF SLAM, FastSLAM, Kinematics Jacobian, RANSAC, Canny Edge, Lucas-Kanade Optical Flow |
| **Y** | **Lie Groups & Advanced Topology** | 3 | ✅ 3/3 Pass | Lie Groups SO(3)/SE(3) & Equivariance, Topological Data Analysis Persistent Homology, Gromov-Wasserstein & Sliced OT |
| **Z** | **Geometry, Crypto & Causal Discovery**| 5 | ✅ 5/5 Pass | Convex Hull & Delaunay, Zero-Knowledge Proofs & R1CS, Elliptic Curves & Merkle Trees, Huffman & LZW & NCD, PC Causal Discovery |
| **Total** | **All 26 Modules (A through Z)** | **174** | **✅ 174/174** | **100% From-Scratch Pure NumPy & Python Verification** |

---

## ⚡ Quick Start & Verification

### Running the Test Suite
All 174 algorithm files are self-contained and run in standard Python 3 with pure NumPy:

```bash
# Verify all 174 algorithms end-to-end
python Code/verify_all.py
```

Expected output:
```
================================================================================
FINAL AUDIT SUMMARY: 174/174 PASSED | 0 FAILED | Total Time: ~55s
================================================================================
ALL 174 ALGORITHMS VERIFIED AND FULLY FUNCTIONAL WITH ZERO FAILURES!
```

---

## 🎮 Interactive Simulation Lab

Open [`Sims/index.html`](Sims/index.html) in any modern web browser to interact with all 34 visualizers:

1. **Linear Regression Gradient Descent**: Real-time loss descent and dynamic data point addition.
2. **k-Means & Voronoi Partitions**: Dynamic centroid movement and cluster territory visualization.
3. **CART Decision Trees**: Orthogonal Gini cuts carving up 2D coordinate spaces.
4. **Support Vector Machines (SVM)**: Maximum margin hyperplane with interactive margin tuning.
5. **Principal Component Analysis (PCA)**: Principal eigenvector decomposition of 2D point clouds.
6. **Markov Decision Process GridWorld**: Dynamic programming value heatmap with greedy policy arrows.
7. **Multi-Armed Bandits & UCB1**: Slot machine pulls balancing confidence intervals and regret.
8. **A* Heuristic Pathfinding**: Real-time grid pathfinding around user-drawn walls.
9. **Genetic Algorithm TSP**: Natural selection and 2-opt order crossover evolving shortest city tours.
10. **Particle Swarm Optimization (PSO)**: Multi-particle swarm converging on Rastrigin minima.
11. **Hopfield Associative Memory**: Energy descent restoring noisy, corrupted 8x8 pixel patterns.
12. **Recommender Systems Matrix Factorization**: Real-time SGD low-rank matrix decomposition predicting missing ratings.
13. **Graph Algorithms (Dijkstra & PageRank)**: Interactive directed graph computing shortest paths and stationary probability mass.
14. **HNSW Vector Search**: Hierarchical Navigable Small World multi-layer skip-graph routing from highway to ground Voronoi cells.
15. **Computational Graph & Tape Autodiff**: Interactive DAG execution visualizing topological sorting, forward primal flow, and reverse-mode adjoint backprop.
16. **Entropic Optimal Transport & Sinkhorn**: 2D distribution transport visualizer morphing entropic fog into sparse Wasserstein mass matching couplings.
17. **Barnes-Hut N-Body Gravity**: 2D quadtree spatial decomposition simulating galactic orbital clustering.
18. **Lattice Boltzmann Fluid Dynamics**: D2Q9 mesoscopic fluid lattice simulating flow around an obstacle.
19. **Schrödinger Quantum Wavepacket**: Split-operator FFT propagation tunneling through potential barriers.
20. **Gray-Scott Turing Reaction-Diffusion**: Chemical morphogenesis generating organic spots and stripes.
21. **Fourier Transform & Frequency Filtering**: Real-time signal decomposition and bandpass filtering.
22. **PID Feedback Control Plant**: Interactive second-order dynamical plant with anti-windup clamping.
23. **SPICE Circuit Solver**: Modified Nodal Analysis solving voltages on user-wired resistive circuits.
24. **Gram-Schmidt & QR Decomposition**: Progressive orthogonalization of column vectors onto orthonormal hyperplanes.
25. **Lee Algorithm Maze Router**: Wavefront expansion finding shortest VLSI interconnects around blockages.
26. **Force-Directed Graph Placement**: Spring-electrical mechanical equilibrium layout optimization.
27. **Quine-McCluskey Logic Minimization**: Prime implicant tabular reduction for Boolean logic.
28. **SIMP Topology Optimization**: 2D structural density optimization maximizing cantilever stiffness.
29. **RRT Path Planning**: Rapidly-exploring Random Tree 2D trajectory exploration with obstacle avoidance.
30. **EKF SLAM**: Mobile robot mapping landmarks with shrinking covariance uncertainty ellipses.
31. **RANSAC Outlier Rejection**: Extreme outlier consensus linear regression.
32. **Canny Edge Detection**: Multi-stage gradient convolution, non-maximum suppression, and hysteresis edges.
33. **SE(3) Rigid-Body Lie Group Manifold**: 3D twist coordinates, exponential Rodrigues map, and geodesic screw motion.
34. **Convex Hull & Delaunay Triangulation**: Interactive 2D spatial playground computing live Graham scan convex hulls and Bowyer-Watson Delaunay triangulations on user-clicked vertices.

---

## 🔗 Cross-Reference: Connecting Classical AI to Modern Transformers

For detailed mappings between classical concepts and modern LLM/Transformer architectures, see [`CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md`](CROSS_REFERENCE_WITH_TRANSFORMER_REPO.md):

- **Self-Attention as Modern Hopfield Networks**: Softmax attention $\text{softmax}(Q K^\top / \sqrt{d_k}) V$ is a continuous Hopfield retrieval step with exponential energy functions.
- **RLHF Alignment & PPO**: Modern alignment (ChatGPT, Claude, Gemini) relies directly on Actor-Critic Advantage Estimation (GAE) and PPO clipped surrogate bounds.
- **Reasoning Models (o1 / DeepSeek-R1) as MCTS**: Test-time compute scaling is a direct manifestation of Monte Carlo Tree Search exploring branching chains of thought.
- **SE(3) Equivariant Transformers**: AlphaFold and biomolecular foundation models constrain attention weights to commute with Lie group transformations.
- **Verifiable Computation (zk-SNARKs)**: Zero-knowledge arithmetic circuits enable cryptographically verifiable private AI inference without disclosing weights or user prompts.
- **Prediction as Universal Compression**: Byte-Pair Encoding (BPE) and next-token prediction are the modern operationalization of LZW dictionary compression and algorithmic information theory.
- **Mechanistic Interpretability as Causal Discovery**: Causal mediation analysis and causal scrubbing reconstruct the true causal DAG of attention heads and circuit features, separating genuine causal mechanisms from passive correlational activations.

# Comprehensive Study Roadmap: The Classical Algorithm Codex

Welcome to the **Algorithm Codex**, the rigorous, from-scratch curriculum covering classical machine learning, reinforcement learning, probabilistic graphical models, nature-inspired heuristics, optimization theory, time series, and symbolic reasoning.

This roadmap organizes all **174 algorithms across 26 modules (A through Z)** into structured learning phases. Every topic contains a pair of files:
- **Theory Deep Dive:** `Reports/Deep_Dives/<ID>_<Name>.md` (Full derivation in LaTeX, intuition, step-by-step algorithms, failure modes, and architectural connections).
- **From-Scratch Implementation:** `Code/Deep_Dives/<ID>_<Name>.py` (Hand-derived NumPy matrix calculus, zero third-party ML frameworks, zero autodiff, verifiable numeric sanity checks).

---

## Phase 0: Foundations & Probabilistic Primitives
*Objective: Master point estimation, empirical resampling, and raw statistical hypothesis testing.*

1. **A01: Probability & Estimation Primitives**
   - Files: [[Reports/Deep_Dives/A01_Probability_and_Estimation_Primitives|Report]] | [[Code/Deep_Dives/A01_Probability_and_Estimation_Primitives.py|Code]]
   - **Task:** Derive the likelihood and posterior for Bernoulli and Gaussian variables; code the closed-form MLE and MAP estimators in NumPy and verify that as prior sample size increases, MAP shrinks to the prior mean.
2. **A02: Bias-Variance Decomposition & Resampling**
   - Files: [[Reports/Deep_Dives/A02_Bias_Variance_Decomposition_and_Resampling|Report]] | [[Code/Deep_Dives/A02_Bias_Variance_Decomposition_and_Resampling.py|Code]]
   - **Task:** Write an index splitter for $k$-fold cross-validation, leave-one-out (LOOCV), and non-parametric bootstrap resampling; verify empirical variance against theoretical asymptotic bounds.
3. **A03: Hypothesis Testing Toolkit**
   - Files: [[Reports/Deep_Dives/A03_Hypothesis_Testing_Toolkit|Report]] | [[Code/Deep_Dives/A03_Hypothesis_Testing_Toolkit.py|Code]]
   - **Task:** Compute Student's two-sample $t$-statistic, Pearson's $\chi^2$ contingency test, and an exact permutation test from raw sum formulas without importing `scipy.stats`.

---

## Phase 1: Classical Supervised Learning
*Objective: Build linear, non-linear, non-parametric, and margin-maximizing classifiers and regressors from mathematical first principles.*

4. **B01: Linear Regression**
   - Files: [[Reports/Deep_Dives/B01_Linear_Regression|Report]] | [[Code/Deep_Dives/B01_Linear_Regression.py|Code]]
   - **Task:** Code the Normal Equation $\hat{\beta} = (X^T X)^{-1} X^T y$, batch gradient descent, $L_2$ Ridge regression, and coordinate descent soft-thresholding for $L_1$ Lasso. Verify that gradient descent converges to the Normal Equation solution.
5. **B02: Logistic Regression**
   - Files: [[Reports/Deep_Dives/B02_Logistic_Regression|Report]] | [[Code/Deep_Dives/B02_Logistic_Regression.py|Code]]
   - **Task:** Derive the binary cross-entropy gradient $\nabla_w \mathcal{L} = X^T(\sigma(Xw) - y)$ and the multinomial Softmax gradient; train both via batch gradient descent on synthetic classification blobs.
6. **B03: k-Nearest Neighbors & KD-Trees**
   - Files: [[Reports/Deep_Dives/B03_k_Nearest_Neighbors|Report]] | [[Code/Deep_Dives/B03_k_Nearest_Neighbors.py|Code]]
   - **Task:** Construct a spatial KD-Tree recursively by splitting data on median coordinate values along alternating dimensions; implement branch-and-bound $k$-NN search and cross-verify results against brute-force pairwise distance matrix.
7. **B04: Naive Bayes**
   - Files: [[Reports/Deep_Dives/B04_Naive_Bayes|Report]] | [[Code/Deep_Dives/B04_Naive_Bayes.py|Code]]
   - **Task:** Implement Gaussian Naive Bayes (computing class mean and variance vectors), Multinomial Naive Bayes with additive Laplace smoothing, and Bernoulli Naive Bayes.
8. **B05: Decision Trees (CART)**
   - Files: [[Reports/Deep_Dives/B05_Decision_Trees|Report]] | [[Code/Deep_Dives/B05_Decision_Trees.py|Code]]
   - **Task:** Implement recursive tree splitting evaluating Gini impurity and Information Gain over continuous features; implement minimal cost-complexity post-pruning parameter $\alpha$.
9. **B06: Support Vector Machines & SVR**
   - Files: [[Reports/Deep_Dives/B06_Support_Vector_Machines|Report]] | [[Code/Deep_Dives/B06_Support_Vector_Machines.py|Code]]
   - **Task:** Implement a simplified Sequential Minimal Optimization (SMO) solver for the soft-margin dual SVM with Linear, Polynomial, and RBF kernels; implement Support Vector Regression with $\epsilon$-insensitive loss.
10. **B07: Linear & Quadratic Discriminant Analysis**
    - Files: [[Reports/Deep_Dives/B07_Linear_and_Quadratic_Discriminant_Analysis|Report]] | [[Code/Deep_Dives/B07_Linear_and_Quadratic_Discriminant_Analysis.py|Code]]
    - **Task:** Derive and code the pooled covariance matrix for LDA and class-specific covariance matrices for QDA; verify decision boundaries against Bayes optimal boundaries.
11. **B08: Generalized Linear Models (GLM) & IRLS**
    - Files: [[Reports/Deep_Dives/B08_Generalized_Linear_Models|Report]] | [[Code/Deep_Dives/B08_Generalized_Linear_Models.py|Code]]
    - **Task:** Formulate Poisson regression with log link and Gamma regression with inverse link; solve parameters using Iteratively Reweighted Least Squares (IRLS / Fisher scoring).
12. **B09: Probability Calibration**
    - Files: [[Reports/Deep_Dives/B09_Probability_Calibration|Report]] | [[Code/Deep_Dives/B09_Probability_Calibration.py|Code]]
    - **Task:** Code Platt Scaling (logistic regression on uncalibrated margins) and Isotonic Regression via the Pool Adjacent Violators Algorithm (PAVA). Measure Brier score reduction.

---

## Phase 2: Unsupervised Learning & Dimensionality Reduction
*Objective: Uncover hidden clusters, latent subspaces, and non-linear data manifolds.*

13. **C01: k-Means & k-Means++**
    - Files: [[Reports/Deep_Dives/C01_k_Means_and_k_Means_Plus_Plus|Report]] | [[Code/Deep_Dives/C01_k_Means_and_k_Means_Plus_Plus.py|Code]]
    - **Task:** Implement $D^2$-weighted centroid seeding (k-means++), Lloyd's coordinate descent iterations, inertia calculation for the elbow method, and pairwise silhouette score.
14. **C02: Hierarchical Agglomerative Clustering**
    - Files: [[Reports/Deep_Dives/C02_Hierarchical_Agglomerative_Clustering|Report]] | [[Code/Deep_Dives/C02_Hierarchical_Agglomerative_Clustering.py|Code]]
    - **Task:** Build bottom-up agglomerative clustering supporting Single, Complete, Average, and Ward linkage from scratch; output the step-by-step merge matrix.
15. **C03: DBSCAN**
    - Files: [[Reports/Deep_Dives/C03_DBSCAN|Report]] | [[Code/Deep_Dives/C03_DBSCAN.py|Code]]
    - **Task:** Implement $\epsilon$-neighborhood queries, Core/Border/Noise point labeling, and density-connected cluster expansion.
16. **C04: Gaussian Mixture Models & Expectation-Maximization (EM)**
    - Files: [[Reports/Deep_Dives/C04_Gaussian_Mixture_Models_and_EM|Report]] | [[Code/Deep_Dives/C04_Gaussian_Mixture_Models_and_EM.py|Code]]
    - **Task:** Derive and code the E-step (posterior responsibilities $\gamma_{ik}$) and M-step (updating mixing weights $\pi_k$, mean vectors $\mu_k$, and full covariance matrices $\Sigma_k$). Verify monotonic ascent of the log-likelihood.
17. **C05: Principal Component Analysis (PCA)**
    - Files: [[Reports/Deep_Dives/C05_Principal_Component_Analysis|Report]] | [[Code/Deep_Dives/C05_Principal_Component_Analysis.py|Code]]
    - **Task:** Compute PCA via sample covariance eigendecomposition $\Sigma = V \Lambda V^T$ and via data matrix Singular Value Decomposition (SVD) $X = U S V^T$. Prove numerically that both approaches yield identical projected coordinates.
18. **C06: Independent Component Analysis (FastICA)**
    - Files: [[Reports/Deep_Dives/C06_Independent_Component_Analysis|Report]] | [[Code/Deep_Dives/C06_Independent_Component_Analysis.py|Code]]
    - **Task:** Prewhiten mixed signals via PCA; implement the fixed-point iteration maximizing negentropy using the non-quadratic approximation $g(u) = \tanh(u)$. Separate mixed audio/synthetic signals.
19. **C07: t-Distributed Stochastic Neighbor Embedding (t-SNE)**
    - Files: [[Reports/Deep_Dives/C07_t_SNE|Report]] | [[Code/Deep_Dives/C07_t_SNE.py|Code]]
    - **Task:** Compute high-dimensional conditional probabilities $p_{j|i}$ with bisection search for perplexity; define low-dimensional Student-t probabilities $q_{ij}$; derive and hand-code the KL divergence gradient $\frac{\partial \mathcal{L}}{\partial y_i}$ with momentum updates.
20. **C08: Association Rule Mining (Apriori & FP-Growth)**
    - Files: [[Reports/Deep_Dives/C08_Association_Rule_Mining|Report]] | [[Code/Deep_Dives/C08_Association_Rule_Mining.py|Code]]
    - **Task:** Implement candidate generation with downward-closure pruning (Apriori) and an FP-Tree data structure with recursive conditional FP-tree mining (FP-Growth). Compute Support, Confidence, and Lift.
21. **C09: Manifold Learning — Isomap & LLE**
    - Files: [[Reports/Deep_Dives/C09_Manifold_Learning_Isomap_and_LLE|Report]] | [[Code/Deep_Dives/C09_Manifold_Learning_Isomap_and_LLE.py|Code]]
    - **Task:** Construct a nearest-neighbor graph, compute shortest paths via Dijkstra/Floyd-Warshall to approximate geodesic distances, and embed via Classical MDS (Isomap); implement constrained local linear weight reconstruction and null-space embedding (LLE).
22. **C10: UMAP (Uniform Manifold Approximation & Projection)**
    - Files: [[Reports/Deep_Dives/C10_UMAP|Report]] | [[Code/Deep_Dives/C10_UMAP.py|Code]]
    - **Task:** Construct fuzzy simplicial sets with local metric scaling $\sigma_i$ and minimum distance $\rho_i$; minimize fuzzy set cross-entropy in the embedding space via raw NumPy gradient descent.

---

## Phase 3: Ensembles & Classical Optimization Theory
*Objective: Combine models into robust ensembles and derive foundational convex and non-convex optimizers.*

23. **D01: Bagging & Random Forests**
    - Files: [[Reports/Deep_Dives/D01_Bagging_and_Random_Forests|Report]] | [[Code/Deep_Dives/D01_Bagging_and_Random_Forests.py|Code]]
    - **Task:** Build an ensemble of bootstrap-aggregated B05 decision trees with random feature subsampling ($\\sqrt{d}$); compute out-of-bag (OOB) error estimates without test set splits.
24. **D02: AdaBoost**
    - Files: [[Reports/Deep_Dives/D02_AdaBoost|Report]] | [[Code/Deep_Dives/D02_AdaBoost.py|Code]]
    - **Task:** Implement decision stumps as weak learners; update sample weights $w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(x_i))$ and compute learner weights $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$.
25. **D03: Gradient Boosting Machines (GBM)**
    - Files: [[Reports/Deep_Dives/D03_Gradient_Boosting_Machines|Report]] | [[Code/Deep_Dives/D03_Gradient_Boosting_Machines.py|Code]]
    - **Task:** Derive negative gradient residuals for mean squared error and binary cross-entropy; fit sequential regression trees to pseudo-residuals with learning rate shrinkage $\eta$.
26. **D04: Stacking & Blending Ensembles**
    - Files: [[Reports/Deep_Dives/D04_Stacking_and_Blending_Ensembles|Report]] | [[Code/Deep_Dives/D04_Stacking_and_Blending_Ensembles.py|Code]]
    - **Task:** Implement out-of-fold feature generation across multiple heterogeneous base learners (Logistic Regression, Tree, KNN); train a meta-estimator on out-of-fold prediction matrices.
27. **I01: Gradient Descent Optimizer Family**
    - Files: [[Reports/Deep_Dives/I01_Gradient_Descent_Optimizer_Family|Report]] | [[Code/Deep_Dives/I01_Gradient_Descent_Optimizer_Family.py|Code]]
    - **Task:** Derive and hand-code Classical Momentum, Nesterov Accelerated Gradient, AdaGrad, RMSProp, and Adam with bias correction; test all 5 optimizers on the Rosenbrock and Beale loss surfaces.
28. **I02: Newton's Method & Quasi-Newton (BFGS)**
    - Files: [[Reports/Deep_Dives/I02_Newton_and_Quasi_Newton_BFGS|Report]] | [[Code/Deep_Dives/I02_Newton_and_Quasi_Newton_BFGS.py|Code]]
    - **Task:** Compute exact Hessian matrices for Newton-Raphson optimization; implement the rank-2 BFGS inverse Hessian approximation update $H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$ with backtracking line search.
29. **I03: Constrained Optimization & KKT Conditions**
    - Files: [[Reports/Deep_Dives/I03_Constrained_Optimization_KKT|Report]] | [[Code/Deep_Dives/I03_Constrained_Optimization_KKT.py|Code]]
    - **Task:** Formulate the Lagrangian $\mathcal{L}(x, \lambda, \nu)$; verify primal feasibility, dual feasibility, complementary slackness, and stationarity numerically for quadratic programs.
30. **I04: Convex Optimization & Projected Gradient Descent**
    - Files: [[Reports/Deep_Dives/I04_Convex_Optimization_and_Duality|Report]] | [[Code/Deep_Dives/I04_Convex_Optimization_and_Duality.py|Code]]
    - **Task:** Implement Projected Gradient Descent onto convex sets ($\ell_2$ ball, non-negative orthant, probability simplex); demonstrate convergence of primal-dual gap.
31. **I05: Linear Programming — The Simplex Method**
    - Files: [[Reports/Deep_Dives/I05_Linear_Programming_Simplex|Report]] | [[Code/Deep_Dives/I05_Linear_Programming_Simplex.py|Code]]
    - **Task:** Convert standard LPs into tableau form with slack variables; implement pivot selection via Bland's rule to prevent cycling and find exact optimal vertex solutions.

---

## Phase 4: Probabilistic Graphical Models & Bayesian Methods
*Objective: Perform exact and approximate probabilistic inference, structure latent representations, and reason under uncertainty.*

32. **E01: Bayesian Networks & Variable Elimination**
    - Files: [[Reports/Deep_Dives/E01_Bayesian_Networks|Report]] | [[Code/Deep_Dives/E01_Bayesian_Networks.py|Code]]
    - **Task:** Define conditional probability tables (CPTs) on a directed acyclic graph (DAG); implement factor multiplication, marginalization, and the exact Variable Elimination algorithm.
33. **E02: Hidden Markov Models (HMM)**
    - Files: [[Reports/Deep_Dives/E02_Hidden_Markov_Models|Report]] | [[Code/Deep_Dives/E02_Hidden_Markov_Models.py|Code]]
    - **Task:** Implement the Forward algorithm (filtering), Backward algorithm (smoothing), Viterbi algorithm (max-path decoding in log-space), and Baum-Welch (EM parameter estimation).
34. **E03: Markov Chain Monte Carlo (MCMC)**
    - Files: [[Reports/Deep_Dives/E03_Markov_Chain_Monte_Carlo|Report]] | [[Code/Deep_Dives/E03_Markov_Chain_Monte_Carlo.py|Code]]
    - **Task:** Code the Metropolis-Hastings acceptance rule $\alpha = \min(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)})$ and Gibbs sampling for bivariate/multivariate distributions; compute chain autocorrelation and burn-in diagnostic.
35. **E04: Variational Inference (Mean-Field CAVI)**
    - Files: [[Reports/Deep_Dives/E04_Variational_Inference|Report]] | [[Code/Deep_Dives/E04_Variational_Inference.py|Code]]
    - **Task:** Derive the Evidence Lower Bound (ELBO); derive and implement Coordinate Ascent Variational Inference (CAVI) for a Gaussian mixture model with conjugate priors.
36. **E05: Gaussian Processes (GP)**
    - Files: [[Reports/Deep_Dives/E05_Gaussian_Processes|Report]] | [[Code/Deep_Dives/E05_Gaussian_Processes.py|Code]]
    - **Task:** Implement the Squared Exponential (RBF) kernel; compute posterior mean $K_*(K + \sigma_n^2 I)^{-1}y$ and posterior covariance matrix $K_{**} - K_*(K + \sigma_n^2 I)^{-1}K_*^T$; sample prior and posterior functions with 95% confidence intervals.
37. **E06: Kalman & Particle Filters**
    - Files: [[Reports/Deep_Dives/E06_Kalman_Filters_and_Particle_Filters|Report]] | [[Code/Deep_Dives/E06_Kalman_Filters_and_Particle_Filters.py|Code]]
    - **Task:** Implement linear Gaussian Kalman Filter predict/update equations; implement a non-linear Sequential Importance Resampling (SIR) Particle Filter with systematic resampling.
38. **E07: Conditional Random Fields (Linear-Chain CRF)**
    - Files: [[Reports/Deep_Dives/E07_Conditional_Random_Fields|Report]] | [[Code/Deep_Dives/E07_Conditional_Random_Fields.py|Code]]
    - **Task:** Implement state and transition feature functions for sequence labeling; compute exact gradients using the forward-backward algorithm on the partition function $Z(x)$.
39. **E08: Latent Dirichlet Allocation (LDA)**
    - Files: [[Reports/Deep_Dives/E08_Latent_Dirichlet_Allocation|Report]] | [[Code/Deep_Dives/E08_Latent_Dirichlet_Allocation.py|Code]]
    - **Task:** Implement Collapsed Gibbs Sampling over document-word tokens; update topic assignments $z_{di} \propto \frac{n_{d,k}^{(-i)} + \alpha}{\sum_k (n_{d,k}^{(-i)} + \alpha)} \cdot \frac{n_{k,v}^{(-i)} + \beta}{\sum_v (n_{k,v}^{(-i)} + \beta)}$ and extract topic-word distributions.
40. **E09: Bayesian Optimization**
    - Files: [[Reports/Deep_Dives/E09_Bayesian_Optimization|Report]] | [[Code/Deep_Dives/E09_Bayesian_Optimization.py|Code]]
    - **Task:** Combine E05's Gaussian Process surrogate with Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions; optimize a multi-modal black-box function.

---

## Phase 5: Classical Time Series & Volatility
*Objective: Model temporal dependencies, seasonality, non-stationarity, and time-varying volatility.*

41. **F01: Decomposition & Smoothing**
    - Files: [[Reports/Deep_Dives/F01_Decomposition_and_Smoothing|Report]] | [[Code/Deep_Dives/F01_Decomposition_and_Smoothing.py|Code]]
    - **Task:** Implement classical additive/multiplicative moving-average decomposition; code Simple Exponential Smoothing, Holt's linear trend, and Holt-Winters additive/multiplicative seasonal smoothing.
42. **F02: ARIMA & SARIMA**
    - Files: [[Reports/Deep_Dives/F02_ARIMA_and_SARIMA|Report]] | [[Code/Deep_Dives/F02_ARIMA_and_SARIMA.py|Code]]
    - **Task:** Implement lag differencing $(1-B)^d$; estimate AR parameters via the Yule-Walker equations using sample autocovariance; estimate MA parameters via Hannan-Rissanen residuals.
43. **F03: Survival Analysis**
    - Files: [[Reports/Deep_Dives/F03_Survival_Analysis|Report]] | [[Code/Deep_Dives/F03_Survival_Analysis.py|Code]]
    - **Task:** Derive and code the non-parametric Kaplan-Meier product-limit estimator $\hat{S}(t) = \prod_{t_i \le t} (1 - \frac{d_i}{n_i})$ on right-censored data; implement Newton-Raphson optimization for Cox Proportional Hazards partial likelihood.
44. **F04: Volatility Modeling (ARCH & GARCH)**
    - Files: [[Reports/Deep_Dives/F04_Volatility_Modeling_ARCH_GARCH|Report]] | [[Code/Deep_Dives/F04_Volatility_Modeling_ARCH_GARCH.py|Code]]
    - **Task:** Formulate ARCH(1) and GARCH(1,1) conditional variance equations $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$; estimate parameters via maximum likelihood using hand-rolled numerical gradient ascent.

---

## Phase 6: Reinforcement Learning Mastery
*Objective: Build agents spanning tabular bandits, dynamic programming, temporal-difference control, deep Q-networks, policy gradients, and model-based tree search.*

45. **G01: Multi-Armed Bandits**
    - Files: [[Reports/Deep_Dives/G01_Multi_Armed_Bandits|Report]] | [[Code/Deep_Dives/G01_Multi_Armed_Bandits.py|Code]]
    - **Task:** Implement $\epsilon$-greedy (with decay), UCB1 ($Q(a) + c\sqrt{\frac{\ln t}{N(a)}}$), and Bayesian Thompson Sampling with Beta priors; simulate cumulative regret curves over 10-armed bandit testbeds.
46. **G02: MDPs & Dynamic Programming**
    - Files: [[Reports/Deep_Dives/G02_MDPs_and_Dynamic_Programming|Report]] | [[Code/Deep_Dives/G02_MDPs_and_Dynamic_Programming.py|Code]]
    - **Task:** Construct a discrete GridWorld environment; implement Bellman optimality backup for Value Iteration; implement iterative Policy Evaluation and Policy Improvement for Policy Iteration.
47. **G03: Monte Carlo Prediction & Control**
    - Files: [[Reports/Deep_Dives/G03_Monte_Carlo_Prediction_and_Control|Report]] | [[Code/Deep_Dives/G03_Monte_Carlo_Prediction_and_Control.py|Code]]
    - **Task:** Generate full episodes; implement First-Visit and Every-Visit Monte Carlo evaluation; implement on-policy $\epsilon$-soft Monte Carlo control and off-policy control with weighted importance sampling.
48. **G04: Temporal-Difference Learning (SARSA & Q-Learning)**
    - Files: [[Reports/Deep_Dives/G04_Temporal_Difference_Learning|Report]] | [[Code/Deep_Dives/G04_Temporal_Difference_Learning.py|Code]]
    - **Task:** Implement on-policy SARSA $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$, off-policy Q-Learning $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$, and Expected SARSA; compare trajectories on Cliff Walking.
49. **G05: Deep Q-Networks (DQN, Double DQN, Dueling DQN)**
    - Files: [[Reports/Deep_Dives/G05_Deep_Q_Networks|Report]] | [[Code/Deep_Dives/G05_Deep_Q_Networks.py|Code]]
    - **Task:** Build a 2-layer neural network with hand-derived forward and backward matrix passes in pure NumPy; implement an experience replay buffer and target network synchronization; implement Double DQN action selection and Dueling architecture ($V(s) + A(s,a) - \bar{A}(s)$).
50. **G06: Policy Gradient (REINFORCE)**
    - Files: [[Reports/Deep_Dives/G06_Policy_Gradient_REINFORCE|Report]] | [[Code/Deep_Dives/G06_Policy_Gradient_REINFORCE.py|Code]]
    - **Task:** Parameterize a Softmax policy in NumPy; compute analytic log-derivative gradients $\nabla_\theta \ln \pi(a|s)$; update weights using discounted return $G_t$ with baseline subtraction to reduce variance.
51. **G07: Actor-Critic (Advantage Actor-Critic / A2C)**
    - Files: [[Reports/Deep_Dives/G07_Actor_Critic|Report]] | [[Code/Deep_Dives/G07_Actor_Critic.py|Code]]
    - **Task:** Code a simultaneous Actor policy $\pi_\theta(a|s)$ and Critic value estimator $V_\phi(s)$ in pure NumPy; compute TD advantage $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ and update both networks concurrently.
52. **G08: PPO — Clipped Surrogate Objective**
    - Files: [[Reports/Deep_Dives/G08_PPO_Clipped_Objective|Report]] | [[Code/Deep_Dives/G08_PPO_Clipped_Objective.py|Code]]
    - **Task:** Collect batch rollouts; compute probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$; evaluate $\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)$; perform multi-epoch mini-batch updates in raw NumPy.
53. **G09: Continuous Control (DDPG)**
    - Files: [[Reports/Deep_Dives/G09_Continuous_Control_DDPG|Report]] | [[Code/Deep_Dives/G09_Continuous_Control_DDPG.py|Code]]
    - **Task:** Implement deterministic actor $\mu_\theta(s)$ and continuous action-value critic $Q_\phi(s,a)$ with hand-derived chain rule updates $\nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s,a)|_{a=\mu(s)} \nabla_\theta \mu(s)]$; add Ornstein-Uhlenbeck exploration noise.
54. **G10: Model-Based RL & Monte Carlo Tree Search (MCTS)**
    - Files: [[Reports/Deep_Dives/G10_Model_Based_RL_and_MCTS|Report]] | [[Code/Deep_Dives/G10_Model_Based_RL_and_MCTS.py|Code]]
    - **Task:** Implement the 4 MCTS phases (Selection via UCT $Q + c\sqrt{\frac{\ln N}{n}}$, Expansion, Simulation/Rollout, Backpropagation) on a hand-built board game environment.
55. **G11: Multi-Agent RL Basics**
    - Files: [[Reports/Deep_Dives/G11_Multi_Agent_RL_Basics|Report]] | [[Code/Deep_Dives/G11_Multi_Agent_RL_Basics.py|Code]]
    - **Task:** Implement Independent Q-Learners (IQL) playing a competitive/cooperative grid environment; observe non-stationarity dynamics and convergence to joint strategies.
56. **G12: Imitation Learning & Inverse RL**
    - Files: [[Reports/Deep_Dives/G12_Imitation_Learning_and_Inverse_RL|Report]] | [[Code/Deep_Dives/G12_Imitation_Learning_and_Inverse_RL.py|Code]]
    - **Task:** Implement Behavioral Cloning (supervised policy regression); implement Maximum Entropy Inverse Reinforcement Learning to recover underlying reward function vectors from expert state frequencies.
57. **G13: Offline & Hierarchical RL**
    - Files: [[Reports/Deep_Dives/G13_Offline_and_Hierarchical_RL|Report]] | [[Code/Deep_Dives/G13_Offline_and_Hierarchical_RL.py|Code]]
    - **Task:** Implement Batch/Offline Q-Learning with conservative value penalization on out-of-distribution actions; implement the Sutton-Precup-Singh Options framework with initiation sets, intra-option policies, and termination conditions.
58. **G14: Partially Observable MDPs (POMDP)**
    - Files: [[Reports/Deep_Dives/G14_Partially_Observable_MDPs|Report]] | [[Code/Deep_Dives/G14_Partially_Observable_MDPs.py|Code]]
    - **Task:** Implement discrete belief state updates $b'(s') \propto O(o|s',a) \sum_s T(s'|s,a) b(s)$; compute belief-state value functions over simplex grids.

---

## Phase 7: Evolutionary & Nature-Inspired Optimization
*Objective: Optimize non-differentiable, deceptive, multi-modal, and multi-objective fitness landscapes.*

59. **H01: Genetic Algorithms (GA)**
    - Files: [[Reports/Deep_Dives/H01_Genetic_Algorithms|Report]] | [[Code/Deep_Dives/H01_Genetic_Algorithms.py|Code]]
    - **Task:** Implement tournament selection, Order-1 crossover (OX1) for permutations, and inversion mutation; solve the Traveling Salesman Problem (TSP) from scratch.
60. **H02: Genetic Programming (GP)**
    - Files: [[Reports/Deep_Dives/H02_Genetic_Programming|Report]] | [[Code/Deep_Dives/H02_Genetic_Programming.py|Code]]
    - **Task:** Represent programs as mathematical AST expression trees; implement subtree crossover and point mutation to perform symbolic regression finding an unknown generating equation.
61. **H03: Evolution Strategies (CMA-ES)**
    - Files: [[Reports/Deep_Dives/H03_Evolution_Strategies|Report]] | [[Code/Deep_Dives/H03_Evolution_Strategies.py|Code]]
    - **Task:** Implement a simplified Covariance Matrix Adaptation Evolution Strategy; update mean vector $\mu$, step size $\sigma$, and covariance matrix $C$ using rank-$\mu$ recombination on non-convex test surfaces.
62. **H04: Particle Swarm Optimization (PSO)**
    - Files: [[Reports/Deep_Dives/H04_Particle_Swarm_Optimization|Report]] | [[Code/Deep_Dives/H04_Particle_Swarm_Optimization.py|Code]]
    - **Task:** Implement particle velocity updates $v_i \leftarrow w v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i)$ with constriction coefficients; find global minima of the Rastrigin and Griewank functions.
63. **H05: Ant Colony Optimization (ACO)**
    - Files: [[Reports/Deep_Dives/H05_Ant_Colony_Optimization|Report]] | [[Code/Deep_Dives/H05_Ant_Colony_Optimization.py|Code]]
    - **Task:** Simulate artificial ants choosing edges via $P_{ij} \propto \tau_{ij}^\alpha \eta_{ij}^\beta$; implement pheromone evaporation $\tau \leftarrow (1-\rho)\tau$ and deposit $\Delta \tau = \frac{Q}{L_k}$ to solve TSP.
64. **H06: Simulated Annealing**
    - Files: [[Reports/Deep_Dives/H06_Simulated_Annealing|Report]] | [[Code/Deep_Dives/H06_Simulated_Annealing.py|Code]]
    - **Task:** Implement Metropolis transition probability $P = \exp(-\frac{\Delta E}{T})$ with geometric and logarithmic cooling schedules; avoid local minima in rugged combinatorial landscapes.
65. **H07: Differential Evolution (DE)**
    - Files: [[Reports/Deep_Dives/H07_Differential_Evolution|Report]] | [[Code/Deep_Dives/H07_Differential_Evolution.py|Code]]
    - **Task:** Implement mutation scheme $\text{DE/rand/1/bin}$: $v_i = x_{r1} + F(x_{r2} - x_{r3})$; apply binomial crossover and selection against target vectors.
66. **H08: NEAT (NeuroEvolution of Augmenting Topologies)**
    - Files: [[Reports/Deep_Dives/H08_NEAT_NeuroEvolution|Report]] | [[Code/Deep_Dives/H08_NEAT_NeuroEvolution.py|Code]]
    - **Task:** Implement historical innovation tracking numbers; code topological structural mutation (add node, add connection) and speciation distance metrics.
67. **H09: Multi-Objective Optimization (NSGA-II)**
    - Files: [[Reports/Deep_Dives/H09_Multi_Objective_NSGA_II|Report]] | [[Code/Deep_Dives/H09_Multi_Objective_NSGA_II.py|Code]]
    - **Task:** Implement fast non-dominated sorting into Pareto fronts ($F_1, F_2, \dots$); compute crowding distance; perform elitist selection on bi-objective benchmark problems.

---

## Phase 8: Alternative Learning Paradigms
*Objective: Implement causal inference, recommender systems, self-supervised learning, lifelong learning, and explainability.*

68. **J01: Semi-Supervised Learning**
    - Files: [[Reports/Deep_Dives/J01_Semi_Supervised_Learning|Report]] | [[Code/Deep_Dives/J01_Semi_Supervised_Learning.py|Code]]
    - **Task:** Implement pseudo-label self-training with confidence thresholds; implement harmonic function Label Propagation using Gaussian graph affinities $W_{ij} = \exp(-\gamma \|x_i - x_j\|^2)$.
69. **J02: Self-Supervised Contrastive Learning (InfoNCE / SimCLR)**
    - Files: [[Reports/Deep_Dives/J02_Self_Supervised_Contrastive_Learning|Report]] | [[Code/Deep_Dives/J02_Self_Supervised_Contrastive_Learning.py|Code]]
    - **Task:** Create stochastic data transformations in NumPy; compute pairwise cosine similarity matrices; derive and code the InfoNCE loss $\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}$ and its exact backprop gradients.
70. **J03: Meta-Learning (MAML)**
    - Files: [[Reports/Deep_Dives/J03_Meta_Learning_MAML|Report]] | [[Code/Deep_Dives/J03_Meta_Learning_MAML.py|Code]]
    - **Task:** Implement Model-Agnostic Meta-Learning for few-shot regression; compute inner-loop task adaptation $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}$ and outer-loop meta-gradient updates via finite-difference or hand-derived second derivatives in NumPy.
71. **J04: Federated Learning (FedAvg)**
    - Files: [[Reports/Deep_Dives/J04_Federated_Learning_FedAvg|Report]] | [[Code/Deep_Dives/J04_Federated_Learning_FedAvg.py|Code]]
    - **Task:** Simulate non-IID data partitions across decentralized nodes; perform local gradient descent steps and aggregate client models using weighted coordinate averaging $w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k$.
72. **J05: Online Learning & Concept Drift**
    - Files: [[Reports/Deep_Dives/J05_Online_Learning_and_Concept_Drift|Report]] | [[Code/Deep_Dives/J05_Online_Learning_and_Concept_Drift.py|Code]]
    - **Task:** Implement streaming Perceptron updates on sequential data; implement the ADWIN (Adaptive Windowing) algorithm using statistical sub-window comparison tests to trigger drift alerts.
73. **J06: Continual Learning & Elastic Weight Consolidation (EWC)**
    - Files: [[Reports/Deep_Dives/J06_Continual_Lifelong_Learning_EWC|Report]] | [[Code/Deep_Dives/J06_Continual_Lifelong_Learning_EWC.py|Code]]
    - **Task:** Train a model on Task A; compute the empirical Fisher Information Matrix diagonal $F_{ii}$; regularize Task B training with $\frac{\lambda}{2}\sum_i F_{ii}(\theta_i - \theta_{A,i}^*)^2$ to prevent catastrophic forgetting.
74. **J07: Causal Inference**
    - Files: [[Reports/Deep_Dives/J07_Causal_Inference|Report]] | [[Code/Deep_Dives/J07_Causal_Inference.py|Code]]
    - **Task:** Formulate a Structural Causal Model (SCM); estimate Average Treatment Effect (ATE) via Propensity Score Matching and verify parallel trends for Difference-in-Differences (DiD).
75. **J08: Anomaly Detection**
    - Files: [[Reports/Deep_Dives/J08_Anomaly_Detection|Report]] | [[Code/Deep_Dives/J08_Anomaly_Detection.py|Code]]
    - **Task:** Implement recursive Isolation Trees choosing random split dimensions and thresholds; compute path length anomaly scores $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$; compare against parametric Mahalanobis distance outliers.
76. **J09: Recommender Systems**
    - Files: [[Reports/Deep_Dives/J09_Recommender_Systems|Report]] | [[Code/Deep_Dives/J09_Recommender_Systems.py|Code]]
    - **Task:** Implement user-based and item-based collaborative filtering with adjusted cosine similarity; implement matrix factorization via SGD minimizing $\sum_{(u,i)} (r_{ui} - p_u^T q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2)$.
77. **J10: Explainable AI (SHAP & LIME)**
    - Files: [[Reports/Deep_Dives/J10_Explainable_AI|Report]] | [[Code/Deep_Dives/J10_Explainable_AI.py|Code]]
    - **Task:** Compute exact Shapley values via exhaustive $2^{|F|}$ coalition permutation marginal contributions; implement a local surrogate linear model with exponential distance kernel weights (mini-LIME).
78. **J11: Active Learning**
    - Files: [[Reports/Deep_Dives/J11_Active_Learning|Report]] | [[Code/Deep_Dives/J11_Active_Learning.py|Code]]
    - **Task:** Implement Uncertainty Sampling (least confidence, margin sampling, entropy) and Query-by-Committee (vote entropy) on an unlabeled pool; simulate sample efficiency curves.
79. **J12: Multi-Task & Transfer Learning**
    - Files: [[Reports/Deep_Dives/J12_Multi_Task_and_Transfer_Learning|Report]] | [[Code/Deep_Dives/J12_Multi_Task_and_Transfer_Learning.py|Code]]
    - **Task:** Build a shared-trunk neural architecture with separate task-specific linear heads; derive simultaneous backpropagation through shared representations in NumPy; implement linear probing transfer.

---

## Phase 9: Neural Network Architectures Missing from Standard DL
*Objective: Build exotic, non-standard, physical, and dynamical neural paradigms from raw matrix calculus.*

80. **K01: Hopfield Networks & Modern Hopfield Memory**
    - Files: [[Reports/Deep_Dives/K01_Hopfield_Networks|Report]] | [[Code/Deep_Dives/K01_Hopfield_Networks.py|Code]]
    - **Task:** Implement classic binary Hopfield energy $E = -\frac{1}{2} s^T W s$ and Hebbian storage; implement Modern Continuous Hopfield energy and prove mathematically that its one-step update is softmax self-attention.
81. **K02: Normalizing Flows (RealNVP)**
    - Files: [[Reports/Deep_Dives/K02_Normalizing_Flows|Report]] | [[Code/Deep_Dives/K02_Normalizing_Flows.py|Code]]
    - **Task:** Implement forward and inverse affine coupling layers $y_{1:d} = x_{1:d}, y_{d+1:D} = x_{d+1:D} \odot e^{s(x_{1:d})} + t(x_{1:d})$; compute exact log-determinant Jacobian $\sum s(x_{1:d})$; train generative density via change-of-variables maximum likelihood in NumPy.
82. **K03: Neural ODEs**
    - Files: [[Reports/Deep_Dives/K03_Neural_ODEs|Report]] | [[Code/Deep_Dives/K03_Neural_ODEs.py|Code]]
    - **Task:** Implement explicit Euler and Runge-Kutta 4th-order (RK4) numerical ODE integrators; parameterize dynamics $\frac{dz}{dt} = f_\theta(z,t)$; perform forward integration and backpropagate gradients using hand-written discrete adjoint updates.
83. **K04: Capsule Networks**
    - Files: [[Reports/Deep_Dives/K04_Capsule_Networks|Report]] | [[Code/Deep_Dives/K04_Capsule_Networks.py|Code]]
    - **Task:** Implement vector-valued capsule activations with squashing non-linearity $v_j = \frac{\|s_j\|^2}{1+\|s_j\|^2}\frac{s_j}{\|s_j\|}$; implement iterative dynamic routing-by-agreement updating coupling coefficients $c_{ij} = \text{softmax}(b_{ij})$.
84. **K05: Physics-Informed Neural Networks (PINNs)**
    - Files: [[Reports/Deep_Dives/K05_Physics_Informed_Neural_Networks|Report]] | [[Code/Deep_Dives/K05_Physics_Informed_Neural_Networks.py|Code]]
    - **Task:** Formulate physical differential equation residuals (e.g., damped harmonic oscillator $\ddot{x} + 2\zeta\omega_0 \dot{x} + \omega_0^2 x = 0$); compute spatial/temporal derivatives via hand-coded finite differences or exact symbolic matrix gradients; train neural parameters to satisfy both boundary conditions and physical law.
85. **K06: Classical Word Embeddings (Word2Vec Skip-gram)**
    - Files: [[Reports/Deep_Dives/K06_Classical_Word_Embeddings_Word2Vec|Report]] | [[Code/Deep_Dives/K06_Classical_Word_Embeddings_Word2Vec.py|Code]]
    - **Task:** Construct unigram table with $3/4$ power scaling; implement Skip-gram with Negative Sampling (SGNS); derive and code gradients for target and context embedding matrices in NumPy.
86. **K07: Reservoir Computing (Echo State Networks)**
    - Files: [[Reports/Deep_Dives/K07_Reservoir_Computing_Echo_State_Networks|Report]] | [[Code/Deep_Dives/K07_Reservoir_Computing_Echo_State_Networks.py|Code]]
    - **Task:** Initialize a large sparse recurrent reservoir matrix $W_{res}$ and scale its spectral radius $\rho(W_{res}) < 1$ to satisfy the Echo State Property; keep recurrent weights completely frozen; train readout weights $W_{out}$ via Ridge closed-form regression.

---

## Phase 10: Classical AI Search, Constraints & Game Theory
*Objective: Master tree search, heuristic traversal, constraint satisfaction, formal logic, graph centrality, and game-theoretic equilibria.*

87. **L01: Uninformed Search**
    - Files: [[Reports/Deep_Dives/L01_Uninformed_Search|Report]] | [[Code/Deep_Dives/L01_Uninformed_Search.py|Code]]
    - **Task:** Implement Breadth-First Search (queue), Depth-First Search (stack/recursion), and Iterative Deepening DFS (IDDFS) on arbitrary directed state graphs; track visited states to prevent cycles.
88. **L02: Informed Search (A*)**
    - Files: [[Reports/Deep_Dives/L02_Informed_Search_A_Star|Report]] | [[Code/Deep_Dives/L02_Informed_Search_A_Star.py|Code]]
    - **Task:** Implement A* with priority queue evaluation $f(n) = g(n) + h(n)$; test Euclidean and Manhattan heuristics on grid pathfinding; prove that consistent heuristics guarantee optimal paths without reopening closed nodes.
89. **L03: Adversarial Search (Minimax & Alpha-Beta Pruning)**
    - Files: [[Reports/Deep_Dives/L03_Adversarial_Search_Minimax|Report]] | [[Code/Deep_Dives/L03_Adversarial_Search_Minimax.py|Code]]
    - **Task:** Build a game engine for Tic-Tac-Toe / Connect-4; implement recursive Minimax decision rule; implement $\alpha$-$\beta$ branch pruning and measure nodes explored reduction.
90. **L04: Constraint Satisfaction Problems (CSP)**
    - Files: [[Reports/Deep_Dives/L04_Constraint_Satisfaction_Problems|Report]] | [[Code/Deep_Dives/L04_Constraint_Satisfaction_Problems.py|Code]]
    - **Task:** Implement backtracking search with Minimum Remaining Values (MRV) and Degree heuristic variable ordering; implement the AC-3 arc-consistency algorithm to solve N-Queens and Sudoku.
91. **L05: Fuzzy Logic Systems**
    - Files: [[Reports/Deep_Dives/L05_Fuzzy_Logic_Systems|Report]] | [[Code/Deep_Dives/L05_Fuzzy_Logic_Systems.py|Code]]
    - **Task:** Define triangular and trapezoidal fuzzy membership functions $\mu_A(x)$; implement Mamdani fuzzy inference with min-conjunction and max-aggregation; defuzzify via Center of Gravity (Centroid).
92. **L06: Rule-Based Expert Systems**
    - Files: [[Reports/Deep_Dives/L06_Rule_Based_Expert_Systems|Report]] | [[Code/Deep_Dives/L06_Rule_Based_Expert_Systems.py|Code]]
    - **Task:** Represent production rules $(IF \dots THEN \dots)$; implement Forward-Chaining (data-driven) using conflict resolution strategies; implement Backward-Chaining (goal-driven) with recursive hypothesis verification.
93. **L07: Classical Graph Algorithms for ML**
    - Files: [[Reports/Deep_Dives/L07_Classical_Graph_Algorithms_for_ML|Report]] | [[Code/Deep_Dives/L07_Classical_Graph_Algorithms_for_ML.py|Code]]
    - **Task:** Implement Dijkstra's shortest path; implement Google's PageRank via Power Iteration on Google matrix $G = d M + \frac{1-d}{N}\mathbf{1}\mathbf{1}^T$; implement Normalized Spectral Clustering via graph Laplacian $L = D - W$ eigendecomposition without importing `networkx`.
94. **L08: Game Theory & Multi-Agent Equilibria**
    - Files: [[Reports/Deep_Dives/L08_Game_Theory_and_Multi_Agent_Equilibria|Report]] | [[Code/Deep_Dives/L08_Game_Theory_and_Multi_Agent_Equilibria.py|Code]]
    - **Task:** Find pure Nash equilibria via iterated elimination of strictly dominated strategies; compute mixed Nash equilibria via Fictitious Play and verify von Neumann's Minimax Theorem for zero-sum matrix games.

---

## Phase 11: High-Dimensional Vector Search & Streaming Systems
*Objective: Build sub-linear geometric graphs, probabilistic hashing, and streaming sketching algorithms powering modern Vector DBs and RAG.*

95. **M01: Hierarchical Navigable Small World (HNSW)**
    - Files: [[Reports/Deep_Dives/M01_Hierarchical_Navigable_Small_World_HNSW|Report]] | [[Code/Deep_Dives/M01_Hierarchical_Navigable_Small_World_HNSW.py|Code]]
    - **Task:** Construct multi-layer geometric skip-graphs with exponential height distribution $m_L = 1/\ln(M)$; implement greedy beam search across express and local layers with diversity heuristic pruning.
96. **M02: Locality-Sensitive Hashing (LSH: SimHash & MinHash)**
    - Files: [[Reports/Deep_Dives/M02_Locality_Sensitive_Hashing|Report]] | [[Code/Deep_Dives/M02_Locality_Sensitive_Hashing.py|Code]]
    - **Task:** Implement Random Hyperplane Projections for angular similarity and MinHash permutation hashing for Jaccard sets; prove and verify the S-curve candidate amplification formula $1 - (1 - s^r)^b$.
97. **M03: Space Partitioning Trees (k-d Tree & Ball Tree)**
    - Files: [[Reports/Deep_Dives/M03_Space_Partitioning_Trees|Report]] | [[Code/Deep_Dives/M03_Space_Partitioning_Trees.py|Code]]
    - **Task:** Build recursive orthogonal split $k$-d trees and metric Ball trees; implement branch-and-bound pruning utilizing hyperplane bounds and reverse triangle inequality hypersphere tests.
98. **M04: Probabilistic Streaming Sketches**
    - Files: [[Reports/Deep_Dives/M04_Probabilistic_Streaming_Sketches|Report]] | [[Code/Deep_Dives/M04_Probabilistic_Streaming_Sketches.py|Code]]
    - **Task:** Implement Count-Min Sketch frequency estimation with universal hashing; implement HyperLogLog distinct cardinality tracking via register leading zeros; implement Reservoir Sampling (Algorithm R).

---

## Phase 12: Computational Graph & Automatic Differentiation Engines
*Objective: Demystify deep learning compilers and autodiff engines by constructing tape-based reverse mode, dual forward mode, and symbolic ASTs.*

99. **N01: Tape-Based Reverse-Mode Autodiff**
    - Files: [[Reports/Deep_Dives/N01_Tape_Based_Reverse_Mode_Autodiff|Report]] | [[Code/Deep_Dives/N01_Tape_Based_Reverse_Mode_Autodiff.py|Code]]
    - **Task:** Build a computational graph `Tensor` class with operator overloading (`+`, `*`, `@`, activations); implement topological sort via DFS and backward vector-Jacobian product (VJP) accumulation; train a non-linear MLP from scratch.
100. **N02: Forward-Mode Autodiff & Dual Numbers**
    - Files: [[Reports/Deep_Dives/N02_Forward_Mode_Autodiff_and_Dual_Numbers|Report]] | [[Code/Deep_Dives/N02_Forward_Mode_Autodiff_and_Dual_Numbers.py|Code]]
    - **Task:** Implement dual number algebra $z = a + b\epsilon$ where $\epsilon^2 = 0$; evaluate exact Jacobian-Vector Products (JVPs) and directional derivatives in a single forward pass without graph retention.
101. **N03: Symbolic Expression Trees & Simplification**
    - Files: [[Reports/Deep_Dives/N03_Symbolic_Expression_Trees_and_Simplification|Report]] | [[Code/Deep_Dives/N03_Symbolic_Expression_Trees_and_Simplification.py|Code]]
    - **Task:** Construct an AST symbolic calculus engine; implement recursive product/quotient/chain rules; implement algebraic graph simplification (constant folding, additive identities, annihilation).

---

## Phase 13: Optimal Transport & Geometric Machine Learning
*Objective: Master Wasserstein geometry, entropic matrix scaling, and Riemannian statistical manifolds.*

102. **O01: Earth Mover's Distance & Wasserstein Metrics**
    - Files: [[Reports/Deep_Dives/O01_Earth_Movers_Distance_Wasserstein_Metrics|Report]] | [[Code/Deep_Dives/O01_Earth_Movers_Distance_Wasserstein_Metrics.py|Code]]
    - **Task:** Derive the Kantorovich relaxation and Kantorovich-Rubinstein duality theorem; implement exact 1D sorted quantile Wasserstein distance and discrete multi-dimensional linear programming transport matching.
103. **O02: Entropic Regularized Optimal Transport (Sinkhorn)**
    - Files: [[Reports/Deep_Dives/O02_Entropic_Regularized_Optimal_Transport_Sinkhorn|Report]] | [[Code/Deep_Dives/O02_Entropic_Regularized_Optimal_Transport_Sinkhorn.py|Code]]
    - **Task:** Derive the Gibbs kernel formulation from KKT conditions; implement Sinkhorn-Knopp alternating matrix scaling in $O(N^2)$ time; implement log-domain Log-Sum-Exp stabilization for low temperatures $\varepsilon \to 0$.
104. **O03: Information Geometry & Natural Gradients**
    - Files: [[Reports/Deep_Dives/O03_Information_Geometry_and_Natural_Gradients|Report]] | [[Code/Deep_Dives/O03_Information_Geometry_and_Natural_Gradients.py|Code]]
    - **Task:** Derive the Fisher Information Matrix (FIM) as the Riemannian metric tensor invariant under reparameterization; implement Natural Gradient Descent $\tilde{\nabla} \mathcal{L} = F^{-1}\nabla \mathcal{L}$; prove accelerated convergence over Euclidean GD.

---

## Phase 14: Physics-Based Sampling & Continuous Diffusion
*Objective: Bridge classical statistical mechanics to modern score-based generative diffusion models.*

105. **P01: Hamiltonian Monte Carlo (HMC) & Leapfrog Integrator**
    - Files: [[Reports/Deep_Dives/P01_Hamiltonian_Monte_Carlo_Leapfrog|Report]] | [[Code/Deep_Dives/P01_Hamiltonian_Monte_Carlo_Leapfrog.py|Code]]
    - **Task:** Augment target density with auxiliary momentum in phase space; implement symplectic, volume-preserving Leapfrog numerical integration; apply Metropolis acceptance filtering to sample correlated anisotropic Gaussians.
106. **P02: Langevin Dynamics & MALA**
    - Files: [[Reports/Deep_Dives/P02_Langevin_Dynamics_and_MALA|Report]] | [[Code/Deep_Dives/P02_Langevin_Dynamics_and_MALA.py|Code]]
    - **Task:** Discretize overdamped Langevin SDEs via Euler-Maruyama (ULA); implement Metropolis-Adjusted Langevin Algorithm (MALA) with transition proposal density ratios; sample multimodal distributions.
107. **P03: Energy-Based Models & Denoising Score Matching**
    - Files: [[Reports/Deep_Dives/P03_Energy_Based_Models_and_Score_Matching|Report]] | [[Code/Deep_Dives/P03_Energy_Based_Models_and_Score_Matching.py|Code]]
    - **Task:** Eliminate the intractable partition function $Z_\theta$ by modeling the score $\nabla_x \ln p(x)$; train score models via Vincent's Denoising Score Matching (DSM) objective; generate synthetic samples from noise via Langevin MCMC.

---

## Phase 15: Multi-Agent Coordination & Algorithmic Mechanism Design
*Objective: Solve cooperative credit assignment, continuous multi-agent games, and incentive-compatible market design.*

108. **Q01: Value-Decomposition Networks (VDN) & QMIX**
    - Files: [[Reports/Deep_Dives/Q01_Value_Decomposition_Networks_and_QMIX|Report]] | [[Code/Deep_Dives/Q01_Value_Decomposition_Networks_and_QMIX.py|Code]]
    - **Task:** Formulate the Dec-POMDP cooperative MARL problem; implement VDN additive decomposition and QMIX non-negative hypernetwork monotonic mixing ($\frac{\partial Q_{\text{tot}}}{\partial Q_a} \ge 0$); verify Individual-Global-Max consistency.
109. **Q02: Multi-Agent Deep Deterministic Policy Gradients (MADDPG)**
    - Files: [[Reports/Deep_Dives/Q02_Multi_Agent_Deep_Deterministic_Policy_Gradients_MADDPG|Report]] | [[Code/Deep_Dives/Q02_Multi_Agent_Deep_Deterministic_Policy_Gradients_MADDPG.py|Code]]
    - **Task:** Implement Centralized Training with Decentralized Execution (CTDE); build decentralized continuous actors $\mu_i(o_i)$ and centralized critics $Q_i(\mathbf{x}, \mathbf{a})$; backpropagate multi-agent deterministic policy gradients.
110. **Q03: Algorithmic Mechanism Design & VCG Auctions**
    - Files: [[Reports/Deep_Dives/Q03_Algorithmic_Mechanism_Design_and_VCG_Auctions|Report]] | [[Code/Deep_Dives/Q03_Algorithmic_Mechanism_Design_and_VCG_Auctions.py|Code]]
    - **Task:** Implement Vickrey second-price and combinatorial Vickrey-Clarke-Groves (VCG) mechanisms with Clarke pivot externality payments; prove Dominant-Strategy Incentive Compatibility (DSIC / Truthfulness) via counterfactual bid perturbation.

---

## Phase 16: Quantum Computing Fundamentals & Algorithms
*Objective: Master complex Hilbert space state vectors, quantum gates, variational quantum algorithms, quantum search, and fault-tolerant error correction.*

111. **R01: Qubit State Vectors & Quantum Gates**
    - Files: [[Reports/Deep_Dives/R01_Qubit_State_Vectors_and_Quantum_Gates|Report]] | [[Code/Deep_Dives/R01_Qubit_State_Vectors_and_Quantum_Gates.py|Code]]
    - **Task:** Formulate pure state vectors $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$; map states to the 3D Bloch sphere; verify unitarity ($U^\dagger U = I$) for Pauli, Hadamard, Phase, and rotation gates; construct multi-qubit Bell states via Kronecker tensor products.
112. **R02: Quantum Circuit Simulator**
    - Files: [[Reports/Deep_Dives/R02_Quantum_Circuit_Simulator|Report]] | [[Code/Deep_Dives/R02_Quantum_Circuit_Simulator.py|Code]]
    - **Task:** Build an $n$-qubit state-vector simulator ($2^n$ complex amplitudes); apply arbitrary single/multi-qubit gates via tensor contraction; implement projective measurement and state collapse under the Born rule; verify quantum teleportation protocol.
113. **R03: Variational Quantum Eigensolver (VQE)**
    - Files: [[Reports/Deep_Dives/R03_Variational_Quantum_Eigensolver_VQE|Report]] | [[Code/Deep_Dives/R03_Variational_Quantum_Eigensolver_VQE.py|Code]]
    - **Task:** Decompose molecular electronic Hamiltonians via Jordan-Wigner transformation into Pauli strings; implement parameterized hardware-efficient ansatz; optimize parameters classically to reach chemical accuracy for molecular $H_2$.
114. **R04: Quantum Approximate Optimization Algorithm (QAOA)**
    - Files: [[Reports/Deep_Dives/R04_Quantum_Approximate_Optimization_Algorithm_QAOA|Report]] | [[Code/Deep_Dives/R04_Quantum_Approximate_Optimization_Algorithm_QAOA.py|Code]]
    - **Task:** Formulate MaxCut on graphs as a diagonal Ising cost Hamiltonian; apply alternating layers of cost unitary $U(C, \gamma)$ and transverse mixer $U(B, \beta)$; optimize variational parameters and evaluate approximation ratios.
115. **R05: Grover's Search Algorithm**
    - Files: [[Reports/Deep_Dives/R05_Grovers_Search_Algorithm|Report]] | [[Code/Deep_Dives/R05_Grovers_Search_Algorithm.py|Code]]
    - **Task:** Construct phase-inversion oracle $O_w = I - 2|w\rangle\langle w|$ and diffusion operator $D = 2|s\rangle\langle s| - I$; demonstrate quadratic amplitude amplification over $\mathcal{O}(\sqrt{N})$ iterations; verify $>95\%$ success probabilities.
116. **R06: Shor's Algorithm (Period Finding Component)**
    - Files: [[Reports/Deep_Dives/R06_Shors_Algorithm_Period_Finding|Report]] | [[Code/Deep_Dives/R06_Shors_Algorithm_Period_Finding.py|Code]]
    - **Task:** Implement Quantum Fourier Transform (QFT) and Quantum Phase Estimation; simulate modular exponentiation $a^x \bmod N$; extract order $r$ via continued fractions and factor composite integers ($N=15, 21$).
117. **R07: Tensor Networks & Matrix Product States (MPS)**
    - Files: [[Reports/Deep_Dives/R07_Tensor_Networks_and_Matrix_Product_States|Report]] | [[Code/Deep_Dives/R07_Tensor_Networks_and_Matrix_Product_States.py|Code]]
    - **Task:** Represent 1D quantum many-body wavefunctions as MPS; implement Schmidt decomposition via SVD; truncate bond dimensions $\chi$; compute bipartite von Neumann entanglement entropy $S = -\sum \lambda_i^2 \ln(\lambda_i^2)$.
118. **R08: Quantum Error Correction (3-Qubit Bit-Flip Code)**
    - Files: [[Reports/Deep_Dives/R08_Quantum_Error_Correction_Bit_Flip_Code|Report]] | [[Code/Deep_Dives/R08_Quantum_Error_Correction_Bit_Flip_Code.py|Code]]
    - **Task:** Encode logical qubits into 3-qubit entangled codewords; extract error syndromes via stabilizer measurements ($Z_1 Z_2, Z_2 Z_3$) without collapsing quantum superpositions; apply Pauli-$X$ recovery operations.

---

## Phase 17: Computational Physics & Celestial Mechanics
*Objective: Simulate gravitational dynamics, molecular interactions, phase transitions, and orbital mechanics from first physical principles.*

119. **S01: N-Body Gravitational Simulation**
    - Files: [[Reports/Deep_Dives/S01_N_Body_Gravitational_Simulation|Report]] | [[Code/Deep_Dives/S01_N_Body_Gravitational_Simulation.py|Code]]
    - **Task:** Implement pairwise Newtonian gravitational forces with Plummer softening; integrate trajectories using symplectic Leapfrog / Velocity-Verlet schemes; verify total energy and angular momentum conservation.
120. **S02: Molecular Dynamics (Velocity Verlet)**
    - Files: [[Reports/Deep_Dives/S02_Molecular_Dynamics_Velocity_Verlet|Report]] | [[Code/Deep_Dives/S02_Molecular_Dynamics_Velocity_Verlet.py|Code]]
    - **Task:** Model microscopic atomic interactions via Lennard-Jones 12-6 potentials; apply periodic boundary conditions with minimum image convention; track microcanonical energy conservation ($<0.4\%$ drift).
121. **S03: 2D Ising Model (Metropolis-Hastings)**
    - Files: [[Reports/Deep_Dives/S03_2D_Ising_Model_Metropolis|Report]] | [[Code/Deep_Dives/S03_2D_Ising_Model_Metropolis.py|Code]]
    - **Task:** Simulate magnetic dipole interactions on periodic square lattices; implement single-spin-flip Metropolis acceptance criteria; demonstrate spontaneous magnetization phase transitions at Onsager Curie temperature $T_c \approx 2.269$.
122. **S04: Finite Difference PDE Solving**
    - Files: [[Reports/Deep_Dives/S04_Finite_Difference_PDE_Solving|Report]] | [[Code/Deep_Dives/S04_Finite_Difference_PDE_Solving.py|Code]]
    - **Task:** Solve 1D diffusion and 2D Poisson/Wave equations via central finite differences; verify von Neumann stability conditions ($r = \alpha \Delta t / \Delta x^2 \le 1/2$); compare numerical solutions against analytical Fourier series.
123. **S05: Lomb-Scargle Periodogram**
    - Files: [[Reports/Deep_Dives/S05_Lomb_Scargle_Periodogram|Report]] | [[Code/Deep_Dives/S05_Lomb_Scargle_Periodogram.py|Code]]
    - **Task:** Formulate generalized least-squares spectral analysis for irregularly sampled astrophysical time series; compute spectral power $P_N(\omega)$ invariant to time translation; recover sinusoidal periods in heavy noise.
124. **S06: Kepler Equation & Orbit Propagation**
    - Files: [[Reports/Deep_Dives/S06_Kepler_Equation_Orbit_Propagation|Report]] | [[Code/Deep_Dives/S06_Kepler_Equation_Orbit_Propagation.py|Code]]
    - **Task:** Solve Kepler's transcendental equation $M = E - e\sin E$ via Newton-Raphson; convert orbital elements $(a, e, i, \Omega, \omega, \nu)$ to 3D Cartesian state vectors $(r, v)$; propagate orbits forward in time.

---

## Phase 18: Computational Biology & Bioinformatics
*Objective: Implement sequence alignment algorithms, evolutionary phylogenetic reconstruction, cheminformatics, and reaction-diffusion morphogenesis.*

125. **T01: Sequence Alignment (Needleman-Wunsch & Smith-Waterman)**
    - Files: [[Reports/Deep_Dives/T01_Sequence_Alignment_Needleman_Wunsch_Smith_Waterman|Report]] | [[Code/Deep_Dives/T01_Sequence_Alignment_Needleman_Wunsch_Smith_Waterman.py|Code]]
    - **Task:** Build global alignment (Needleman-Wunsch) and local motif alignment (Smith-Waterman) dynamic programming scoring matrices; implement backtracking to recover optimal alignment strings.
126. **T02: Phylogenetic Trees (UPGMA & Neighbor-Joining)**
    - Files: [[Reports/Deep_Dives/T02_Phylogenetic_Trees_UPGMA_Neighbor_Joining|Report]] | [[Code/Deep_Dives/T02_Phylogenetic_Trees_UPGMA_Neighbor_Joining.py|Code]]
    - **Task:** Reconstruct evolutionary trees from genetic distance matrices; implement ultrametric clustering (UPGMA) and additive neighbor-joining ($Q$-matrix reduction); output trees in Newick format.
127. **T03: Molecular Fingerprints & Tanimoto Similarity**
    - Files: [[Reports/Deep_Dives/T03_Molecular_Fingerprints_and_Similarity|Report]] | [[Code/Deep_Dives/T03_Molecular_Fingerprints_and_Similarity.py|Code]]
    - **Task:** Parse chemical connectivity graphs; generate Morgan / Extended-Connectivity Fingerprints (ECFP) via circular neighborhood hashing; evaluate Tanimoto similarity coefficients.
128. **T04: Molecular Mechanics Force Fields**
    - Files: [[Reports/Deep_Dives/T04_Molecular_Mechanics_Force_Fields|Report]] | [[Code/Deep_Dives/T04_Molecular_Mechanics_Force_Fields.py|Code]]
    - **Task:** Formulate empirical potential energy functions: harmonic bond stretching, harmonic angle bending, and periodic torsional dihedrals; minimize molecular geometry using gradient descent.
129. **T05: Population Genetics (Wright-Fisher & Kingman Coalescent)**
    - Files: [[Reports/Deep_Dives/T05_Population_Genetics_Wright_Fisher_Coalescent|Report]] | [[Code/Deep_Dives/T05_Population_Genetics_Wright_Fisher_Coalescent.py|Code]]
    - **Task:** Simulate forward-in-time neutral allele frequency drift via Wright-Fisher binomial sampling; simulate backward-in-time genealogy mergers via Kingman's coalescent; verify expected coalescent times $\mathbb{E}[T_{MRCA}] = 2N(1 - 1/k)$.
130. **T06: Reaction-Diffusion Turing Patterns (Gray-Scott)**
    - Files: [[Reports/Deep_Dives/T06_Reaction_Diffusion_Turing_Patterns|Report]] | [[Code/Deep_Dives/T06_Reaction_Diffusion_Turing_Patterns.py|Code]]
    - **Task:** Discretize 2D coupled nonlinear Gray-Scott reaction-diffusion PDEs on spatial grids; simulate morphogenesis across feed ($F$) and kill ($k$) parameter spaces; generate self-organizing spot and stripe patterns.

---

## Phase 19: Signal Processing & Classical Control Theory
*Objective: Master spectral analysis, digital filtering, feedback control, state-space estimators, and error-correcting codes.*

131. **U01: Fast Fourier Transform (FFT)**
    - Files: [[Reports/Deep_Dives/U01_Fast_Fourier_Transform_FFT|Report]] | [[Code/Deep_Dives/U01_Fast_Fourier_Transform_FFT.py|Code]]
    - **Task:** Derive and implement Cooley-Tukey Radix-2 decimation-in-time FFT; exploit bit-reversal indexing and twiddle factor symmetry to accelerate discrete Fourier transforms from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.
132. **U02: Digital Filter Design (FIR / IIR Butterworth)**
    - Files: [[Reports/Deep_Dives/U02_Digital_Filter_Design|Report]] | [[Code/Deep_Dives/U02_Digital_Filter_Design.py|Code]]
    - **Task:** Design FIR filters via windowed sinc kernels (Hamming window); design IIR Butterworth filters using bilinear transforms $s = \frac{2}{T}\frac{z-1}{z+1}$; apply Direct Form II difference filtering.
133. **U03: PID Controller & Anti-Windup**
    - Files: [[Reports/Deep_Dives/U03_PID_Controller|Report]] | [[Code/Deep_Dives/U03_PID_Controller.py|Code]]
    - **Task:** Formulate continuous and discrete-time Proportional-Integral-Derivative feedback loops; implement integrator anti-windup clamping and derivative low-pass filtering; stabilize second-order dynamical plants.
134. **U04: State-Space Control & Linear Quadratic Regulator (LQR)**
    - Files: [[Reports/Deep_Dives/U04_State_Space_Control_and_LQR|Report]] | [[Code/Deep_Dives/U04_State_Space_Control_and_LQR.py|Code]]
    - **Task:** Model continuous linear systems $\dot{x} = Ax + Bu$; evaluate Kalman controllability matrix $\mathcal{C} = [B, AB, \dots, A^{n-1}B]$; solve Continuous Algebraic Riccati Equations (CARE) to derive optimal feedback gain $K = R^{-1} B^T P$.
135. **U05: Circuit Simulation (Modified Nodal Analysis)**
    - Files: [[Reports/Deep_Dives/U05_Circuit_Nodal_Analysis|Report]] | [[Code/Deep_Dives/U05_Circuit_Nodal_Analysis.py|Code]]
    - **Task:** Construct conductance and incidence matrices from SPICE netlists; formulate Modified Nodal Analysis (MNA) equations $Gx = b$ enforcing Kirchhoff's Current and Voltage Laws; solve linear DC circuit operating points.
136. **U06: Error-Correcting Codes (Hamming & Reed-Solomon)**
    - Files: [[Reports/Deep_Dives/U06_Error_Correcting_Codes_Hamming_Reed_Solomon|Report]] | [[Code/Deep_Dives/U06_Error_Correcting_Codes_Hamming_Reed_Solomon.py|Code]]
    - **Task:** Build generator $G$ and parity-check $H$ matrices for Hamming(7,4) code; perform non-destructive syndrome detection and 1-bit error correction; implement Galois field $\text{GF}(2^8)$ arithmetic for Reed-Solomon codes.

---

## Phase 20: Numerical Linear Algebra & Scientific Computing
*Objective: Build foundational factorization algorithms, eigenvalue solvers, adaptive numerical quadrature, and root finders.*

137. **V01: QR Decomposition (Householder & Gram-Schmidt)**
    - Files: [[Reports/Deep_Dives/V01_QR_Decomposition|Report]] | [[Code/Deep_Dives/V01_QR_Decomposition.py|Code]]
    - **Task:** Implement Modified Gram-Schmidt and Householder reflector reflections $H = I - 2vv^T/(v^Tv)$; factor $A = QR$ with strict orthogonality $\lVert Q^T Q - I \rVert < 10^{-14}$; solve overdetermined least-squares systems.
138. **V02: LU Decomposition with Partial Pivoting & Linear Solver**
    - Files: [[Reports/Deep_Dives/V02_LU_Decomposition_and_Linear_Solver|Report]] | [[Code/Deep_Dives/V02_LU_Decomposition_and_Linear_Solver.py|Code]]
    - **Task:** Implement Gaussian elimination with row permutation pivoting $PA = LU$; compute matrix determinants via product of diagonal elements $\det(A) = \det(P)\prod U_{ii}$; solve linear systems via forward and back substitution.
139. **V03: QR Algorithm for Eigenvalues & Schur Form**
    - Files: [[Reports/Deep_Dives/V03_QR_Algorithm_for_Eigenvalues|Report]] | [[Code/Deep_Dives/V03_QR_Algorithm_for_Eigenvalues.py|Code]]
    - **Task:** Reduce general matrices to upper Hessenberg form via similarity transformations; apply shifted QR iterations with Wilkinson shifts; isolate real and complex eigenvalues.
140. **V04: Numerical Integration (Simpson, Gauss-Legendre, Adaptive Quadrature)**
    - Files: [[Reports/Deep_Dives/V04_Numerical_Integration|Report]] | [[Code/Deep_Dives/V04_Numerical_Integration.py|Code]]
    - **Task:** Implement composite Simpson's 1/3 rule; compute Gauss-Legendre quadrature roots and weights via Legendre polynomials; implement recursive adaptive Simpson quadrature with local error estimators.
141. **V05: Root Finding Methods (Newton-Raphson, Bisection, Brent-Dekker)**
    - Files: [[Reports/Deep_Dives/V05_Root_Finding_Methods|Report]] | [[Code/Deep_Dives/V05_Root_Finding_Methods.py|Code]]
    - **Task:** Implement Bisection, Newton-Raphson, and Secant solvers; combine bisection reliability with inverse quadratic interpolation in Brent-Dekker's algorithm; demonstrate quadratic convergence $\mathcal{O}(\epsilon^2)$.

---

## Phase 21: Electronic Design Automation & Engineering Optimization
*Objective: Solve NP-hard combinatorial layout problems, logic minimization, structural FEA, and surrogate aerodynamic optimization.*

142. **W01: Maze Routing (Lee's BFS Algorithm)**
    - Files: [[Reports/Deep_Dives/W01_Maze_Routing_Lee_Algorithm|Report]] | [[Code/Deep_Dives/W01_Maze_Routing_Lee_Algorithm.py|Code]]
    - **Task:** Implement breadth-first wavefront propagation on 2D VLSI grid fabrics; route optimal interconnect wires around obstacle blockages; backtrack shortest path connections.
143. **W02: Rectilinear Steiner Minimum Tree (RSMT)**
    - Files: [[Reports/Deep_Dives/W02_Rectilinear_Steiner_Minimum_Tree|Report]] | [[Code/Deep_Dives/W02_Rectilinear_Steiner_Minimum_Tree.py|Code]]
    - **Task:** Construct Hanan grid intersection candidate points; implement minimum spanning tree heuristics over rectilinear metric spaces; minimize total net wirelength.
144. **W03: Force-Directed & Simulated Annealing Placement**
    - Files: [[Reports/Deep_Dives/W03_Force_Directed_and_Simulated_Annealing_Placement|Report]] | [[Code/Deep_Dives/W03_Force_Directed_and_Simulated_Annealing_Placement.py|Code]]
    - **Task:** Model circuit netlists as spring-electrical mechanical systems (Fruchterman-Reingold); optimize macro cell placement via Simulated Annealing minimizing Half-Perimeter Wirelength (HPWL).
145. **W04: VLSI Floorplanning (Slicing Tree & Normalized Polish Expression)**
    - Files: [[Reports/Deep_Dives/W04_VLSI_Floorplanning_Slicing_Tree|Report]] | [[Code/Deep_Dives/W04_VLSI_Floorplanning_Slicing_Tree.py|Code]]
    - **Task:** Represent floorplans as slicing trees encoded by Normalized Polish Expressions; evaluate bounding box dimensions via Stockmeyer area-delay curve additions; optimize chip area via simulated annealing.
146. **W05: Min-Cut Circuit Partitioning (Kernighan-Lin & Fiduccia-Mattheyses)**
    - Files: [[Reports/Deep_Dives/W05_Min_Cut_Circuit_Partitioning|Report]] | [[Code/Deep_Dives/W05_Min_Cut_Circuit_Partitioning.py|Code]]
    - **Task:** Implement Kernighan-Lin pairwise cell swap partitioning; implement linear-time Fiduccia-Mattheyses (FM) bucket array updates; minimize cut nets across balanced partitions.
147. **W06: Boolean Logic Minimization (Quine-McCluskey)**
    - Files: [[Reports/Deep_Dives/W06_Boolean_Logic_Minimization_Quine_McCluskey|Report]] | [[Code/Deep_Dives/W06_Boolean_Logic_Minimization_Quine_McCluskey.py|Code]]
    - **Task:** Group binary minterms by Hamming weight; iteratively eliminate redundant literals to find all Prime Implicants; solve prime implicant coverage tables to find minimal Sum-of-Products (SOP).
148. **W07: Binary Decision Diagrams (BDD & Canonicity)**
    - Files: [[Reports/Deep_Dives/W07_Binary_Decision_Diagrams_BDD|Report]] | [[Code/Deep_Dives/W07_Binary_Decision_Diagrams_BDD.py|Code]]
    - **Task:** Implement Shannon expansions $f = x f_x + x' f_{x'}$; build Reduced Ordered Binary Decision Diagrams (ROBDD) with unique node tables; enforce canonical equivalence checking.
149. **W08: Static Timing Analysis (STA & Setup/Hold Slacks)**
    - Files: [[Reports/Deep_Dives/W08_Static_Timing_Analysis|Report]] | [[Code/Deep_Dives/W08_Static_Timing_Analysis.py|Code]]
    - **Task:** Model synchronous circuits as directed acyclic timing graphs; propagate Arrival Times (AT) and Required Arrival Times (RAT); evaluate setup and hold timing slacks and report critical paths.
150. **W09: SAT Solving (DPLL with Unit Propagation)**
    - Files: [[Reports/Deep_Dives/W09_SAT_Solving_DPLL|Report]] | [[Code/Deep_Dives/W09_SAT_Solving_DPLL.py|Code]]
    - **Task:** Implement Davis-Putnam-Logemann-Loveland (DPLL) Boolean satisfiability solver; apply recursive unit propagation and pure literal elimination; verify 3-SAT satisfiability.
151. **W10: Finite Element Method (Truss Analysis)**
    - Files: [[Reports/Deep_Dives/W10_Finite_Element_Method_Truss_Analysis|Report]] | [[Code/Deep_Dives/W10_Finite_Element_Method_Truss_Analysis.py|Code]]
    - **Task:** Formulate local element stiffness matrices $k_e = rac{EA}{L}$; assemble global stiffness matrix $K$; apply Dirichlet boundary constraints; solve structural displacements $Ku = F$ and internal member stresses.
152. **W11: Topology Optimization (SIMP Method)**
    - Files: [[Reports/Deep_Dives/W11_Topology_Optimization_SIMP|Report]] | [[Code/Deep_Dives/W11_Topology_Optimization_SIMP.py|Code]]
    - **Task:** Implement Solid Isotropic Material with Penalization (SIMP); formulate compliance minimization under volume fraction constraints; apply mesh-independency sensitivity filters and density updates.
153. **W12: Multibody Dynamics (Slider-Crank Mechanism)**
    - Files: [[Reports/Deep_Dives/W12_Multi_Body_Dynamics_Slider_Crank|Report]] | [[Code/Deep_Dives/W12_Multi_Body_Dynamics_Slider_Crank.py|Code]]
    - **Task:** Derive loop closure constraint equations for slider-crank linkages; compute Jacobian matrices; evaluate piston position $x(	heta)$, velocity $v(	heta)$, and acceleration $a(	heta)$ over full crankshaft cycles.
154. **W13: Thermodynamic Cycle Simulation (Otto & Brayton)**
    - Files: [[Reports/Deep_Dives/W13_Engine_Thermodynamic_Cycle_Simulation|Report]] | [[Code/Deep_Dives/W13_Engine_Thermodynamic_Cycle_Simulation.py|Code]]
    - **Task:** Model ideal gas state transformations across isentropic, isobaric, and isochoric processes; calculate net work output and thermal efficiencies $\eta = 1 - r^{1-\gamma}$ for Otto and Brayton gas turbine cycles.
155. **W14: Surrogate-Based Design Optimization (Kriging & Expected Improvement)**
    - Files: [[Reports/Deep_Dives/W14_Surrogate_Based_Design_Optimization|Report]] | [[Code/Deep_Dives/W14_Surrogate_Based_Design_Optimization.py|Code]]
    - **Task:** Train Gaussian Process Kriging metamodels on expensive engineering simulations; evaluate analytical Expected Improvement (EI) acquisition functions; locate global aerodynamic/structural optima.

---

## Phase 22: Robotics, Navigation & Classical Computer Vision
*Objective: Implement autonomous motion planning, sensor fusion, SLAM, robot arm kinematics, and edge/motion vision primitives.*

156. **X01: Rapidly-Exploring Random Trees (RRT)**
    - Files: [[Reports/Deep_Dives/X01_Rapidly_Exploring_Random_Trees_RRT|Report]] | [[Code/Deep_Dives/X01_Rapidly_Exploring_Random_Trees_RRT.py|Code]]
    - **Task:** Sample state spaces incrementally; find nearest tree neighbors and extend configurations with obstacle collision checks; plan collision-free paths in 2D maze environments.
157. **X02: RRT* Optimal Path Planning**
    - Files: [[Reports/Deep_Dives/X02_RRT_Star|Report]] | [[Code/Deep_Dives/X02_RRT_Star.py|Code]]
    - **Task:** Implement near-neighbor ball queries ($r_n = \gamma (\log n / n)^{1/d}$); choose minimum-cost parents and rewire existing vertices; demonstrate asymptotic convergence to optimal path length.
158. **X03: Probabilistic Roadmaps (PRM)**
    - Files: [[Reports/Deep_Dives/X03_Probabilistic_Roadmaps_PRM|Report]] | [[Code/Deep_Dives/X03_Probabilistic_Roadmaps_PRM.py|Code]]
    - **Task:** Multi-query path planning: construct collision-free configuration graphs via random sampling and local planners; query shortest paths between arbitrary start and goal coordinates using $A^*$.
159. **X04: Extended Kalman Filter (EKF)**
    - Files: [[Reports/Deep_Dives/X04_Extended_Kalman_Filter_EKF|Report]] | [[Code/Deep_Dives/X04_Extended_Kalman_Filter_EKF.py|Code]]
    - **Task:** Linearize non-linear motion and range-bearing measurement models via analytical Jacobian matrices ($F_k, H_k$); propagate state and covariance; demonstrate robust tracking under non-linear unicycle kinematics.
160. **X05: Unscented Kalman Filter (UKF)**
    - Files: [[Reports/Deep_Dives/X05_Unscented_Kalman_Filter_UKF|Report]] | [[Code/Deep_Dives/X05_Unscented_Kalman_Filter_UKF.py|Code]]
    - **Task:** Generate deterministically chosen sigma points capturing mean and covariance; propagate sigma points through exact non-linear transformations without Jacobians; evaluate posterior Gaussian parameters.
161. **X06: EKF SLAM (Simultaneous Localization and Mapping)**
    - Files: [[Reports/Deep_Dives/X06_EKF_SLAM|Report]] | [[Code/Deep_Dives/X06_EKF_SLAM.py|Code]]
    - **Task:** Maintain augmented joint state vector $[\mathbf{x}_R, \mathbf{m}_1, \dots, \mathbf{m}_N]^T$; update landmark estimates and robot pose simultaneously; track landmark uncertainty covariance shrinkage upon loop closure.
162. **X07: FastSLAM (Particle Filter SLAM)**
    - Files: [[Reports/Deep_Dives/X07_Particle_Filter_SLAM|Report]] | [[Code/Deep_Dives/X07_Particle_Filter_SLAM.py|Code]]
    - **Task:** Factor SLAM posterior via Rao-Blackwellization into robot path particle filter and independent low-dimensional Kalman filters for each landmark; perform importance weight updates and resampling.
163. **X08: Robot Arm Kinematics (Forward & Inverse Jacobian)**
    - Files: [[Reports/Deep_Dives/X08_Forward_and_Inverse_Kinematics_Jacobian|Report]] | [[Code/Deep_Dives/X08_Forward_and_Inverse_Kinematics_Jacobian.py|Code]]
    - **Task:** Formulate Denavit-Hartenberg (DH) forward kinematic transformation matrices; compute geometric Jacobian $J(q)$; solve inverse kinematics via Damped Least Squares (Levenberg-Marquardt) to reach target 2D/3D end-effector poses.
164. **X09: RANSAC Robust Model Fitting**
    - Files: [[Reports/Deep_Dives/X09_RANSAC|Report]] | [[Code/Deep_Dives/X09_RANSAC.py|Code]]
    - **Task:** Fit geometric models under heavy outlier contamination ($>40\%$ outliers); determine required iterations $k = \frac{\ln(1-p)}{\ln(1-(1-\epsilon)^m)}$; fit consensus lines and homographies cleanly where OLS fails.
165. **X10: Canny Edge Detection**
    - Files: [[Reports/Deep_Dives/X10_Canny_Edge_Detection|Report]] | [[Code/Deep_Dives/X10_Canny_Edge_Detection.py|Code]]
    - **Task:** Multi-stage image processing: 2D Gaussian blur convolution, Sobel gradient vector computation, Non-Maximum Suppression (NMS) along gradient directions, and dual-threshold hysteresis edge tracking.
166. **X11: Lucas-Kanade Optical Flow**
    - Files: [[Reports/Deep_Dives/X11_Lucas_Kanade_Optical_Flow|Report]] | [[Code/Deep_Dives/X11_Lucas_Kanade_Optical_Flow.py|Code]]
    - **Task:** Formulate optical flow brightness constancy equation $I_x u + I_y v + I_t = 0$; solve spatial patch least-squares systems via $A^T A egin{pmatrix} u \ v \end{pmatrix} = -A^T b$; track motion vector fields across image frames.

---

## Phase 23: Advanced Geometry, Topology & Measure Theory (Module Y)
*Objective: Master continuous symmetry groups on manifolds, persistent homology topological shape invariants, and non-Euclidean cross-domain optimal transport.*

167. **Y01: Lie Groups SO(3), SE(3) and Geometric Equivariance**
    - Files: [[Reports/Deep_Dives/Y01_Lie_Groups_SO3_SE3_and_Equivariance|Report]] | [[Code/Deep_Dives/Y01_Lie_Groups_SO3_SE3_and_Equivariance.py|Code]]
    - **Task:** Implement skew-symmetric hat and vee algebra operators; derive closed-form Rodrigues exponential and logarithmic maps; evaluate $6 \times 6$ Adjoint representation matrices; verify exact geometric $\text{SE}(3)$ coordinate equivariance on 3D point cloud manifolds.
168. **Y02: Topological Data Analysis (TDA) & Persistent Homology**
    - Files: [[Reports/Deep_Dives/Y02_Topological_Data_Analysis_Persistent_Homology|Report]] | [[Code/Deep_Dives/Y02_Topological_Data_Analysis_Persistent_Homology.py|Code]]
    - **Task:** Construct Vietoris-Rips simplicial complexes across filtration scales; formulate boundary operator matrices; implement standard column-reduction algorithm over $\mathbb{Z}_2$; extract Betti numbers ($\beta_0, \beta_1$) and persistent topological barcodes from noisy metric clouds.
169. **Y03: Gromov-Wasserstein & Sliced Optimal Transport**
    - Files: [[Reports/Deep_Dives/Y03_Gromov_Wasserstein_and_Sliced_Optimal_Transport|Report]] | [[Code/Deep_Dives/Y03_Gromov_Wasserstein_and_Sliced_Optimal_Transport.py|Code]]
    - **Task:** Implement Sliced Wasserstein Distance via 1D Radon random unit projections and analytical sorting in $\mathcal{O}(K N \log N)$; formulate entropic Gromov-Wasserstein discrepancy for cross-domain metric alignment across incomparable dimensions; recover ground-truth graph isomorphisms.

---

## Phase 24: Geometry, Cryptography, Information Theory & Causal Discovery (Module Z)
*Objective: Master spatial computational geometry, interactive zero-knowledge proofs, elliptic curve digital signatures, universal Kolmogorov compression, and constraint-based causal discovery.*

170. **Z01: Classical Computational Geometry (Convex Hull & Delaunay Triangulation)**
    - Files: [[Reports/Deep_Dives/Z01_Computational_Geometry_Convex_Hull_and_Delaunay|Report]] | [[Code/Deep_Dives/Z01_Computational_Geometry_Convex_Hull_and_Delaunay.py|Code]]
    - **Task:** Implement Graham Scan 2D convex hull with polar angle pivot sorting and CCW cross-product turns; build the Bowyer-Watson incremental algorithm for Delaunay triangulation enforcing empty circumcircle determinants across 2D point clouds.
171. **Z02: Zero-Knowledge Proofs & Verifiable Computation (Schnorr Protocol & R1CS)**
    - Files: [[Reports/Deep_Dives/Z02_Zero_Knowledge_Proofs_and_Verifiable_Computation|Report]] | [[Code/Deep_Dives/Z02_Zero_Knowledge_Proofs_and_Verifiable_Computation.py|Code]]
    - **Task:** Implement the Schnorr $\Sigma$-protocol proving discrete logarithm knowledge in zero-knowledge; verify completeness, special soundness knowledge extraction, and honest-verifier zero-knowledge simulation; compile and verify Rank-1 Constraint System (R1CS) arithmetic circuits over prime fields.
172. **Z03: Elliptic Curve Cryptography & Hash Trees (secp256k1 ECDSA & Merkle Trees)**
    - Files: [[Reports/Deep_Dives/Z03_Elliptic_Curve_Cryptography_and_Hash_Trees|Report]] | [[Code/Deep_Dives/Z03_Elliptic_Curve_Cryptography_and_Hash_Trees.py|Code]]
    - **Task:** Implement short Weierstrass elliptic curve group laws ($y^2 = x^3 + 7 \pmod p$), affine point addition, doubling, and double-and-add scalar multiplication on secp256k1; generate and verify ECDSA digital signatures; build cryptographic binary Merkle trees and verify logarithmic audit path proofs.
173. **Z04: Algorithmic Information Theory & Universal Compression (Huffman, LZW & NCD)**
    - Files: [[Reports/Deep_Dives/Z04_Algorithmic_Information_Theory_and_Universal_Compression|Report]] | [[Code/Deep_Dives/Z04_Algorithmic_Information_Theory_and_Universal_Compression.py|Code]]
    - **Task:** Construct prefix-free Huffman trees via min-heaps and prove the Shannon Source Coding theorem $H(X) \le \bar{L} < H(X) + 1$; implement streaming LZW dictionary compression and exact inversion; evaluate Normalized Compression Distance (NCD) approximating Kolmogorov complexity for parameter-free clustering.
174. **Z05: Causal Discovery & Graph Learning (The PC Algorithm)**
    - Files: [[Reports/Deep_Dives/Z05_Causal_Discovery_PC_Algorithm|Report]] | [[Code/Deep_Dives/Z05_Causal_Discovery_PC_Algorithm.py|Code]]
    - **Task:** Reconstruct the Markov Equivalence Class (CPDAG) of observational data from scratch; execute Phase 1 skeleton search using precision submatrix partial correlations and Fisher's $z$-transforms; orient unshielded colliders (v-structures) via separating sets; propagate Meek's orientation rules to avoid spurious cycles.


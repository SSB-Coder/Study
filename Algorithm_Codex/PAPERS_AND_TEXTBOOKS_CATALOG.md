# Canonical Papers and Textbooks Catalog

This catalog documents the foundational literature, seminal research papers, and definitive graduate-level textbooks that establish the theoretical bedrock of the algorithms in **`Algorithm_Codex/`**.

---

## 1. Foundational Textbooks

1. **Reinforcement Learning: An Introduction (2nd Edition)**
   - *Authors:* Richard S. Sutton and Andrew G. Barto (2018)
   - *The Standard Reference For:* Multi-armed bandits, dynamic programming (Value/Policy Iteration), Monte Carlo methods, Temporal-Difference learning (SARSA, Q-learning), n-step bootstrapping, policy gradients, and eligibility traces.
   - *Core Takeaway:* The mathematical unification of model-based and model-free reinforcement learning under the Bellman equations and generalized policy iteration.

2. **Pattern Recognition and Machine Learning (PRML)**
   - *Author:* Christopher M. Bishop (2006)
   - *The Standard Reference For:* Bayesian probability, linear models for regression and classification, kernel methods, sparse kernel machines (SVM), graphical models, EM algorithm, continuous latent variables (PCA, PPCA), mixture models, and approximate inference (variational inference and sampling).
   - *Core Takeaway:* A Bayesian perspective unifying linear estimation, graphical modeling, and non-parametric learning.

3. **The Elements of Statistical Learning: Data Mining, Inference, and Prediction (ESL)**
   - *Authors:* Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2nd Edition, 2009)
   - *The Standard Reference For:* Shrinkage and regularized regression (Ridge, Lasso, Elastic Net), generalized additive models, tree-based methods (CART), ensemble learning (Bagging, Random Forests, AdaBoost, Gradient Boosting), support vector machines, and unsupervised learning.
   - *Core Takeaway:* The frequentist, statistical machine learning bible formalizing the bias-variance tradeoff and empirical risk minimization.

4. **Probabilistic Machine Learning: An Introduction & Advanced Topics**
   - *Author:* Kevin P. Murphy (MIT Press, 2022 / 2023)
   - *The Standard Reference For:* Modern treatment of exponential families, Generalized Linear Models, Gaussian Processes, MCMC, Variational Inference, and the bridge between classical Bayesian statistics and modern deep learning.

5. **Convex Optimization**
   - *Authors:* Stephen Boyd and Lieven Vandenberghe (Cambridge University Press, 2004)
   - *The Standard Reference For:* Convex sets and functions, convex optimization problems, Lagrange duality, Karush-Kuhn-Tucker (KKT) optimality conditions, unconstrained and constrained minimization, interior-point methods, and the Simplex algorithm.

6. **Probabilistic Graphical Models: Principles and Techniques**
   - *Authors:* Daphne Koller and Nir Friedman (MIT Press, 2009)
   - *The Standard Reference For:* Directed Bayesian networks, undirected Markov random fields, exact inference via variable elimination and clique trees, approximate inference (loopy belief propagation), parameter estimation, and structure learning.

7. **Artificial Intelligence: A Modern Approach (AIMA, 4th Edition)**
   - *Authors:* Stuart Russell and Peter Norvig (Pearson, 2020)
   - *The Standard Reference For:* Uninformed and heuristic search (BFS, DFS, A*), adversarial search (Minimax, Alpha-Beta pruning), constraint satisfaction problems (CSP, AC-3), logical agents, classical planning, and game theory.

---

## 2. Seminal Research Papers by Module

### Module A — Statistical Foundations
- **Student (William Sealy Gosset)** (1908). *"The Probable Error of a Mean."* Biometrika, 6(1), 1-25. (Derivation of Student's $t$-distribution).
- **Pearson, K.** (1900). *"On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling."* Philosophical Magazine, 50(302), 157-175. ($\chi^2$ test).
- **Efron, B.** (1979). *"Bootstrap Methods: Another Look at the Jackknife."* The Annals of Statistics, 7(1), 1-26. (The non-parametric statistical bootstrap).

### Module B — Classical Supervised Learning
- **Cortes, C., & Vapnik, V.** (1995). *"Support-Vector Networks."* Machine Learning, 20(3), 273-297. (Soft-margin SVM and the kernel trick).
- **Platt, J.** (1998). *"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines."* Microsoft Research Technical Report MSR-TR-98-14. (Fast analytical SMO solver).
- **Breiman, L., Friedman, J., Olshen, R., & Stone, C.** (1984). *"Classification and Regression Trees."* Wadsworth. (The definitive CART algorithm).
- **Nelder, J. A., & Wedderburn, R. W. M.** (1972). *"Generalized Linear Models."* Journal of the Royal Statistical Society: Series A, 135(3), 370-384. (Unified framework for GLMs and IRLS).
- **Platt, J.** (1999). *"Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods."* Advances in Large Margin Classifiers. (Platt scaling calibration).

### Module C — Unsupervised Learning & Dimensionality Reduction
- **Arthur, D., & Vassilvitskii, S.** (2007). *"k-means++: The Advantages of Careful Seeding."* Proceedings of the 18th Annual ACM-SIAM SODA, 1027-1035. ($D^2$-sampling $O(\log k)$ approximation guarantee).
- **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.** (1996). *"A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN)."* KDD-96, 226-231.
- **Dempster, A. P., Laird, N. M., & Rubin, D. B.** (1977). *"Maximum Likelihood from Incomplete Data via the EM Algorithm."* Journal of the Royal Statistical Society, Series B, 39(1), 1-38.
- **Pearson, K.** (1901). *"On Lines and Planes of Closest Fit to Systems of Points in Space."* Philosophical Magazine, 2(11), 559-572. (First mathematical derivation of PCA).
- **Hyvärinen, A., & Oja, E.** (2000). *"Independent Component Analysis: Algorithms and Applications."* Neural Networks, 13(4-5), 411-430. (FastICA algorithm via negentropy).
- **van der Maaten, L., & Hinton, G.** (2008). *"Visualizing Data using t-SNE."* Journal of Machine Learning Research, 9, 2579-2605.
- **Tenenbaum, J. B., de Silva, V., & Langford, J. C.** (2000). *"A Global Geometric Framework for Nonlinear Dimensionality Reduction (Isomap)."* Science, 290(5500), 2319-2323.
- **Roweis, S. T., & Saul, L. K.** (2000). *"Nonlinear Dimensionality Reduction by Locally Linear Embedding (LLE)."* Science, 290(5500), 2323-2326.
- **McInnes, L., Healy, J., & Melville, J.** (2018). *"UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction."* arXiv:1802.03426.

### Module D — Ensemble Methods
- **Breiman, L.** (1996). *"Bagging Predictors."* Machine Learning, 24(2), 123-140.
- **Breiman, L.** (2001). *"Random Forests."* Machine Learning, 45(1), 5-32.
- **Freund, Y., & Schapire, R. E.** (1997). *"A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (AdaBoost)."* Journal of Computer and System Sciences, 55(1), 119-139.
- **Friedman, J. H.** (2001). *"Greedy Function Approximation: A Gradient Boosting Machine."* The Annals of Statistics, 29(5), 1189-1232.
- **Wolpert, D. H.** (1992). *"Stacked Generalization."* Neural Networks, 5(2), 241-259.

### Module E — Probabilistic Graphical Models & Bayesian Methods
- **Rabiner, L. R.** (1989). *"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition."* Proceedings of the IEEE, 77(2), 257-286. (The definitive HMM tutorial).
- **Hastings, W. K.** (1970). *"Monte Carlo Sampling Methods Using Markov Chains and Their Applications."* Biometrika, 57(1), 97-109. (Metropolis-Hastings MCMC).
- **Geman, S., & Geman, D.** (1984). *"Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images."* IEEE PAMI, 6(6), 721-741. (Gibbs sampling).
- **Blei, D. M., Kucukelbir, A., & McAuliffe, J. D.** (2017). *"Variational Inference: A Review for Statisticians."* Journal of the American Statistical Association, 112(518), 859-877. (CAVI).
- **Rasmussen, C. E., & Williams, C. K. I.** (2006). *"Gaussian Processes for Machine Learning."* MIT Press.
- **Lafferty, J., McCallum, A., & Pereira, F.** (2001). *"Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data."* ICML-01, 282-289.
- **Blei, D. M., Ng, A. Y., & Jordan, M. I.** (2003). *"Latent Dirichlet Allocation."* Journal of Machine Learning Research, 3, 993-1022.

### Module F — Classical Time Series
- **Box, G. E. P., & Jenkins, G. M.** (1970). *"Time Series Analysis: Forecasting and Control."* Holden-Day. (ARIMA modeling framework).
- **Kaplan, E. L., & Meier, P.** (1958). *"Nonparametric Estimation from Incomplete Observations."* JASA, 53(282), 457-481.
- **Cox, D. R.** (1972). *"Regression Models and Life-Tables."* Journal of the Royal Statistical Society, Series B, 34(2), 187-220. (Proportional Hazards model).
- **Engle, R. F.** (1982). *"Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation."* Econometrica, 50(4), 987-1007. (ARCH, Nobel Prize 2003).
- **Bollerslev, T.** (1986). *"Generalized Autoregressive Conditional Heteroskedasticity."* Journal of Econometrics, 31(3), 307-327. (GARCH).

### Module G — Reinforcement Learning
- **Auer, P., Cesa-Bianchi, N., & Fischer, P.** (2002). *"Finite-time Analysis of the Multiarmed Bandit Problem."* Machine Learning, 47(2-3), 235-256. (UCB1 algorithm).
- **Watkins, C. J. C. H., & Dayan, P.** (1992). *"Q-learning."* Machine Learning, 8(3-4), 279-292.
- **Mnih, V. et al.** (2015). *"Human-level control through deep reinforcement learning."* Nature, 518(7540), 529-533. (Nature DQN).
- **van Hasselt, H., Guez, A., & Silver, D.** (2016). *"Deep Reinforcement Learning with Double Q-learning."* AAAI-16.
- **Wang, Z. et al.** (2016). *"Dueling Network Architectures for Deep Reinforcement Learning."* ICML-16.
- **Williams, R. J.** (1992). *"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning."* Machine Learning, 8(3-4), 229-256. (REINFORCE algorithm).
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *"Proximal Policy Optimization Algorithms."* arXiv:1707.06347. (PPO).
- **Lillicrap, T. P. et al.** (2015). *"Continuous control with deep reinforcement learning."* arXiv:1509.02971. (DDPG).
- **Coulom, R.** (2006). *"Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search."* Computers and Games, 72-83.
- **Kocsis, L., & Szepesvári, C.** (2006). *"Bandit Based Monte-Carlo Planning (UCT)."* ECML-06, 282-293.
- **Ng, A. Y., & Russell, S.** (2000). *"Algorithms for Inverse Reinforcement Learning."* ICML-00.

### Module H — Evolutionary & Nature-Inspired Optimization
- **Holland, J. H.** (1975). *"Adaptation in Natural and Artificial Systems."* University of Michigan Press. (Genetic Algorithms).
- **Koza, J. R.** (1992). *"Genetic Programming: On the Programming of Computers by Means of Natural Selection."* MIT Press.
- **Hansen, N., & Ostermeier, A.** (2001). *"Completely Derandomized Self-Adaptation in Evolution Strategies."* Evolutionary Computation, 9(2), 159-195. (CMA-ES).
- **Kennedy, J., & Eberhart, R.** (1995). *"Particle Swarm Optimization."* Proceedings of IEEE ICNN, 1942-1948.
- **Dorigo, M., Maniezzo, V., & Colorni, A.** (1996). *"Ant System: Optimization by a Colony of Cooperating Agents."* IEEE TSMC-B, 26(1), 29-41.
- **Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P.** (1983). *"Optimization by Simulated Annealing."* Science, 220(4598), 671-680.
- **Storn, R., & Price, K.** (1997). *"Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces."* Journal of Global Optimization, 11(4), 341-359.
- **Stanley, K. O., & Miikkulainen, R.** (2002). *"Evolving Neural Networks through Augmenting Topologies (NEAT)."* Evolutionary Computation, 10(2), 99-127.
- **Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T.** (2002). *"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."* IEEE TEVC, 6(2), 182-197.

### Module I — Classical Optimization Theory
- **Polyak, B. T.** (1964). *"Some methods of speeding up the convergence of iteration methods."* USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17. (Heavy-Ball Momentum).
- **Nesterov, Y.** (1983). *"A method for solving the convex programming problem with convergence rate $O(1/k^2)$."* Soviet Mathematics Doklady.
- **Duchi, J., Hazan, E., & Singer, Y.** (2011). *"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (AdaGrad)."* JMLR, 12, 2121-2159.
- **Tieleman, T., & Hinton, G.** (2012). *"Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude."* COURSERA: Neural Networks for Machine Learning.
- **Kingma, D. P., & Ba, J.** (2014). *"Adam: A Method for Stochastic Optimization."* arXiv:1412.6980.
- **Broyden, C. G.** (1970). *"The Convergence of a Class of Double-rank Minimization Algorithms."* Journal of the Institute of Mathematics and Its Applications, 6(1), 76-90. (BFGS).
- **Dantzig, G. B.** (1947). *"Linear Programming and Extensions."* Princeton University Press. (The Simplex algorithm).

### Module J — Other Learning Paradigms
- **Zhu, X., & Ghahramani, Z.** (2002). *"Learning from Labeled and Unlabeled Data with Label Propagation."* CMU-CALD-02-107.
- **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.** (2020). *"A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)."* ICML-20.
- **Finn, C., Abbeel, P., & Levine, S.** (2017). *"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)."* ICML-17.
- **McMahan, H. B. et al.** (2017). *"Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)."* AISTATS-17.
- **Kirkpatrick, J. et al.** (2017). *"Overcoming catastrophic forgetting in neural networks (EWC)."* PNAS, 114(13), 3521-3526.
- **Pearl, J.** (2009). *"Causality: Models, Reasoning, and Inference (2nd Edition)."* Cambridge University Press.
- **Liu, F. T., Ting, K. M., & Zhou, Z.-H.** (2008). *"Isolation Forest."* IEEE ICDM-08, 413-422.
- **Koren, Y., Bell, R., & Volinsky, C.** (2009). *"Matrix Factorization Techniques for Recommender Systems."* Computer, 42(8), 30-37.
- **Lundberg, S. M., & Lee, S.-I.** (2017). *"A Unified Approach to Interpreting Model Predictions (SHAP)."* NeurIPS-17.
- **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). *"'Why Should I Trust You?': Explaining the Predictions of Any Classifier (LIME)."* KDD-16.

### Module K — Neural-Net Techniques Missing From Transformer/
- **Hopfield, J. J.** (1982). *"Neural networks and physical systems with emergent collective computational abilities."* PNAS, 79(8), 2554-2558.
- **Ramsauer, H. et al.** (2020). *"Hopfield Networks is All You Need."* ICLR-21.
- **Dinh, L., Sohl-Dickstein, J., & Bengio, S.** (2016). *"Density estimation using Real NVP."* ICLR-17.
- **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D.** (2018). *"Neural Ordinary Differential Equations."* NeurIPS-18. (Best Paper Award).
- **Sabour, S., Frosst, N., & Hinton, G. E.** (2017). *"Dynamic Routing Between Capsules."* NeurIPS-17.
- **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations."* Journal of Computational Physics, 378, 686-707.
- **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J.** (2013). *"Distributed Representations of Words and Phrases and their Compositionality (Word2Vec SGNS)."* NeurIPS-13.
- **Jaeger, H.** (2001). *"The 'echo state' approach to analysing and training recurrent neural networks."* GMD Report 148, German National Research Center for Information Technology.

### Module L — Classical AI Search & Symbolic Reasoning
- **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). *"A Formal Basis for the Heuristic Determination of Minimum Cost Paths (A*)."* IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
- **Knuth, D. E., & Moore, R. W.** (1975). *"An analysis of alpha-beta pruning."* Artificial Intelligence, 6(4), 293-326.
- **Mackworth, A. K.** (1977). *"Consistency in networks of relations (AC-3)."* Artificial Intelligence, 8(1), 99-118.
- **Zadeh, L. A.** (1965). *"Fuzzy sets."* Information and Control, 8(3), 338-353.
- **Mamdani, E. H.** (1974). *"Application of fuzzy algorithms for control of simple dynamic plant."* Proceedings of the IEE, 121(12), 1585-1588.
- **Page, L., Brin, S., Motwani, R., & Winograd, T.** (1999). *"The PageRank Citation Ranking: Bringing Order to the Web."* Stanford InfoLab Technical Report.
- **von Neumann, J.** (1928). *"Zur Theorie der Gesellschaftsspiele."* Mathematische Annalen, 100, 295-320. (The Minimax Theorem).
- **Nash, J. F.** (1950). *"Equilibrium Points in N-person Games."* PNAS, 36(1), 48-49. (Nobel Prize in Economics).
---

## Seminal Papers for Modules F, H, J, K, and L Foundations

### Survival Analysis (Module F)
- **Kaplan, E. L., & Meier, P. (1958).** *Nonparametric estimation from incomplete observations.* Journal of the American Statistical Association, 53(282), 457-481.
  - *Significance:* Formulated the product-limit estimator for right-censored time-to-event survival curves.
- **Cox, D. R. (1972).** *Regression models and life-tables.* Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187-202.
  - *Significance:* Introduced the semi-parametric proportional hazards model and the concept of partial likelihood.

### Causal Inference & Explainability (Module J)
- **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference.* Cambridge University Press.
  - *Significance:* The definitive treatise on structural causal models (SCMs), causal DAGs, and the do-calculus.
- **Rosenbaum, P. R., & Rubin, D. B. (1983).** *The central role of the propensity score in observational studies for causal effects.* Biometrika, 70(1), 41-55.
  - *Significance:* Proved that conditioning on propensity scores eliminates confounding in observational data.
- **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* NeurIPS 2017.
  - *Significance:* Connected cooperative game theory Shapley values to machine learning explanations (SHAP).

### Neural ODEs & Normalizing Flows (Module K)
- **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018).** *Neural ordinary differential equations.* NeurIPS 2018.
  - *Significance:* Continuous-depth neural networks solved via numerical ODE integrators and continuous adjoint sensitivity.
- **Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017).** *Density estimation using Real NVP.* ICLR 2017.
  - *Significance:* Affine coupling layers providing exact invertible generative modeling with triangular Jacobians.
- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
  - *Significance:* Formulated PINNs embedding physical differential residuals into neural loss functions.
- **Sabour, S., Frosst, N., & Hinton, G. E. (2017).** *Dynamic routing between capsules.* NeurIPS 2017.
  - *Significance:* Vector-valued capsule activations with squashing non-linearities and routing-by-agreement.

### Classical Graph Algorithms & Fuzzy Systems (Module L)
- **Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).** *The PageRank citation ranking: Bringing order to the web.* Stanford InfoLab Technical Report.
  - *Significance:* Power iteration over the stochastic Google transition matrix with random damping.
- **von Luxburg, U. (2007).** *A tutorial on spectral clustering.* Statistics and Computing, 17(4), 395-416.
  - *Significance:* Comprehensive mathematical formulation of normalized graph Laplacians and spectral graph partitioning.
- **Zadeh, L. A. (1965).** *Fuzzy sets.* Information and Control, 8(3), 338-353.
  - *Significance:* Introduced continuous fuzzy membership functions and fuzzy set theory.
- **Mamdani, E. H. (1974).** *Application of fuzzy algorithms for control of simple dynamic plant.* Proceedings of the Institution of Electrical Engineers, 121(12), 1585-1588.
  - *Significance:* Designed the first practical fuzzy logic controller using min-implication and centroid defuzzification.

### Vector Search & Streaming Systems (Module M)
- **Malkov, Y. A., & Yashunin, D. A. (2018).** *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824-836.
  - *Significance:* The seminal paper introducing HNSW, powering FAISS, Milvus, Weaviate, and modern Vector DB retrieval for RAG.
- **Charikar, M. S. (2002).** *Similarity estimation techniques from rounding algorithms (SimHash).* Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC), 380-388.
  - *Significance:* Formulated random hyperplane locality-sensitive hashing for angular cosine similarity.
- **Cormode, G., & Muthukrishnan, S. (2005).** *An improved data stream summary: the count-min sketch and its applications.* Journal of Algorithms, 55(1), 58-75.
  - *Significance:* Proved sub-linear frequency estimation bounds under universal pairwise independent hashing.
- **Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).** *HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm.* Discrete Mathematics & Theoretical Computer Science, AH, 137-156.
  - *Significance:* Harmonic mean stochastic register averaging for big data distinct count estimation.

### Autodiff Engines & Compilers (Module N)
- **Griewank, A., & Walther, A. (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd Edition).* SIAM.
  - *Significance:* The definitive mathematical textbook on reverse-mode and forward-mode algorithmic differentiation.
- **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).** *Automatic differentiation in machine learning: a survey.* Journal of Machine Learning Research, 18(153), 1-43.
  - *Significance:* Rigorous survey tracing dual numbers, computational DAGs, and deep learning autodiff tape systems.

### Optimal Transport & Information Geometry (Module O)
- **Cuturi, M. (2013).** *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* Advances in Neural Information Processing Systems (NeurIPS 2013), 2292-2300.
  - *Significance:* Introduced entropic regularization to optimal transport, reducing computation from cubic linear programs to $O(N^2)$ matrix scaling.
- **Peyré, G., & Cuturi, M. (2019).** *Computational Optimal Transport.* Foundations and Trends in Machine Learning, 11(5-6), 355-607.
  - *Significance:* Comprehensive reference on Wasserstein metrics, Monge-Kantorovich formulations, and entropic algorithms.
- **Amari, S. (1998).** *Natural Gradient Works Efficiently in Learning.* Neural Computation, 10(2), 251-276.
  - *Significance:* Formulated Natural Gradient Descent on Riemannian statistical manifolds using the Fisher Information Metric.

### Physics-Based Sampling & Diffusion Foundations (Module P)
- **Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987).** *Hybrid Monte Carlo.* Physics Letters B, 195(2), 216-222.
  - *Significance:* Combined Hamiltonian dynamics with Metropolis MCMC to eliminate random-walk behavior in high-dimensional sampling.
- **Roberts, G. O., & Tweedie, R. L. (1996).** *Exponential convergence of Langevin distributions and their discrete approximations.* Bernoulli, 2(4), 341-363.
  - *Significance:* Established convergence theory for ULA and the Metropolis-Adjusted Langevin Algorithm (MALA).
- **Vincent, P. (2011).** *A connection between score matching and denoising autoencoders.* Neural Computation, 23(7), 1661-1674.
  - *Significance:* Proved that denoising autoencoders implicitly learn the score of smoothed data distributions via Denoising Score Matching.
- **Song, Y., & Ermon, S. (2019).** *Generative modeling by estimating gradients of the data distribution.* NeurIPS 2019.
  - *Significance:* Bridged score matching and annealed Langevin dynamics, launching modern Score-Based Generative Models.

### Multi-Agent Coordination & Mechanism Design (Module Q)
- **Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018).** *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.* ICML 2018.
  - *Significance:* Formulated monotonic hypernetwork-conditioned value factorization for Centralized Training with Decentralized Execution.
- **Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).** *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG).* NeurIPS 2017.
  - *Significance:* Decentralized actors with centralized joint-state critics for continuous multi-agent reinforcement learning.
- **Nisan, N., Roughgarden, T., Tardos, E., & Vazirani, V. V. (2007).** *Algorithmic Game Theory.* Cambridge University Press.
  - *Significance:* Comprehensive foundation on mechanism design, truthful bidding, and the Vickrey-Clarke-Groves (VCG) auction family.

### Quantum Computing & Quantum Information (Module R)
- **Nielsen, M. A., & Chuang, I. L. (2010).** *Quantum Computation and Quantum Information (10th Anniversary Edition).* Cambridge University Press.
  - *Significance:* The definitive textbook on quantum circuits, quantum algorithms, density matrices, and quantum error correction.
- **Peruzzo, A., et al. (2014).** *A variational eigenvalue solver on a photonic quantum processor.* Nature Communications, 5, 4213.
  - *Significance:* The seminal paper introducing the Variational Quantum Eigensolver (VQE) for NISQ quantum chemistry.
- **Farhi, E., Goldstone, J., & Gutmann, S. (2014).** *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028.
  - *Significance:* Formulated QAOA for combinatorial optimization on near-term quantum processors.
- **Grover, L. K. (1996).** *A fast quantum mechanical algorithm for database search.* Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC), 212-219.
  - *Significance:* Proved quadratic speedup $\mathcal{O}(\sqrt{N})$ for quantum unstructured database queries.
- **Shor, P. W. (1994).** *Algorithms for quantum computation: discrete logarithms and factoring.* IEEE FOCS 1994, 124-134.
  - *Significance:* Introduced polynomial-time quantum period finding and integer factorization via QFT.

### Computational Physics & Celestial Mechanics (Module S)
- **Verlet, L. (1967).** *Computer "experiments" on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules.* Physical Review, 159(1), 98-103.
  - *Significance:* Introduced the Velocity-Verlet symplectic integrator for molecular dynamics.
- **Onsager, L. (1944).** *Crystal statistics. I. A two-dimensional model with an order-disorder transition.* Physical Review, 65(3-4), 117-149.
  - *Significance:* Exact analytical solution of the 2D Ising model showing spontaneous magnetization at $T_c \approx 2.269$.
- **Scargle, J. D. (1982).** *Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data.* The Astrophysical Journal, 263, 835-853.
  - *Significance:* Formulated the Lomb-Scargle periodogram invariant to time translations for uneven astronomical observations.
- **Battin, R. H. (1999).** *An Introduction to the Mathematics and Methods of Astrodynamics.* AIAA Education Series.
  - *Significance:* Canonical reference on Keplerian orbital propagation and Lambert's problem.

### Computational Biology & Bioinformatics (Module T)
- **Needleman, S. B., & Wunsch, C. D. (1970).** *A general method applicable to the search for similarities in the amino acid sequence of two proteins.* Journal of Molecular Biology, 48(3), 443-453.
  - *Significance:* First dynamic programming algorithm for global biological sequence alignment.
- **Smith, T. F., & Waterman, M. S. (1981).** *Identification of common molecular subsequences.* Journal of Molecular Biology, 147(1), 195-197.
  - *Significance:* Formulated local sequence motif alignment with non-negative score resets.
- **Saitou, N., & Nei, M. (1987).** *The neighbor-joining method: a new method for reconstructing phylogenetic trees.* Molecular Biology and Evolution, 4(4), 406-425.
  - *Significance:* The Neighbor-Joining algorithm for building phylogenetic trees from distance matrices.
- **Rogers, D., & Hahn, M. (2010).** *Extended-connectivity fingerprints.* Journal of Chemical Information and Modeling, 50(5), 742-754.
  - *Significance:* Defined circular Morgan fingerprints (ECFP) for molecular representation and similarity scoring.
- **Turing, A. M. (1952).** *The chemical basis of morphogenesis.* Philosophical Transactions of the Royal Society B, 237(641), 37-72.
  - *Significance:* Proved that coupled reaction-diffusion systems break symmetry to create emergent biological patterns.

### Signal Processing & Classical Control Theory (Module U)
- **Cooley, J. W., & Tukey, J. W. (1965).** *An algorithm for the machine calculation of complex Fourier series.* Mathematics of Computation, 19(90), 297-301.
  - *Significance:* Introduced the Radix-2 decimation-in-time Fast Fourier Transform.
- **Oppenheim, A. V., & Schafer, R. W. (2009).** *Discrete-Time Signal Processing (3rd Edition).* Prentice Hall.
  - *Significance:* The standard reference on digital filter design (FIR/IIR), Z-transforms, and FFT algorithms.
- **Åström, K. J., & Murray, R. M. (2021).** *Feedback Systems: An Introduction for Scientists and Engineers (2nd Edition).* Princeton University Press.
  - *Significance:* Foundational textbook on PID feedback, stability margins, and state-space control.
- **Ho, C.-W., Ruehli, A. E., & Brennan, P. A. (1975).** *The modified nodal approach to network analysis.* IEEE Transactions on Circuits and Systems, 22(6), 504-509.
  - *Significance:* Introduced Modified Nodal Analysis (MNA), the matrix solver foundation of SPICE.

### Numerical Linear Algebra & Scientific Computing (Module V)
- **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations (4th Edition).* Johns Hopkins University Press.
  - *Significance:* The universal bible of numerical linear algebra: QR, LU, SVD, Schur forms, and condition numbers.
- **Trefethen, L. N., & Bau, D. (1997).** *Numerical Linear Algebra.* SIAM.
  - *Significance:* Classical treatise on Householder reflections, Modified Gram-Schmidt stability, and QR iterations.
- **Brent, R. P. (1973).** *Algorithms for Minimization without Derivatives.* Prentice-Hall.
  - *Significance:* Formulated the robust Brent-Dekker algorithm combining bisection and inverse quadratic interpolation.

### Electronic Design Automation & Engineering Optimization (Module W)
- **Lee, C. Y. (1961).** *An algorithm for path connections and its applications.* IRE Transactions on Electronic Computers, EC-10(3), 346-365.
  - *Significance:* Seminal paper introducing Lee's BFS wavefront algorithm for printed circuit routing.
- **Bryant, R. E. (1986).** *Graph-based algorithms for boolean function manipulation.* IEEE Transactions on Computers, C-35(8), 677-691.
  - *Significance:* Introduced Reduced Ordered Binary Decision Diagrams (ROBDDs) and canonical graph equivalence.
- **Bendsøe, M. P., & Sigmund, O. (2003).** *Topology Optimization: Theory, Methods, and Applications.* Springer.
  - *Significance:* The definitive textbook on the SIMP method for structural finite element topology optimization.
- **Jones, D. R., Schonlau, M., & Welch, W. J. (1998).** *Efficient global optimization of expensive black-box functions.* Journal of Global Optimization, 13(4), 455-492.
  - *Significance:* Formulated the Expected Improvement (EI) acquisition function for Kriging/Gaussian Process optimization.

### Robotics, Navigation & Classical Computer Vision (Module X)
- **LaValle, S. M. (1998).** *Rapidly-exploring random trees: A new tool for path planning.* Technical Report, Iowa State University.
  - *Significance:* Introduced RRT sampling for high-dimensional robotic motion planning.
- **Karaman, S., & Frazzoli, E. (2011).** *Sampling-based algorithms for optimal motion planning (RRT*).* International Journal of Robotics Research, 30(7), 846-894.
  - *Significance:* Proved asymptotic optimality of RRT* via near-neighbor rewiring.
- **Julier, S. J., & Uhlmann, J. K. (2004).** *Unscented filtering and nonlinear estimation.* Proceedings of the IEEE, 92(3), 401-422.
  - *Significance:* Introduced the Unscented Kalman Filter (UKF) and deterministic sigma point propagation.
- **Canny, J. (1986).** *A computational approach to edge detection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
  - *Significance:* Seminal paper defining the multi-stage Canny edge detector: Gaussian, Sobel, NMS, and hysteresis.
- **Lucas, B. D., & Kanade, T. (1981).** *An iterative image registration technique with an application to stereo vision.* Proceedings of Imaging Understanding Workshop, 121-130.
  - *Significance:* Introduced the Lucas-Kanade spatial least-squares optical flow algorithm.

### Advanced Geometry, Topology & Measure Theory (Module Y)
- **Hall, B. C. (2015).** *Lie Groups, Lie Algebras, and Representations: An Elementary Introduction (2nd Edition).* Springer Graduate Texts in Mathematics.
  - *Significance:* The standard mathematical reference on matrix Lie groups, Baker-Campbell-Hausdorff formula, and Adjoint representations.
- **Edelsbrunner, H., & Harer, J. (2010).** *Computational Topology: An Introduction.* American Mathematical Society.
  - *Significance:* Foundational textbook establishing the Vietoris-Rips filtration and matrix reduction algorithm for persistent homology.
- **Mémoli, F. (2011).** *Gromov-Wasserstein distances and the metric approach to object matching.* Foundations of Computational Mathematics, 11(4), 417-487.
  - *Significance:* Seminal paper formulating the Gromov-Wasserstein discrepancy for invariant shape and graph matching.
- **Bonneel, N., et al. (2015).** *Sliced and radon wasserstein barycenters of measures.* Journal of Mathematical Imaging and Vision, 51(1), 22-45.
  - *Significance:* Established analytical properties and computational efficiency of Sliced Wasserstein projections.

### Geometry, Cryptography, Information Theory & Causal Discovery (Module Z)
- **Graham, R. L. (1972).** *An efficient algorithm for determining the convex hull of a finite planar set.* Information Processing Letters, 1(4), 132-133.
  - *Significance:* Introduced the $O(N \log N)$ Graham scan convex hull algorithm via polar angle sorting and stack backtracking.
- **Bowyer, A. (1981).** *Computing Dirichlet tessellations.* The Computer Journal, 24(2), 162-166. / **Watson, D. F. (1981).** *Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes.* The Computer Journal, 24(2), 167-172.
  - *Significance:* Concurrently introduced the Bowyer-Watson incremental algorithm for Delaunay triangulation based on empty circumcircle insertion.
- **Schnorr, C. P. (1991).** *Efficient signature generation by smart cards.* Journal of Cryptology, 4(3), 161-174.
  - *Significance:* Introduced the Schnorr identification protocol and digital signature scheme, the foundational $\Sigma$-protocol for zero-knowledge proofs.
- **Goldwasser, S., Micali, S., & Rackoff, C. (1989).** *The knowledge complexity of interactive proof systems.* SIAM Journal on Computing, 18(1), 186-208.
  - *Significance:* Gödel Prize winning paper defining Zero-Knowledge Proofs, interactive proof systems, and computational indistinguishability.
- **Koblitz, N. (1987).** *Elliptic curve cryptosystems.* Mathematics of Computation, 48(177), 203-209. / **Miller, V. S. (1985).** *Use of elliptic curves in cryptography.* CRYPTO '85.
  - *Significance:* Independently introduced Elliptic Curve Cryptography (ECC) based on the hardness of the elliptic curve discrete logarithm problem.
- **Merkle, R. C. (1987).** *A digital signature based on a conventional encryption function.* CRYPTO '87.
  - *Significance:* Introduced Merkle trees and cryptographic hash tree proofs of inclusion.
- **Huffman, D. A. (1952).** *A method for the construction of minimum-redundancy codes.* Proceedings of the IRE, 40(9), 1098-1101.
  - *Significance:* Formulated the optimal prefix-free variable-length binary tree code achieving the Shannon entropy bound.
- **Welch, T. A. (1984).** *A technique for high-performance data compression (LZW).* Computer, 17(6), 8-19.
  - *Significance:* Designed the Lempel-Ziv-Welch universal dictionary compression algorithm powering GIF, compress, and modern tokenizers.
- **Cilibrasi, R., & Vitányi, P. M. (2005).** *Clustering by compression.* IEEE Transactions on Information Theory, 51(4), 1523-1545.
  - *Significance:* Formulated Normalized Compression Distance (NCD) as a parameter-free metric for clustering based on Kolmogorov complexity.
- **Spirtes, P., Glymour, C., & Scheines, R. (1993).** *Causation, Prediction, and Search.* MIT Press / Springer-Verlag.
  - *Significance:* The seminal treatise introducing the PC Algorithm for constraint-based causal discovery and Markov equivalence classes from observational data.


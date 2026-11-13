# J06: Continual Learning & Elastic Weight Consolidation (EWC)

## 1. The Beginner's Analogy
When a human learns to ride a bicycle, they do not immediately forget how to walk or write. The brain consolidates important synapses, making them chemically resistant to future rewiring.
In contrast, standard neural networks suffer from **Catastrophic Forgetting**: when trained on a new task, gradient descent overwrites the weights learned for prior tasks. **Elastic Weight Consolidation (EWC)** acts like cognitive memory glue. It anchors parameters that were crucial for past tasks with quadratic spring penalties, allowing uncritical parameters to bend freely to learn new knowledge.

## 2. The Deep Architecture & Mechanics
EWC adopts a Bayesian perspective on sequential task learning:
1. **Task A Posterior:** After training on Task A, the parameter posterior is approximated as a Gaussian distribution centered at optimal parameters $\theta_A^*$ with precision given by the diagonal of the **Fisher Information Matrix $F$**.
2. **Laplace Approximation:** The curvature of the log-likelihood surface around $\theta_A^*$ indicates parameter sensitivity: steep directions (high Fisher values) are vital for Task A; flat directions (low Fisher values) can be altered without degrading Task A performance.
3. **Consolidated Objective:** When training on Task B, the loss combines Task B empirical risk with quadratic penalties weighted by the Fisher diagonal.

## 3. The Hardcore Mathematics

### Bayesian Derivation
From Bayes' rule, the log-posterior given tasks $\mathcal{D}_A$ and $\mathcal{D}_B$ decomposes as:
$$\ln p(\theta \mid \mathcal{D}_A, \mathcal{D}_B) = \ln p(\mathcal{D}_B \mid \theta) + \ln p(\theta \mid \mathcal{D}_A) - \ln p(\mathcal{D}_B)$$
Approximating $\ln p(\theta \mid \mathcal{D}_A)$ with a 2nd-order Taylor expansion around $\theta_A^*$:
$$\ln p(\theta \mid \mathcal{D}_A) \approx \ln p(\theta_A^* \mid \mathcal{D}_A) - \frac{1}{2} (\theta - \theta_A^*)^T F (\theta - \theta_A^*)$$

### The EWC Loss Function
Minimizing negative log-posterior yields the consolidated loss:
$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_{ii} (\theta_i - \theta_{A,i}^*)^2$$
Where:
- $\mathcal{L}_B(\theta)$: Standard task loss on Task B.
- $\lambda$: Regularization strength balancing old vs. new task performance.
- $\theta_{A,i}^*$: Optimal parameter value for Task A.
- $F_{ii}$: Diagonal entry of the empirical Fisher Information Matrix:
  $$F_{ii} = \frac{1}{N} \sum_{n=1}^N \left( \frac{\partial \ln p(y_n \mid x_n, \theta)}{\partial \theta_i} \right)^2$$

## 4. The Algorithm Step-by-Step
1. **Train Task A:** Train base model to convergence on Task A; record optimal parameters $\theta_A^*$.
2. **Compute Fisher Matrix:** Run a forward/backward pass over Task A data to compute squared gradients per parameter $F_{ii} = \mathbb{E}[(\nabla_{\theta_i} \log p)^2]$.
3. **Train Task B with EWC:** For each batch of Task B data:
   - Compute standard Task B gradients $\nabla_\theta \mathcal{L}_B$.
   - Add EWC penalty gradient $\lambda F_{ii}(\theta_i - \theta_{A,i}^*)$.
   - Update weights via gradient descent.
4. **Evaluate:** Verify that model retains high accuracy on Task A while achieving high accuracy on Task B.

## 5. Limitations & Failure Modes
- **Diagonal Fisher Approximation:** Omits off-diagonal covariance between parameters, underestimating joint sensitivity along correlated weight directions (mitigated by K-FAC).
- **Capacity Saturation:** As the number of tasks grows, more parameters become anchored, reducing plasticity for learning downstream tasks.

## 6. Where This Fits
Cross-reference: Continual learning addresses catastrophic forgetting during domain-specific fine-tuning (SFT) of foundation models and continual pre-training of LLMs on newly published corpora.

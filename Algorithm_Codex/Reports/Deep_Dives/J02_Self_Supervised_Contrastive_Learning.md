# J02: Self-Supervised Contrastive Learning (InfoNCE / SimCLR)

## 1. The Beginner's Analogy
Imagine showing someone two photos of the same golden retriever—one taken in bright sunlight, the other in the rain with a red collar. Even with dramatic pixel differences, you tell them: *"These two depict the same dog."* At the same time, you show thousands of photos of cats, cars, and landscapes and say: *"These are not this dog."*
**Self-Supervised Contrastive Learning** works on this principle. Without a single human label, it pulls different augmented views of the same instance together in a high-dimensional embedding space while pushing representations of distinct instances apart.

## 2. The Deep Architecture & Mechanics
Modern contrastive learning frameworks (SimCLR, MoCo) operate via four coupled stages:
1. **Stochastic Data Augmentations:** Given an unlabeled sample $x$, sample two independent stochastic transformations $t_1, t_2 \sim \mathcal{T}$ to yield positive views $\tilde{x}_i, \tilde{x}_j$.
2. **Base Encoder $f(\cdot)$:** Maps raw high-dimensional inputs into representation space $h_i = f(\tilde{x}_i)$.
3. **Projection Head $g(\cdot)$:** A small non-linear MLP mapping representations to a low-dimensional hypersphere $z_i = g(h_i) / \|g(h_i)\|_2$.
4. **Contrastive Loss Function (InfoNCE):** Evaluates pairwise cosine similarities with temperature parameter $\tau$.

## 3. The Hardcore Mathematics

### The InfoNCE Objective
For a minibatch of $N$ examples yielding $2N$ augmented views, the contrastive loss for positive pair $(i, j)$ is:
$$\mathcal{L}_{i, j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \ne i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$
Where $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ is cosine similarity and $\tau > 0$ is the temperature scalar.

### Mutual Information Lower Bound
InfoNCE provides a variational lower bound on the mutual information $I(X; Y)$ between augmented views:
$$I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$$
Where $K$ is the number of negative distractors. Minimizing InfoNCE directly maximizes mutual information between representations while discarding nuisance variations.

### Hand-Derived Analytic Gradients
Let $S_{ik} = \frac{z_i^T z_k}{\tau}$ and $P_{ik} = \frac{\exp(S_{ik})}{\sum_m \exp(S_{im})}$. The gradient with respect to projection embedding $z_i$ is:
$$\nabla_{z_i} \mathcal{L} = \frac{1}{\tau} \sum_k (P_{ik} - \delta_{ik}) z_k$$
Through unit hypersphere normalization $\tilde{z} = z / \|z\|_2$:
$$\frac{\partial \mathcal{L}}{\partial z} = \frac{1}{\|z\|_2} \left( I - \tilde{z} \tilde{z}^T \right) \frac{\partial \mathcal{L}}{\partial \tilde{z}}$$

## 4. The Algorithm Step-by-Step
1. **Sample Minibatch:** Draw $N$ unlabeled instances $\{x_1, \dots, x_N\}$.
2. **Augment Views:** Apply stochastic transformations to generate view pairs $X_a, X_b$.
3. **Forward Pass:** Compute normalized embeddings $Z_a = g(f(X_a))$ and $Z_b = g(f(X_b))$.
4. **Evaluate Similarity Matrix:** Compute scaled Gram matrix $S = \frac{1}{\tau} Z_a Z_b^T$.
5. **Compute InfoNCE Loss:** Evaluate negative log-softmax over matching diagonal indices.
6. **Backpropagation:** Propagate analytic gradients through normalization, projection MLP, and encoder weights.
7. **Iterate:** Update parameters via gradient descent.

## 5. Limitations & Failure Modes
- **Representation Collapse:** If negative pairs are omitted, the network trivially collapses to a constant embedding ($z_i = c$).
- **Batch Size Dependency:** InfoNCE performance scales with the number of negative distractors $K$; small batch sizes provide weak contrastive gradients unless memory banks (MoCo) are utilized.

## 6. Where This Fits
Cross-reference: InfoNCE is the exact loss function that trains OpenAI CLIP, modern dense text embedding models (e.g., text-embedding-3, BGE), and multimodal vision-language projectors.

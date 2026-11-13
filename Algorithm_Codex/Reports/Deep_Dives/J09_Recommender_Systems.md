# J09: Recommender Systems (Collaborative Filtering & Matrix Factorization)

## 1. The Beginner's Analogy
When you visit a streaming service like Netflix, you have only watched a fraction of the 10,000 available movies. How does the system predict whether you will enjoy an obscure sci-fi movie you've never heard of?
**Matrix Factorization** decomposes complex human taste into hidden "latent factors" (e.g., amount of explosive action, intellectual dialogue, romantic tone). By analyzing which factors you consistently favor and how strongly that sci-fi movie exhibits those exact factors, the algorithm accurately predicts your rating.

## 2. The Deep Architecture & Mechanics
Modern recommender architectures encompass two complementary paradigms:
1. **Neighborhood-Based Collaborative Filtering:**
   - *User-Based:* Finds users with similar historical rating profiles and averages their ratings on target items.
   - *Item-Based:* Measures item-item co-rating similarities using adjusted cosine similarity to recommend items similar to ones you liked.
2. **Model-Based Latent Factor Models (SVD / Matrix Factorization):** Decomposes the sparse rating matrix $R \in \mathbb{R}^{U \times I}$ into low-rank user embedding matrix $P \in \mathbb{R}^{U \times k}$ and item embedding matrix $Q \in \mathbb{R}^{I \times k}$, capturing latent preference affinities.

## 3. The Hardcore Mathematics

### Rating Formulation with Biases
Individual ratings are modeled via global baseline, user bias, item bias, and interaction terms:
$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$
Where:
- $\mu$: Global mean rating across the entire dataset.
- $b_u$: User bias (capturing whether a user is systematically harsh or generous).
- $b_i$: Item bias (capturing whether an item is universally acclaimed or despised).
- $p_u \in \mathbb{R}^k, q_i \in \mathbb{R}^k$: Latent factor vectors.

### Regularized Objective Function
Minimizing regularized squared error over the set of observed rating pairs $\mathcal{K} = \{(u, i) : r_{ui} \ne 0\}$:
$$\min_{b, P, Q} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( b_u^2 + b_i^2 + \|p_u\|_2^2 + \|q_i\|_2^2 \right)$$

### Stochastic Gradient Descent Updates
For each observed rating $r_{ui}$, compute prediction error $e_{ui} = r_{ui} - \hat{r}_{ui}$:
$$b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)$$
$$b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)$$
$$p_u \leftarrow p_u + \gamma (e_{ui} q_i - \lambda p_u)$$
$$q_i \leftarrow q_i + \gamma (e_{ui} p_u - \lambda q_i)$$

## 4. The Algorithm Step-by-Step
1. **Initialize:** Compute global mean $\mu$. Initialize bias vectors $b_u = 0, b_i = 0$ and latent matrices $P, Q \sim \mathcal{N}(0, 0.01)$.
2. **Iterate Epochs:** Shuffle observed rating tuples $(u, i, r_{ui})$.
3. **SGD Step:** For each rating, compute error $e_{ui}$ and update parameters.
4. **Predict Unobserved:** Evaluate $\hat{r}_{uj} = \mu + b_u + b_j + p_u^T q_j$ for unrated items.
5. **Top-$K$ Recommendation:** Rank unrated items by predicted rating and serve top $K$.

## 5. Limitations & Failure Modes
- **Cold-Start Problem:** Cannot recommend for new users or items with zero historical ratings (requires hybrid content-based filtering).
- **Popularity Bias:** Popular items dominate recommendations, suppressing long-tail niche items.

## 6. Where This Fits
Cross-reference: Modern Two-Tower neural retrieval systems (e.g. YouTube candidate generation) and Bradley-Terry preference modeling in RLHF are direct mathematical descendants of matrix factorization.

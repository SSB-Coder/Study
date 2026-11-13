# J11: Active Learning (Uncertainty Sampling & Query-by-Committee)

## 1. The Beginner's Analogy
Imagine a student preparing for a medical licensing exam by answering practice questions. An inefficient student randomly flips through a 5,000-page encyclopedia, studying basic facts they already know 90% of the time.
A smart student searches specifically for questions where they are most confused (where their confidence is 50/50), or flags questions where different medical textbooks disagree with each other.
**Active Learning** is this smart student. Instead of passively accepting randomly labeled data, the model inspects an unlabeled pool and asks an expert human annotator to label only the most informative, boundary-adjacent examples.

## 2. The Deep Architecture & Mechanics
Pool-based active learning operates via iterative selection cycles:
1. **Unlabeled Pool $\mathcal{U}$:** A vast collection of raw, uncurated data points.
2. **Labeled Training Set $\mathcal{L}$:** A small initial seed of labeled instances used to train a base model $\mathcal{M}$.
3. **Acquisition Function:** An informativeness scoring metric evaluating unlabeled points:
   - *Least Confidence:* Queries points where the model's top predicted probability is lowest.
   - *Margin Sampling:* Queries points where the difference between top two class probabilities is smallest.
   - *Entropy Sampling:* Queries points with maximum predictive Shannon entropy.
   - *Query-by-Committee (QBC):* An ensemble of models votes; points with maximum disagreement are labeled.
4. **Oracle Annotation:** Selected query batch is annotated by a human expert and transferred from $\mathcal{U}$ to $\mathcal{L}$.

## 3. The Hardcore Mathematics

### Uncertainty Sampling Metrics
For model prediction vector $P(y = c \mid x)$:
1. **Least Confidence:**
   $$x_{\text{LC}} = \arg\max_x \left( 1 - \max_c P(y = c \mid x) \right)$$
2. **Smallest Margin:**
   $$x_{\text{Margin}} = \arg\min_x \left( P(y_{(1)} \mid x) - P(y_{(2)} \mid x) \right)$$
3. **Predictive Entropy:**
   $$x_{\text{Entropy}} = \arg\max_x \left( -\sum_{c=1}^C P(y = c \mid x) \log_2 P(y = c \mid x) \right)$$

### Query-by-Committee Disagreement
For a committee of $K$ distinct models $\{\mathcal{M}_1, \dots, \mathcal{M}_K\}$, let $V(c, x) = \sum_{k=1}^K \mathbb{I}[\mathcal{M}_k(x) = c]$ be the vote count for class $c$:
$$x_{\text{QBC}} = \arg\max_x \left( -\sum_{c=1}^C \frac{V(c, x)}{K} \ln \frac{V(c, x)}{K} \right)$$

## 4. The Algorithm Step-by-Step
1. **Seed Initialization:** Train model $\mathcal{M}$ on small random labeled dataset $\mathcal{L}$.
2. **Score Pool:** Evaluate acquisition function on all $x \in \mathcal{U}$.
3. **Query Batch:** Select top-$B$ most informative instances $Q = \arg\max_{x \in \mathcal{U}} U(x)$.
4. **Human Labeling:** Query the Oracle to label $Q$; update $\mathcal{L} \leftarrow \mathcal{L} \cup Q$ and $\mathcal{U} \leftarrow \mathcal{U} \setminus Q$.
5. **Retrain:** Retrain $\mathcal{M}$ on updated $\mathcal{L}$.
6. **Repeat:** Continue until labeling budget is exhausted or target accuracy is met.

## 5. Limitations & Failure Modes
- **Sampling Bias toward Outliers:** Uncertainty sampling easily mistakes extreme noisy outliers for decision boundary instances.
- **Model Confirmation Bias:** If the initial seed misidentifies a whole cluster, active learning may never query that region.

## 6. Where This Fits
Cross-reference: Active learning governs human annotation pipelines in RLHF, prompt selection for Reinforcement Learning from AI Feedback (RLAIF), and fine-tuning curation for frontier LLMs.

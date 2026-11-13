# Deep Dive: Naive Bayes (Gaussian, Multinomial, Bernoulli)

## 1. The Beginner's Analogy: The Spam Filter with Blindfolds
Imagine a mailroom clerk sorting incoming emails into "Spam" or "Legitimate":
- The clerk has a dictionary of suspicious words: "Viagra", "Inheritance", "Crypto", "Urgent", "Meeting".
- **The "Naive" Assumption:** The clerk assumes every word in the email appears completely independently of every other word. They pretend that seeing the word "Prince" has zero correlation with seeing "Nigerian", calculating the odds by simply multiplying each word's individual probability together: $P(\text{Prince}|\text{Spam}) \times P(\text{Nigerian}|\text{Spam})$.
- Even though this assumption of conditional independence is linguistically absurd, the resulting odds ratio is so stark that Naive Bayes remains an extraordinarily effective, blazingly fast baseline classifier.

---

## 2. The Deep Architecture/Algorithm Breakdown
Naive Bayes is a generative probabilistic classifier applying Bayes' Theorem under the conditional independence assumption:
1. **Bayes' Theorem:**
   $$P(Y=c \mid X) = \frac{P(Y=c) P(X \mid Y=c)}{P(X)} \propto P(Y=c) \prod_{j=1}^d P(X_j \mid Y=c)$$
2. **Prior Probability $P(Y=c)$:** Empirical class prevalence $\frac{N_c}{N}$.
3. **Likelihood Models:**
   - **Gaussian Naive Bayes:** Models continuous features using univariate Gaussians $\mathcal{N}(\mu_{cj}, \sigma_{cj}^2)$.
   - **Multinomial Naive Bayes:** Models feature counts (e.g. word frequencies in documents) using categorical multinomial parameters $\theta_{cj}$.
   - **Bernoulli Naive Bayes:** Models binary word presence/absence $x_j \in \{0, 1\}$.
4. **Laplace (Additive) Smoothing:** Prevents zero-probability multiplication when an unseen token appears in test data.

---

## 3. The Hardcore Mathematics

### 3.1 Decision Rule in Log-Space
To prevent floating-point underflow when multiplying hundreds of probabilities:
$$\hat{y} = \arg\max_{c \in \{1, \dots, K\}} \left[ \ln P(Y=c) + \sum_{j=1}^d \ln P(X_j \mid Y=c) \right]$$

### 3.2 Gaussian Variant
For continuous $x_j \in \mathbb{R}$:
$$P(X_j = x_j \mid Y=c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)$$
$$\ln P(X_j = x_j \mid Y=c) = -\frac{1}{2} \ln(2\pi \sigma_{cj}^2) - \frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}$$

### 3.3 Multinomial Variant with Laplace Smoothing
For integer counts $x_j \in \mathbb{N}$:
Let $N_{cj} = \sum_{i: y_i=c} x_{ij}$ be the total count of feature $j$ across all instances of class $c$, and $N_c = \sum_{j=1}^d N_{cj}$.
With additive smoothing parameter $\alpha > 0$ (typically $\alpha = 1$ for Laplace smoothing):
$$\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha \cdot d}$$
Log-likelihood for query count vector $x$:
$$\sum_{j=1}^d x_j \ln \theta_{cj}$$

### 3.4 Bernoulli Variant
For binary features $x_j \in \{0, 1\}$:
$$P(X \mid Y=c) = \prod_{j=1}^d \theta_{cj}^{x_j} (1 - \theta_{cj})^{1 - x_j}$$
Where $\theta_{cj} = \frac{\sum_{i: y_i=c} x_{ij} + \alpha}{N_c + 2\alpha}$.

---

## 4. The Algorithm Step-by-Step
1. **Compute Class Priors:** $\pi_c = \frac{N_c}{N}$ for each class $c \in \{1, \dots, K\}$.
2. **Estimate Class-Conditional Parameters:**
   - For Gaussian: compute sample mean $\mu_{cj}$ and sample variance $\sigma_{cj}^2$ for each feature.
   - For Multinomial: sum counts $N_{cj}$ and compute smoothed probabilities $\theta_{cj}$.
   - For Bernoulli: count positive binary occurrences and apply smoothing.
3. **Log-Posterior Evaluation:** For each test instance $x$, evaluate the log-prior plus sum of log-likelihoods across all classes.
4. **Prediction:** Output $\arg\max_c \text{score}(c)$.

---

## 5. Limitations / When It Breaks
- **Feature Correlation:** When features are strongly correlated (e.g. "Hong" and "Kong"), Naive Bayes double-counts evidence, over-confidently skewing posterior probabilities toward extreme 0 or 1.
- **Continuous Distribution Mismatch:** Real-world continuous features are frequently multi-modal or heavy-tailed, violating the unimodal Gaussian assumption.
- **Historical Fix:** Tree ensembles (Random Forests) and Generalized Additive Models handle arbitrary feature correlations.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/29_Tokenization_and_Data_Curation.md`. The unigram language modeling tokenization algorithm in SentencePiece directly evaluates Dirichlet-Multinomial marginal likelihoods to prune vocabulary subwords.

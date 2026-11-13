# J08: Anomaly Detection (Isolation Forest)

## 1. The Beginner's Analogy
Imagine playing a guessing game to isolate a specific person in a crowd. If you want to isolate a person dressed in a neon dinosaur suit standing far away on a hill, you only need one question: *"Are they wearing a dinosaur suit?"*
Conversely, if you want to isolate an average student sitting in the middle of a lecture hall of 500 students, you need dozens of detailed questions about eye color, height, and seating row.
**Isolation Forest** isolates anomalies instead of profiling normal points. Because anomalies are "few and different," they are isolated in very few random binary splits near the root of the tree.

## 2. The Deep Architecture & Mechanics
Isolation Forest avoids density estimation or distance matrices, scaling in linear time $\mathcal{O}(n)$:
1. **Isolation Trees (iTrees):** Proper binary trees constructed by recursively choosing a random feature and a uniform random split threshold between the feature's minimum and maximum observed values.
2. **Path Length $h(x)$:** The number of edges traversed from the tree root to a terminating leaf node. Anomalous points consistently exhibit short path lengths.
3. **Subsampling:** Trees are trained on small random subsamples ($n = 256$), which prevents masking (neighboring anomalies concealing each other) and swamping (normal points falsely isolated due to dense clusters).

## 3. The Hardcore Mathematics

### Average Path Length Normalizer
The average path length of an unsuccessful search in a Binary Search Tree (BST) over $n$ items serves as the baseline:
$$c(n) = 2 \ln(n - 1) + 2\gamma - \frac{2(n - 1)}{n}$$
Where $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant.

### Anomaly Score
For instance $x$ across an ensemble of $m$ isolation trees:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
Behavioral interpretation:
- If $\mathbb{E}[h(x)] \to 0$, $s \to 1$: **Definitive Anomaly**.
- If $\mathbb{E}[h(x)] \to c(n)$, $s \to 0.5$: **Indistinguishable from Randomness / Normal Inlier**.
- If $\mathbb{E}[h(x)] \to n - 1$, $s \to 0$: **Centrally Clustered Inlier**.

## 4. The Algorithm Step-by-Step
1. **Subsample:** Draw $m$ independent sub-datasets of size $\psi = 256$ without replacement.
2. **Build iTree:**
   - Terminate if tree depth exceeds $\lceil \log_2 \psi \rceil$ or sample count $\le 1$.
   - Select random feature $q \in \{1, \dots, d\}$.
   - Sample split threshold $p \sim \text{Uniform}(\min(X_{*, q}), \max(X_{*, q}))$.
   - Partition into left ($X_q < p$) and right ($X_q \ge p$) children; recurse.
3. **Inference:** For query point $x$, traverse each tree to calculate path length $h_k(x)$.
4. **Aggregate Score:** Compute mean path length $\bar{h}(x)$ and evaluate anomaly score $s(x, \psi)$.

## 5. Limitations & Failure Modes
- **Axis-Aligned Cuts:** Random axis-aligned cuts struggle when anomalies lie in dense corridors or complex non-linear manifolds (addressed by Extended Isolation Forest with random hyperplanes).
- **Curse of Irrelevant Features:** If data contains hundreds of pure noise dimensions, random feature selection frequently splits on uninformative features.

## 6. Where This Fits
Cross-reference: Anomaly detection underpins out-of-distribution (OOD) input detection, data cleaning in LLM pre-training corpora, and detecting jailbreak prompt injections in production AI safety gateways.

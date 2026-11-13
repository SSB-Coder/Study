# M02: Locality-Sensitive Hashing (LSH: SimHash & MinHash)

## 1. The Beginner's Analogy
Standard cryptographic hash functions (like SHA-256 or MD5) are deliberately designed to be chaotic: changing even a single comma in a 500-page book flips 50% of the output bits (the avalanche effect).
**Locality-Sensitive Hashing (LSH)** flips this philosophy upside-down. 
Instead of scrambling inputs, LSH ensures that **similar items hash into the exact same bucket with high probability**, while dissimilar items land in different buckets. 
- If two web articles share 90% of their text, LSH assigns them identical hash signatures.
- You can find near-duplicate documents or similar image embeddings across billions of records simply by looking inside a few hash buckets in $O(1)$ time, without computing billions of pairwise comparisons.

## 2. The Deep Architecture & Mechanics
LSH operates under two primary similarity metrics:
1. **Cosine / Angular Distance via Random Projections (SimHash):**
   - Draws random Gaussian hyperplanes $r \sim \mathcal{N}(0, I_D)$.
   - Hashes vector $x$ to $+1$ or $-1$ depending on which side of the hyperplane $x$ lies: $h_r(x) = \text{sign}(r^T x)$.
2. **Jaccard Similarity via MinHash:**
   - Evaluates overlap between sparse binary feature sets or n-gram shingles: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$.
   - Applies random permutations $\pi$ to the universe of elements; the hash value is the index of the first non-zero element under permutation.

### S-Curve Amplification via Banding
A single hash function yields weak discriminative power. By combining $k = b \cdot r$ hash functions into $b$ **bands** of $r$ rows:
- Two items become candidate neighbors if they match on **all** $r$ rows in **at least one** band.
- This creates a sharp sigmoidal "S-curve" filter that accepts items with similarity $s > s^*$ and rejects items with $s < s^*$.

## 3. The Hardcore Mathematics

### Random Projection (SimHash) Collision Probability
Let vectors $u, v \in \mathbb{R}^D$ form an angle $\theta(u, v) = \arccos\left(\frac{u^T v}{\|u\|_2 \|v\|_2}\right)$.
A random hyperplane with normal vector $r \sim \mathcal{N}(0, I_D)$ separates $u$ and $v$ with probability proportional to the dihedral angle between them:
$$P(h_r(u) \ne h_r(v)) = \frac{\theta(u, v)}{\pi}$$
Consequently, the collision probability is:
$$P(h_r(u) = h_r(v)) = 1 - \frac{\theta(u, v)}{\pi}$$

### MinHash Collision Theorem
For any two non-empty sets $A, B \subseteq U$ and a uniform random permutation $\pi: U \to U$:
$$P\left(\min_{a \in A} \pi(a) = \min_{b \in B} \pi(b)\right) = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

### Banding S-Curve Amplification
Let $s = \text{sim}(u, v) \in [0, 1]$ be the base collision probability of a single hash function.
Partition $k$ hash outputs into $b$ bands of $r$ rows each ($k = b \cdot r$).
1. Probability of matching on all $r$ hashes within a single band: $s^r$.
2. Probability of not matching within that band: $1 - s^r$.
3. Probability of not matching in any of the $b$ independent bands: $(1 - s^r)^b$.
4. **Candidate Probability (Collision in at least one band):**
   $$P_{\text{cand}}(s) = 1 - \left(1 - s^r\right)^b$$
The threshold inflection point $s^*$ where the derivative is maximal satisfies:
$$s^* \approx \left(\frac{1}{b}\right)^{1/r}$$

## 4. The Algorithm Step-by-Step
1. **Random Projection LSH Table Construction:**
   - Sample $L$ independent hash tables, each with $k$ random projection hyperplanes $R_l \in \mathbb{R}^{k \times D}$.
   - For each database vector $x$: compute bit vector $b = (\text{sign}(R_l x) > 0)$, convert $b$ to integer bucket ID, and append $x$ to `table[l][bucket_id]`.
2. **Querying:**
   - For query $q$, compute bucket ID across all $L$ tables.
   - Aggregate candidate vectors from the union of these buckets.
   - Compute exact cosine distances only on the candidate pool and return the top-$k$.

## 5. Limitations & Failure Modes
- **False Negatives:** Because LSH is probabilistic, true nearest neighbors may fail to collide in all bands with probability $(1 - s^r)^b$.
- **High Memory for High Recall:** Achieving recall $>98\%$ often requires dozens of tables ($L \ge 64$), multiplying memory requirements.
- **Fixed Metric Coupling:** A specific hash family only supports its dedicated metric space (e.g., Random Projections $\leftrightarrow$ Angular distance, MinHash $\leftrightarrow$ Jaccard, $p$-stable distributions $\leftrightarrow$ Euclidean $L_2$).

## 6. Where This Fits
Cross-reference: LSH provides $O(1)$ expected-time candidate retrieval for ultra-large-scale deduplication, web crawling, and approximate nearest neighbor search, serving as the fast hash-based alternative to graph-based HNSW (M01).

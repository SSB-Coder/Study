# K06: Classical Word Embeddings (Word2Vec Skip-Gram)

## 1. The Beginner's Analogy
In early computer science, words were represented as distinct, isolated symbols—like one-hot vectors with 50,000 zeros and a single one. To a computer, the symbol for "cat" had no closer relationship to "kitten" than it did to "bulldozer" or "refrigerator."
**Word2Vec** embodies linguist J.R. Firth's famous 1957 maxim: *"You shall know a word by the company it keeps."* By analyzing which words frequently co-occur in the same sentences across billions of documents, Word2Vec maps words into dense geometric vectors where semantic concepts become spatial directions (e.g. $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$).

## 2. The Deep Architecture & Mechanics
Word2Vec operates under two dual architectures:
1. **Continuous Bag of Words (CBOW):** Predicts the center target word given context words.
2. **Skip-Gram with Negative Sampling (SGNS):** Given a center word $w_t$, predicts the context words within a sliding window $[w_{t-c}, \dots, w_{t+c}]$.
3. **Negative Sampling:** Computing the full softmax denominator over 100,000 words is computationally intractable $\mathcal{O}(|V|)$. Negative sampling transforms multiclass classification into multiple independent binary logistic regressions, differentiating true context words from $k$ noise words drawn from a unigram power distribution $P_n(w) \propto f(w)^{3/4}$.

## 3. The Hardcore Mathematics

### Skip-Gram Negative Sampling Objective
For center word $w_t$ and observed context word $w_c$, with $k$ noise words drawn from $P_n(w)$:
$$\mathcal{L} = \ln \sigma(u_{w_c}^T v_{w_t}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \ln \sigma(-u_{w_i}^T v_{w_t}) \right]$$
Where:
- $v_w \in \mathbb{R}^d$: Input center embedding vector.
- $u_w \in \mathbb{R}^d$: Output context embedding vector.
- $\sigma(z) = \frac{1}{1 + \exp(-z)}$: Standard logistic sigmoid.

### Explicit Analytic Gradient Derivations
Using the identity $\frac{d}{dz} \ln \sigma(z) = 1 - \sigma(z)$ and $\frac{d}{dz} \ln \sigma(-z) = -\sigma(z)$:
$$\frac{\partial \mathcal{L}}{\partial u_{w_c}} = (1 - \sigma(u_{w_c}^T v_{w_t})) v_{w_t}$$
For negative sample $w_i$:
$$\frac{\partial \mathcal{L}}{\partial u_{w_i}} = -\sigma(u_{w_i}^T v_{w_t}) v_{w_t}$$
For the center word vector $v_{w_t}$:
$$\frac{\partial \mathcal{L}}{\partial v_{w_t}} = (1 - \sigma(u_{w_c}^T v_{w_t})) u_{w_c} - \sum_{i=1}^k \sigma(u_{w_i}^T v_{w_t}) u_{w_i}$$

## 4. The Algorithm Step-by-Step
1. **Build Vocabulary:** Tokenize corpus; compute word frequency table; construct smoothed negative sampling distribution table $P_n(w) \propto f(w)^{0.75}$.
2. **Initialize Embeddings:** Initialize input matrix $W \in \mathbb{R}^{|V| \times d}$ and output matrix $W' \in \mathbb{R}^{|V| \times d}$ with small random Gaussian values.
3. **Stream Sliding Window:** For each center word $w_t$, extract surrounding context words $w_c$ within window size $C$.
4. **Draw Noise Samples:** Sample $k$ negative word IDs from unigram table.
5. **SGD Update:** Apply analytic gradients to update $v_{w_t}$, $u_{w_c}$, and $\{u_{w_i}\}$.

## 5. Limitations & Failure Modes
- **Polysemy:** Word2Vec assigns a single static vector to each word, conflating distinct word senses (e.g. "bank" of a river vs. financial "bank").
- **Out-of-Vocabulary (OOV):** Cannot represent unseen words or typos at test time (addressed by subword models like FastText and BPE).

## 6. Where This Fits
Cross-reference: Word2Vec is the historical ancestor of modern embedding lookup tables that every Transformer begins with. Contrastive representation learning in CLIP and LLM token embeddings derive directly from Word2Vec's vector geometry.

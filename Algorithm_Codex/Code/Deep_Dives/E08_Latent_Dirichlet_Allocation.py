"""
E08. Latent Dirichlet Allocation (LDA Topic Modeling via Collapsed Gibbs)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LDATopicModelScratch:
    """Latent Dirichlet Allocation using Collapsed Gibbs Sampling."""
    def __init__(self, n_topics: int = 2, alpha: float = 0.5, beta: float = 0.1, n_iters: int = 40):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters
        self.vocab = {}
        self.inv_vocab = {}
        self.theta_ = None # Document-Topic distributions
        self.phi_ = None   # Topic-Word distributions

    def fit(self, documents: list):
        # 1. Build Vocabulary
        self.vocab = {}
        corpus = []
        for doc in documents:
            doc_indices = []
            for w in doc:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                doc_indices.append(self.vocab[w])
            corpus.append(np.array(doc_indices, dtype=int))
        
        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        V = len(self.vocab)
        D = len(corpus)

        # 2. Initialize count matrices
        # n_dk: count of words in doc d assigned to topic k
        n_dk = np.zeros((D, self.K), dtype=int)
        # n_kv: count of word v assigned to topic k across corpus
        n_kv = np.zeros((self.K, V), dtype=int)
        # n_k: total words assigned to topic k
        n_k = np.zeros(self.K, dtype=int)

        # Initial random topic assignments z_di
        z = []
        for d in range(D):
            z_d = []
            for v in corpus[d]:
                k_rand = np.random.choice(self.K)
                z_d.append(k_rand)
                n_dk[d, k_rand] += 1
                n_kv[k_rand, v] += 1
                n_k[k_rand] += 1
            z.append(np.array(z_d, dtype=int))

        # 3. Collapsed Gibbs Sampling Loop <- eq. 3.1
        for iteration in range(self.n_iters):
            for d in range(D):
                for i in range(len(corpus[d])):
                    v = corpus[d][i]
                    k_curr = z[d][i]

                    # Decrement counts
                    n_dk[d, k_curr] -= 1
                    n_kv[k_curr, v] -= 1
                    n_k[k_curr] -= 1

                    # Compute conditional distribution: P(z = k) propto (n_dk + alpha) * (n_kv + beta) / (n_k + V*beta)
                    p_k = (n_dk[d] + self.alpha) * (n_kv[:, v] + self.beta) / (n_k + V * self.beta)
                    p_k /= np.sum(p_k)

                    # Sample new topic
                    k_new = np.random.choice(self.K, p=p_k)

                    # Increment counts
                    n_dk[d, k_new] += 1
                    n_kv[k_new, v] += 1
                    n_k[k_new] += 1
                    z[d][i] = k_new

        # 4. Posterior Estimates <- eq. 3.2
        self.theta_ = (n_dk + self.alpha) / np.sum(n_dk + self.alpha, axis=1, keepdims=True)
        self.phi_ = (n_kv + self.beta) / np.sum(n_kv + self.beta, axis=1, keepdims=True)
        return self

if __name__ == "__main__":
    print("--- E08: Latent Dirichlet Allocation Sanity Check ---")
    np.random.seed(42)
    # Synthetic corpus with 2 clear topics:
    # Topic 0: AI/Data (neural, learning, data)
    # Topic 1: Cooking/Food (chef, kitchen, recipe)
    docs = [
        ["neural", "learning", "data", "learning"],
        ["neural", "data", "ai", "neural"],
        ["chef", "kitchen", "recipe", "food"],
        ["kitchen", "recipe", "chef", "recipe"],
        ["neural", "ai", "kitchen", "recipe"] # Mixed doc
    ]

    lda = LDATopicModelScratch(n_topics=2, alpha=0.5, beta=0.1, n_iters=50).fit(docs)
    print(f"Document-Topic Matrix Shape: {lda.theta_.shape}")
    print(f"Topic-Word Matrix Shape:     {lda.phi_.shape}")
    # Doc 0 should have high concentration in one topic, Doc 2 in the other
    top_doc0 = np.argmax(lda.theta_[0])
    top_doc2 = np.argmax(lda.theta_[2])
    print(f"Doc 0 Dominant Topic: {top_doc0} | Doc 2 Dominant Topic: {top_doc2}")
    assert top_doc0 != top_doc2 # Distinct documents must map to distinct topics
    print("E08 Sanity Check: PASSED cleanly.")

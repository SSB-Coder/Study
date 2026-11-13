"""
K06: Classical Word Embeddings (Word2Vec Skip-Gram with Negative Sampling)
Pure NumPy implementation of Mikolov et al. (NeurIPS 2013).
Learns dense semantic word vectors from local co-occurrence statistics.
"""

import numpy as np

class Word2VecSGNS:
    """
    Skip-gram with Negative Sampling (SGNS).
    Maximizes probability of true context words while minimizing probability of noise words.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 8, lr: float = 0.05, n_negatives: int = 4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.n_negatives = n_negatives

        # Center word matrix W, Context word matrix W_prime
        self.W = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_prime = np.random.randn(vocab_size, embed_dim) * 0.1

    def sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0))))

    def train_step(self, center_id: int, context_id: int, unigram_probs: np.ndarray):
        """
        Performs single SGD step on center word and true context word with K negative samples.
        """
        # Draw negative samples
        neg_samples = np.random.choice(self.vocab_size, size=self.n_negatives, p=unigram_probs)

        v_c = self.W[center_id] # (D,)
        u_o = self.W_prime[context_id] # (D,)

        # 1. Positive pair dot product: sigma(u_o^T v_c)
        dot_pos = np.dot(u_o, v_c)
        sig_pos = self.sigmoid(dot_pos)
        grad_pos = sig_pos - 1.0 # dL/ddot = sig - 1

        grad_v_c = grad_pos * u_o
        grad_u_o = grad_pos * v_c

        # 2. Negative pairs: sigma(-u_k^T v_c)
        grad_neg_u = {}
        for neg_id in neg_samples:
            u_k = self.W_prime[neg_id]
            dot_neg = np.dot(u_k, v_c)
            sig_neg = self.sigmoid(dot_neg)
            grad_k = sig_neg # dL/ddot_k = sig_neg

            grad_v_c += grad_k * u_k
            grad_neg_u[neg_id] = grad_neg_u.get(neg_id, np.zeros(self.embed_dim)) + grad_k * v_c

        # 3. Update weights
        self.W[center_id] -= self.lr * grad_v_c
        self.W_prime[context_id] -= self.lr * grad_u_o
        for neg_id, g_u in grad_neg_u.items():
            self.W_prime[neg_id] -= self.lr * g_u

    def cosine_similarity(self, word_a: int, word_b: int) -> float:
        v_a = self.W[word_a]
        v_b = self.W[word_b]
        norm_a = np.linalg.norm(v_a) + 1e-12
        norm_b = np.linalg.norm(v_b) + 1e-12
        return float(np.dot(v_a, v_b) / (norm_a * norm_b))


if __name__ == "__main__":
    print("--- K06: Word2Vec (Skip-Gram with Negative Sampling) Sanity Check ---")
    np.random.seed(42)

    # Toy vocabulary: [king, queen, apple, orange, banana, prince]
    vocab = ["king", "queen", "apple", "orange", "banana", "prince"]
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}

    # Synthetic training co-occurrence pairs:
    # Royalty context: (king, queen), (king, prince), (queen, prince)
    # Fruit context: (apple, orange), (apple, banana), (orange, banana)
    royalty_pairs = [(w2i["king"], w2i["queen"]), (w2i["king"], w2i["prince"]), (w2i["queen"], w2i["prince"])]
    fruit_pairs = [(w2i["apple"], w2i["orange"]), (w2i["apple"], w2i["banana"]), (w2i["orange"], w2i["banana"])]

    all_pairs = royalty_pairs * 20 + fruit_pairs * 20

    # Unigram distribution (uniform for toy demo)
    unigram = np.ones(V) / V

    sgns = Word2VecSGNS(vocab_size=V, embed_dim=8, lr=0.08, n_negatives=3)

    # Train over 150 iterations
    for epoch in range(150):
        np.random.shuffle(all_pairs)
        for center, ctx in all_pairs:
            sgns.train_step(center, ctx, unigram)

    # Check semantic cosine similarities
    sim_king_queen = sgns.cosine_similarity(w2i["king"], w2i["queen"])
    sim_apple_orange = sgns.cosine_similarity(w2i["apple"], w2i["orange"])
    sim_king_apple = sgns.cosine_similarity(w2i["king"], w2i["apple"])

    print(f"Cosine Similarity (king, queen):   {sim_king_queen:.4f}")
    print(f"Cosine Similarity (apple, orange): {sim_apple_orange:.4f}")
    print(f"Cosine Similarity (king, apple):   {sim_king_apple:.4f}")

    assert sim_king_queen > sim_king_apple, "Semantic cluster (king, queen) must be closer than cross-cluster (king, apple)"
    assert sim_apple_orange > sim_king_apple, "Semantic cluster (apple, orange) must be closer than cross-cluster (apple, king)"

    print("K06 Sanity Check: PASSED cleanly.")

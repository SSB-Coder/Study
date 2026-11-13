"""
M02: Locality-Sensitive Hashing (LSH: SimHash & MinHash)
Pure NumPy implementation of sub-linear similarity search via probabilistic hashing:
1. Random Projection LSH (SimHash) for Cosine/Angular similarity
2. MinHash with Banding S-Curve Amplification for Jaccard set similarity
"""

import numpy as np
from typing import List, Dict, Set, Tuple


class RandomProjectionLSH:
    """
    Random Projection Locality-Sensitive Hashing (SimHash) for Cosine Similarity.
    Uses L independent hash tables with k random hyperplanes per table.
    """
    def __init__(self, dim: int, n_bits: int = 8, n_tables: int = 5):
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables
        
        # Hyperplanes: shape (n_tables, n_bits, dim)
        self.hyperplanes = [np.random.randn(n_bits, dim) for _ in range(n_tables)]
        # Hash tables: table_idx -> {bucket_int -> list of (doc_id, vector)}
        self.tables: List[Dict[int, List[Tuple[int, np.ndarray]]]] = [{} for _ in range(n_tables)]

    def _hash(self, table_idx: int, vector: np.ndarray) -> int:
        """Projects vector onto hyperplanes and converts boolean signs into an integer bucket ID."""
        projections = np.dot(self.hyperplanes[table_idx], vector)
        bits = (projections >= 0).astype(int)
        # Convert bit array to integer
        bucket = int("".join(bits.astype(str)), 2)
        return bucket

    def index(self, doc_id: int, vector: np.ndarray) -> None:
        """Indexes a vector across all hash tables."""
        for t in range(self.n_tables):
            bucket = self._hash(t, vector)
            if bucket not in self.tables[t]:
                self.tables[t][bucket] = []
            self.tables[t][bucket].append((doc_id, vector))

    def query(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[float, int]]:
        """
        Retrieves candidate vectors from collided buckets across all tables
        and reranks candidates by exact cosine similarity.
        """
        candidate_ids = set()
        candidates = []

        for t in range(self.n_tables):
            bucket = self._hash(t, query_vec)
            for doc_id, vec in self.tables[t].get(bucket, []):
                if doc_id not in candidate_ids:
                    candidate_ids.add(doc_id)
                    candidates.append((doc_id, vec))

        if not candidates:
            return []

        # Rerank by cosine similarity
        q_norm = np.linalg.norm(query_vec)
        scored = []
        for doc_id, vec in candidates:
            denom = q_norm * np.linalg.norm(vec)
            cos_sim = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
            scored.append((cos_sim, doc_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


class MinHashLSH:
    """
    MinHash with Banding technique for Jaccard set similarity.
    P(collision in band) = 1 - (1 - s^r)^b
    """
    def __init__(self, num_perm: int = 64, num_bands: int = 8):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        
        # Universal hashing coefficients: h(x) = (a * x + b) % prime
        self.prime = 2147483647
        self.a = np.random.randint(1, self.prime - 1, size=num_perm, dtype=np.int64)
        self.b = np.random.randint(0, self.prime - 1, size=num_perm, dtype=np.int64)
        
        # Band tables: band_idx -> {band_hash -> set(doc_id)}
        self.band_tables: List[Dict[int, Set[int]]] = [{} for _ in range(num_bands)]
        self.signatures: Dict[int, np.ndarray] = {}

    def compute_signature(self, token_ids: Set[int]) -> np.ndarray:
        """Computes minhash signature vector of length num_perm."""
        if not token_ids:
            return np.full(self.num_perm, self.prime)
        
        tokens = np.array(list(token_ids), dtype=np.int64)[:, None]
        # Hashes: shape (len(tokens), num_perm)
        hashes = (tokens * self.a + self.b) % self.prime
        sig = np.min(hashes, axis=0)
        return sig

    def index(self, doc_id: int, token_ids: Set[int]) -> None:
        """Indexes tokenized set under banded hash buckets."""
        sig = self.compute_signature(token_ids)
        self.signatures[doc_id] = sig

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = sig[start:end]
            band_hash = hash(tuple(band_chunk))
            
            if band_hash not in self.band_tables[b]:
                self.band_tables[b][band_hash] = set()
            self.band_tables[b][band_hash].add(doc_id)

    def query(self, query_token_ids: Set[int]) -> Set[int]:
        """Returns candidate document IDs colliding in at least one band."""
        sig = self.compute_signature(query_token_ids)
        candidates = set()

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = sig[start:end]
            band_hash = hash(tuple(band_chunk))
            candidates.update(self.band_tables[b].get(band_hash, set()))

        return candidates


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Test Random Projection LSH
    dim = 32
    n_docs = 300
    X = np.random.randn(n_docs, dim)
    lsh = RandomProjectionLSH(dim=dim, n_bits=6, n_tables=8)
    for i in range(n_docs):
        lsh.index(i, X[i])

    # Query with a perturbed version of document 10 (guaranteed high similarity)
    target_idx = 10
    noisy_query = X[target_idx] + np.random.randn(dim) * 0.1
    top_matches = lsh.query(noisy_query, k=5)
    matched_ids = [doc_id for _, doc_id in top_matches]
    print(f"LSH Query Target: {target_idx}, Top Collided Matches: {matched_ids}")
    assert target_idx in matched_ids, f"Expected {target_idx} in LSH candidates"

    # 2. Test MinHash Banding
    minhash = MinHashLSH(num_perm=64, num_bands=8)
    set_a = set(range(100, 200))          # Base document
    set_b = set(range(110, 205))          # High overlap with A (~80% Jaccard)
    set_c = set(range(500, 600))          # Zero overlap with A

    minhash.index(1, set_a)
    minhash.index(2, set_c)

    candidates = minhash.query(set_b)
    print(f"MinHash Candidates for near-duplicate set: {candidates}")
    assert 1 in candidates, "Near-duplicate set A must collide with query B"
    assert 2 not in candidates, "Completely disjoint set C must not collide"

    print("M02 Sanity Check: PASSED cleanly.")

"""
29. Dataset Curation (MinHash LSH for Deduplication)
Deep Dive Implementation

This file demonstrates the exact algorithm used to curate massive datasets 
(like FineWeb or The Pile) before training. 
Training an LLM on duplicate data causes it to memorize rather than generalize.
We use MinHash and Locality-Sensitive Hashing (LSH) to find near-duplicates
in O(N) time instead of O(N^2) time.
"""

import random
import re

# ---------------------------------------------------------------------------
# 1. TEXT NORMALIZATION & SHINGLING
# ---------------------------------------------------------------------------

def normalize_text(text):
    """Lowercases text and removes punctuation for robust matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_shingles(text, k=3):
    """
    Converts text into a set of n-grams (shingles).
    e.g. "the cat sat" (k=2) -> {"the cat", "cat sat"}
    """
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    
    shingles = set()
    for i in range(len(words) - k + 1):
        shingles.add(" ".join(words[i:i+k]))
    return shingles

# ---------------------------------------------------------------------------
# 2. MINHASH ALGORITHM
# ---------------------------------------------------------------------------

def generate_hash_functions(num_hashes, max_val):
    """
    Generates 'num_hashes' random hash functions.
    A hash function is defined as h(x) = (a*x + b) % c
    """
    # A large prime number for the modulo
    c = 4294967311
    
    hash_funcs = []
    for _ in range(num_hashes):
        a = random.randint(1, c - 1)
        b = random.randint(0, c - 1)
        hash_funcs.append(lambda x, a=a, b=b, c=c: (a * hash(x) + b) % c)
    return hash_funcs

def compute_minhash_signature(shingles, hash_funcs):
    """
    Computes the MinHash signature for a set of shingles.
    For each hash function, we hash all shingles and keep the MINIMUM hash value.
    This creates a fixed-length signature representing the document.
    """
    signature = []
    for h in hash_funcs:
        min_hash = float('inf')
        for shingle in shingles:
            val = h(shingle)
            if val < min_hash:
                min_hash = val
        signature.append(min_hash)
    return signature

def jaccard_similarity_approx(sig1, sig2):
    """
    The magic of MinHash: The probability that sig1[i] == sig2[i] is EXACTLY 
    EQUAL to the Jaccard Similarity (overlap) of the original shingle sets.
    """
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# ---------------------------------------------------------------------------
# 3. MOCK CURATION PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- 29. Dataset Curation (MinHash Deduplication) ---")
    
    # 1. The Raw Dataset
    # Notice that Doc 1 and Doc 3 are almost identical.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence will transform the global economy.",
        "The quick brown fox jumps over the lazy sleepy dog.",
        "Deep learning requires massive amounts of curated data."
    ]
    
    print("Raw Corpus:")
    for i, doc in enumerate(corpus):
        print(f"  Doc {i}: {doc}")
        
    num_hashes = 50 # The length of our signature vector
    hash_funcs = generate_hash_functions(num_hashes, 2**32-1)
    
    signatures = []
    
    print("\nProcessing Documents...")
    for i, doc in enumerate(corpus):
        norm = normalize_text(doc)
        shingles = get_shingles(norm, k=3)
        sig = compute_minhash_signature(shingles, hash_funcs)
        signatures.append(sig)
        print(f"  Doc {i} processed into a signature of {len(sig)} integers.")
        
    # 2. Finding Near-Duplicates
    print("\nCalculating Approximate Jaccard Similarity (via MinHash):")
    threshold = 0.7 # 70% similarity means it's a duplicate
    
    duplicates = set()
    
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            sim = jaccard_similarity_approx(signatures[i], signatures[j])
            print(f"  Similarity Doc {i} vs Doc {j}: {sim:.2f}")
            
            if sim >= threshold:
                print(f"    [!] DUPLICATE DETECTED! Flagging Doc {j} for removal.")
                duplicates.add(j)
                
    # 3. The Cleaned Dataset
    print("\nFinal Curated Dataset (Ready for Pre-training):")
    for i, doc in enumerate(corpus):
        if i not in duplicates:
            print(f"  Included: {doc}")

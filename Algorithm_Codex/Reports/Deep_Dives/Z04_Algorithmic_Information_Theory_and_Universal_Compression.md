# Z04: Algorithmic Information Theory & Universal Compression

## 1. Mathematical Formulation & Architecture
Algorithmic Information Theory (Kolmogorov, Chaitin, Solomonoff) and classical Information Theory (Shannon) unite computation, probability, and compression. Ray Solomonoff's theory of inductive inference established that the optimal predictor of future data is the shortest computer program capable of generating past observations: **prediction is compression**.

### Huffman Coding & Shannon Source Coding Theorem
Let $X$ be a discrete random variable over alphabet $\Sigma = \{a_1, \dots, a_k\}$ with probability mass function $p(x)$.
The **Shannon Entropy** measures the fundamental limit of lossless data compression:
$$H(X) = -\sum_{x \in \Sigma} p(x) \log_2 p(x) \quad \text{(bits/symbol)}$$
Shannon's Source Coding Theorem establishes that for any uniquely decodable code with codeword lengths $\ell(x)$:
$$H(X) \le \bar{L} = \sum_{x \in \Sigma} p(x) \ell(x) < H(X) + 1$$

David Huffman (1952) introduced an optimal greedy bottom-up algorithm:
1. Initialize a min-priority queue with leaves representing alphabet symbols weighted by frequency $p(x)$.
2. Iteratively extract the two nodes with minimal weights $w_1, w_2$, merge them into a parent node with weight $w = w_1 + w_2$, and reinsert into the queue.
3. Assign binary labels $0$ and $1$ to edges. The path from root to leaf defines a **prefix-free code** (no codeword is a prefix of any other), guaranteeing instantaneous uniquely decodable streaming.

---

### Lempel-Ziv-Welch (LZW) Universal Compression
Unlike Huffman coding, which assumes known stationary symbol probabilities, LZW (Ziv & Lempel 1978, Welch 1984) is a universal dictionary coder that asymptotically achieves the entropy rate of any stationary ergodic source without prior statistical knowledge:
- **Encoding**: Dynamically maintains a dictionary mapping string substrings to integer tokens (initialized to ASCII $0..255$). Matches the longest string $W$ currently in the dictionary. Upon reading next symbol $c$, if $Wc$ is not in the dictionary, emits token for $W$, registers $Wc$ into the dictionary at the next index, and resets $W \leftarrow c$.
- **Decoding**: Reconstructs the exact dictionary on the fly solely from the stream of incoming tokens, handling the classic edge case where code refers to an entry currently being defined ($cScSc$).

---

### Normalized Compression Distance (NCD)
The theoretical **Kolmogorov Complexity** $K(x)$ of an object $x$ is the length of the shortest computer program that outputs $x$. Because $K(x)$ is uncomputable (by reduction to the Halting Problem), real-world universal compressors $C(x)$ serve as practical upper bounds:
$$K(x) \approx C(x)$$

Li, Chen, Li, Ma, and Vitányi (2004) formulated the **Normalized Information Distance (NID)**, approximated in practice by the **Normalized Compression Distance (NCD)**:
$$\text{NCD}(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$
Properties of NCD:
1. **Quasi-metric**: Non-negative, approximately symmetric $\text{NCD}(x, y) \approx \text{NCD}(y, x)$, and obeys the triangle inequality up to compressor error $\mathcal{O}\left(\frac{\log n}{n}\right)$.
2. **Parameter-free Clustering**: Enables clustering of unaligned genomic sequences, musical scores, literature authorship, and programming languages without domain-specific feature engineering.

## 2. Implementation Details
The module `Z04_Algorithmic_Information_Theory_and_Universal_Compression.py`:
- Implements `HuffmanCoding` with priority queue min-heap tree construction, Shannon entropy computation, bit-level stream encoding, and prefix-free branch decoding.
- Implements `LZWCompression` supporting streaming byte inputs and dictionary inversion with edge-case resolution.
- Implements `NormalizedCompressionDistance` using maximal-effort DEFLATE dictionary compression to evaluate semantic distance between text corpora.

## 3. Asymptotic & Numerical Analysis
- **Huffman Coding**: Tree construction runs in $\mathcal{O}(|\Sigma| \log |\Sigma|)$ time; encoding and decoding scale in linear time $\mathcal{O}(N)$.
- **LZW Compression**: $\mathcal{O}(N)$ processing time using hash table lookups for substring extensions.
- **NCD Computation**: Bounded by compressor throughput $\mathcal{O}(|x| + |y|)$.

## 4. Edge Cases & Numerical Stability
- In LZW, when the encoder encounters pattern $c S c S c$, the decoder receives a token that has not yet been registered in its local table. Handled by detecting index equal to current dictionary size and synthesizing entry $W + W[0]$.

## 5. Standalone Verification
The test suite validates:
- Proof that Huffman average code length satisfies $H(X) \le \bar{L} < H(X) + 1$ on English text.
- Byte-for-byte lossless round-trip recovery for both Huffman and LZW pipelines.
- Confirmation that NCD correctly measures zero self-distance and identifies same-language texts as significantly closer ($\text{NCD} \approx 0.49$) than cross-language pairs ($\text{NCD} \approx 0.92$).

## 6. Real-World Applications & Context
Powers universal file formats (ZIP, GZIP, PNG, GIF), network protocol streaming, tokenization algorithms in Large Language Models (Byte-Pair Encoding BPE is a direct variant of dictionary compression), and parameter-free nearest-neighbor classification in genomics and malware detection.

"""
Z04: Algorithmic Information Theory & Universal Compression
Pure Python & NumPy implementation of:
1. Huffman Coding: Prefix-free optimal codebook generation via min-heap tree synthesis,
   entropy evaluation, and Shannon source coding theorem verification.
2. Lempel-Ziv-Welch (LZW): Universal dictionary compression and exact lossless streaming inversion.
3. Normalized Compression Distance (NCD): Parameter-free quasi-metric distance approximating
   Kolmogorov complexity for cross-domain similarity and classification.
"""

import heapq
import collections
import zlib
import numpy as np

class HuffmanCoding:
    """
    Computes prefix-free minimum-redundancy variable-length binary codes
    achieving the Shannon source coding limit H(X) <= L_avg < H(X) + 1.
    """
    @staticmethod
    def compute_entropy(text: str) -> float:
        """Computes empirical Shannon entropy H(X) = -sum p(x) log2 p(x) in bits."""
        counts = collections.Counter(text)
        n = len(text)
        return -sum((c / n) * np.log2(c / n) for c in counts.values())

    @classmethod
    def build_codebook(cls, text: str) -> tuple[dict[str, str], float, float]:
        """
        Builds prefix-free codebook via priority queue min-heap tree reduction.
        Returns: (codebook, entropy H, average_code_length L_avg).
        """
        counts = collections.Counter(text)
        n = len(text)
        heap = [[weight, [char, '']] for char, weight in counts.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        codebook = dict(heapq.heappop(heap)[1:])
        entropy = cls.compute_entropy(text)
        avg_len = sum((counts[char] / n) * len(code) for char, code in codebook.items())
        return codebook, entropy, avg_len

    @staticmethod
    def encode(text: str, codebook: dict[str, str]) -> str:
        """Encodes text to a binary string representation."""
        return ''.join(codebook[c] for c in text)

    @staticmethod
    def decode(bitstring: str, codebook: dict[str, str]) -> str:
        """Decodes binary string back to original characters using prefix-free lookup."""
        inv_codebook = {v: k for k, v in codebook.items()}
        current = ''
        decoded = []
        for bit in bitstring:
            current += bit
            if current in inv_codebook:
                decoded.append(inv_codebook[current])
                current = ''
        assert current == '', "Trailing invalid bits in bitstring"
        return ''.join(decoded)


class LZWCompression:
    """
    Lempel-Ziv-Welch (LZW) Universal Dictionary Compression (Welch 1984).
    Dynamically constructs vocabulary trees during compression and decompression
    without transmitting dictionary codebooks.
    """
    @staticmethod
    def compress(data: bytes) -> list[int]:
        """Compresses byte payload into a sequence of integer dictionary tokens."""
        dictionary = {bytes([i]): i for i in range(256)}
        dict_size = 256
        curr = b''
        result = []

        for byte in data:
            c = bytes([byte])
            wc = curr + c
            if wc in dictionary:
                curr = wc
            else:
                result.append(dictionary[curr])
                dictionary[wc] = dict_size
                dict_size += 1
                curr = c

        if curr:
            result.append(dictionary[curr])
        return result

    @staticmethod
    def decompress(compressed: list[int]) -> bytes:
        """
        Decompresses integer token sequence into original bytes, correctly handling
        the classic special case where token represents c + S + c + S + c.
        """
        if not compressed:
            return b''

        dictionary = {i: bytes([i]) for i in range(256)}
        dict_size = 256
        w = bytes([compressed[0]])
        result = bytearray(w)

        for k in compressed[1:]:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                # Classic edge case: pattern occurs before entry enters dictionary
                entry = w + w[:1]
            else:
                raise ValueError(f"Corrupted LZW code: {k}")

            result.extend(entry)
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
            w = entry

        return bytes(result)


class NormalizedCompressionDistance:
    """
    Normalized Compression Distance (NCD) (Cilibrasi & Vitanyi 2005).
    Approximates Kolmogorov Complexity K(x) using a universal compressor C(x):
        NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
    Forms a parameter-free quasi-metric distance measuring information divergence.
    """
    @staticmethod
    def distance(x: bytes, y: bytes) -> float:
        """Computes NCD between byte sequences x and y in range [0, 1+eps]."""
        if x == y:
            return 0.0

        cx = len(zlib.compress(x, level=9))
        cy = len(zlib.compress(y, level=9))
        cxy = len(zlib.compress(x + y, level=9))

        return float((cxy - min(cx, cy)) / max(cx, cy))


if __name__ == "__main__":
    print("--- Z04: Algorithmic Information Theory & Universal Compression Sanity Check ---")

    # 1. Huffman Coding Test & Shannon Bound Verification
    sample_text = "ALGORITHM_CODEX_COMPREHENSIVE_FIRST_PRINCIPLES_MATHEMATICAL_ENCYCLOPEDIA"
    codebook, H, L_avg = HuffmanCoding.build_codebook(sample_text)
    print(f"Alphabet Size: {len(codebook)} | Shannon Entropy H(X): {H:.4f} bits/symbol")
    print(f"Huffman Average Codeword Length L_avg: {L_avg:.4f} bits/symbol")

    # Shannon Source Coding Theorem: H(X) <= L_avg < H(X) + 1
    assert H <= L_avg < H + 1.0, f"Huffman code must obey Shannon bound: {H} <= {L_avg} < {H + 1.0}"

    # Exact Lossless Encode/Decode Roundtrip
    encoded_bits = HuffmanCoding.encode(sample_text, codebook)
    decoded_text = HuffmanCoding.decode(encoded_bits, codebook)
    assert decoded_text == sample_text, "Decoded text must match original string byte-for-byte"
    print("Huffman Prefix-Free Coding & Shannon Bound: VERIFIED.")

    # 2. LZW Universal Dictionary Compression Test
    raw_payload = b"TOBEORNOTTOBEORTOBEORNOT_ALGORITHM_CODEX" * 15
    compressed_codes = LZWCompression.compress(raw_payload)
    decompressed_payload = LZWCompression.decompress(compressed_codes)
    
    comp_ratio = len(raw_payload) / (len(compressed_codes) * 2) # assuming 16-bit codes
    print(f"LZW Compression: {len(raw_payload)} bytes -> {len(compressed_codes)} tokens (Ratio: {comp_ratio:.2f}x)")
    assert decompressed_payload == raw_payload, "LZW round-trip decompression must be exact"
    print("LZW Universal Compression & Decompression: VERIFIED.")

    # 3. Normalized Compression Distance (NCD) Metric Test
    eng1 = b"The quick brown fox jumps over the lazy dog repeatedly across the meadow."
    eng2 = b"A swift dark fox leaps over the sleepy hound many times across the field."
    greek1 = "Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί επανειλημμένα.".encode('utf-8')
    
    ncd_same_lang = NormalizedCompressionDistance.distance(eng1, eng2)
    ncd_diff_lang = NormalizedCompressionDistance.distance(eng1, greek1)
    ncd_identity = NormalizedCompressionDistance.distance(eng1, eng1)

    print(f"NCD Identity (Eng1, Eng1)      : {ncd_identity:.4f}")
    print(f"NCD Same Language (Eng1, Eng2) : {ncd_same_lang:.4f}")
    print(f"NCD Cross Language (Eng1, Grk1): {ncd_diff_lang:.4f}")

    assert np.isclose(ncd_identity, 0.0), "Self-distance NCD(x, x) must be 0"
    assert ncd_same_lang < ncd_diff_lang, "Information distance within same language must be lower than cross-language"
    print("Normalized Compression Distance Metric Properties: VERIFIED.")

    print("Z04 Sanity Check: PASSED cleanly.")

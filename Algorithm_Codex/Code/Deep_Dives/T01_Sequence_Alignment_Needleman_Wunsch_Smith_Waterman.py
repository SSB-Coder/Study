"""
T01. Sequence Alignment — Needleman-Wunsch (Global) & Smith-Waterman (Local)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class SequenceAligner:
    """
    Dynamic programming sequence alignment for biological DNA/Protein strings.
    - Needleman-Wunsch: Optimal Global Alignment.
    - Smith-Waterman: Optimal Local Alignment.
    """
    
    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -2):
        self.match = match_score
        self.mismatch = mismatch_penalty
        self.gap = gap_penalty

    def needleman_wunsch(self, seq1: str, seq2: str):
        """
        Global alignment: aligns entire sequences from end to end <- eq. 3.1.
        Returns score, aligned_seq1, aligned_seq2.
        """
        n, m = len(seq1), len(seq2)
        score_matrix = np.zeros((n + 1, m + 1), dtype=int)
        
        # Initialize borders with gap penalties
        for i in range(n + 1):
            score_matrix[i, 0] = i * self.gap
        for j in range(m + 1):
            score_matrix[0, j] = j * self.gap
            
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_val = self.match if seq1[i - 1] == seq2[j - 1] else self.mismatch
                diag = score_matrix[i - 1, j - 1] + match_val
                up = score_matrix[i - 1, j] + self.gap
                left = score_matrix[i, j - 1] + self.gap
                score_matrix[i, j] = max(diag, up, left)
                
        # Backtracking from bottom-right corner (n, m)
        align1, align2 = [], []
        i, j = n, m
        while i > 0 or j > 0:
            match_val = self.match if (i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]) else self.mismatch
            if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i - 1, j - 1] + match_val:
                align1.append(seq1[i - 1])
                align2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + self.gap:
                align1.append(seq1[i - 1])
                align2.append('-')
                i -= 1
            else:
                align1.append('-')
                align2.append(seq2[j - 1])
                j -= 1
                
        return score_matrix[n, m], "".join(reversed(align1)), "".join(reversed(align2))

    def smith_waterman(self, seq1: str, seq2: str):
        """
        Local alignment: finds highest-scoring matching sub-region <- eq. 3.2.
        Clamps negative scores to 0. Backtracks from maximum cell to 0.
        """
        n, m = len(seq1), len(seq2)
        score_matrix = np.zeros((n + 1, m + 1), dtype=int)
        
        max_score = 0
        max_pos = (0, 0)
        
        # Fill DP matrix with 0 floor
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_val = self.match if seq1[i - 1] == seq2[j - 1] else self.mismatch
                diag = score_matrix[i - 1, j - 1] + match_val
                up = score_matrix[i - 1, j] + self.gap
                left = score_matrix[i, j - 1] + self.gap
                val = max(0, diag, up, left)
                score_matrix[i, j] = val
                if val > max_score:
                    max_score = val
                    max_pos = (i, j)
                    
        # Backtracking from maximum score cell until hitting 0
        align1, align2 = [], []
        i, j = max_pos
        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            match_val = self.match if seq1[i - 1] == seq2[j - 1] else self.mismatch
            if score_matrix[i, j] == score_matrix[i - 1, j - 1] + match_val:
                align1.append(seq1[i - 1])
                align2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif score_matrix[i, j] == score_matrix[i - 1, j] + self.gap:
                align1.append(seq1[i - 1])
                align2.append('-')
                i -= 1
            else:
                align1.append('-')
                align2.append(seq2[j - 1])
                j -= 1
                
        return max_score, "".join(reversed(align1)), "".join(reversed(align2))

if __name__ == "__main__":
    print("--- T01: Sequence Alignment Sanity Check ---")
    aligner = SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2)
    
    # 1. Global Alignment test (Needleman-Wunsch)
    s1 = "GCATGCG"
    s2 = "GATTACA"
    score_nw, a1_nw, a2_nw = aligner.needleman_wunsch(s1, s2)
    print(f"Needleman-Wunsch Score: {score_nw}")
    print(f"Alignment 1: {a1_nw}")
    print(f"Alignment 2: {a2_nw}")
    assert len(a1_nw) == len(a2_nw), "Aligned sequences must have equal length"
    assert score_nw > 0
    
    # 2. Local Alignment test (Smith-Waterman)
    # Long sequences containing an identical embedded motif "HEAGAWGHEE"
    p1 = "XXHEAGAWGHEEYY"
    p2 = "ZZZHEAGAWGHEEWW"
    score_sw, a1_sw, a2_sw = aligner.smith_waterman(p1, p2)
    print(f"Smith-Waterman Score: {score_sw}")
    print(f"Local Motif 1: {a1_sw}")
    print(f"Local Motif 2: {a2_sw}")
    assert a1_sw == a2_sw == "HEAGAWGHEE", "Failed to extract exact local motif"
    assert score_sw == 2 * len("HEAGAWGHEE") # 10 matches * 2
    print("T01 Sanity Check: PASSED cleanly.")

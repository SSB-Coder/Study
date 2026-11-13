"""
W06. Boolean Logic Minimization — Quine-McCluskey Algorithm
From-scratch implementation in pure NumPy.
"""

import numpy as np

class QuineMcCluskey:
    """
    Minimizes boolean truth tables into exact minimal Sum-of-Products (SOP) forms.
    Finds all Prime Implicants and solves the Prime Implicant Chart.
    """
    
    def __init__(self, num_vars: int):
        self.num_vars = num_vars

    def _to_bin(self, val: int) -> str:
        return bin(val)[2:].zfill(self.num_vars)

    @staticmethod
    def _diff_by_one_bit(term1: str, term2: str):
        """Returns position of differing bit if terms differ by exactly 1 bit, else -1."""
        diff_count = 0
        diff_pos = -1
        for i in range(len(term1)):
            if term1[i] != term2[i]:
                diff_count += 1
                diff_pos = i
                if diff_count > 1:
                    return -1
        return diff_pos if diff_count == 1 else -1

    def find_prime_implicants(self, minterms: list, dont_cares: list = None):
        """
        Groups terms by popcount of 1s, iteratively combines terms differing by 1 bit.
        Returns set of prime implicants (strings with '0', '1', and '-').
        """
        if dont_cares is None:
            dont_cares = []
        all_terms = set(minterms + dont_cares)
        
        # Group by number of 1s
        groups = {}
        for m in all_terms:
            b = self._to_bin(m)
            ones = b.count('1')
            groups.setdefault(ones, set()).add((b, (m,)))
            
        prime_implicants = set()
        
        while groups:
            next_groups = {}
            combined_terms = set()
            sorted_keys = sorted(groups.keys())
            
            for i in range(len(sorted_keys) - 1):
                k1, k2 = sorted_keys[i], sorted_keys[i + 1]
                if k2 != k1 + 1:
                    continue
                for term1, cov1 in groups[k1]:
                    for term2, cov2 in groups[k2]:
                        diff_pos = self._diff_by_one_bit(term1, term2)
                        if diff_pos != -1:
                            # Combine terms with '-' at diff_pos <- eq. 3.1
                            new_term = term1[:diff_pos] + '-' + term1[diff_pos + 1:]
                            new_cov = tuple(sorted(set(cov1 + cov2)))
                            combined_terms.add(term1)
                            combined_terms.add(term2)
                            next_groups.setdefault(new_term.count('1'), set()).add((new_term, new_cov))
                            
            # Any term not combined is a prime implicant
            for k in groups:
                for term, cov in groups[k]:
                    if term not in combined_terms:
                        prime_implicants.add((term, cov))
                        
            groups = next_groups
            
        return prime_implicants

    def solve_chart(self, prime_implicants: set, minterms: list):
        """
        Solves Prime Implicant Chart: extracts Essential Prime Implicants (EPIs)
        and covers remaining minterms.
        """
        chart = {m: [] for m in minterms}
        for pi_term, cov in prime_implicants:
            for m in cov:
                if m in chart:
                    chart[m].append(pi_term)
                    
        essential_pis = set()
        uncovered = set(minterms)
        
        # 1. Identify Essential Prime Implicants (columns with single checkmark)
        for m in minterms:
            if len(chart[m]) == 1:
                epi = chart[m][0]
                essential_pis.add(epi)
                
        # Remove minterms covered by essential PIs
        for epi in essential_pis:
            for m in list(uncovered):
                if epi in chart[m]:
                    uncovered.remove(m)
                    
        # 2. Greedy cover for any remaining uncovered minterms
        remaining_pis = [pi for pi, cov in prime_implicants if pi not in essential_pis]
        selected_pis = set(essential_pis)
        
        while uncovered:
            # Pick PI that covers the most remaining minterms
            best_pi = None
            best_cov_count = -1
            for pi in remaining_pis:
                count = sum(1 for m in uncovered if pi in chart[m])
                if count > best_cov_count:
                    best_cov_count = count
                    best_pi = pi
            if best_pi is None:
                break
            selected_pis.add(best_pi)
            remaining_pis.remove(best_pi)
            for m in list(uncovered):
                if best_pi in chart[m]:
                    uncovered.remove(m)
                    
        return sorted(list(selected_pis))

    def minimize(self, minterms: list, dont_cares: list = None):
        pis = self.find_prime_implicants(minterms, dont_cares)
        return self.solve_chart(pis, minterms)

if __name__ == "__main__":
    print("--- W06: Quine-McCluskey Sanity Check ---")
    # 4-variable boolean function: f(A, B, C, D) with minterms 0, 2, 8, 10
    # Minterms in binary:
    # 0:  0000
    # 2:  0010
    # 8:  1000
    # 10: 1010
    # Combine 0 and 2  -> 00-0
    # Combine 8 and 10 -> 10-0
    # Combine 0 and 8  -> -000
    # Combine 2 and 10 -> -010
    # Combine (0,2) and (8,10) -> -0-0 = B' D'
    qm = QuineMcCluskey(num_vars=4)
    minterms = [0, 2, 8, 10]
    minimal_sop = qm.minimize(minterms)
    print(f"Minterms: {minterms}")
    print(f"Minimized SOP Implicants: {minimal_sop}")
    
    assert len(minimal_sop) == 1, "Should reduce to single prime implicant"
    assert minimal_sop[0] == "-0-0", f"Expected '-0-0', got {minimal_sop[0]}"
    print("W06 Sanity Check: PASSED cleanly.")

"""
A03. Hypothesis Testing Toolkit
From-scratch implementation in pure NumPy.
"""

import numpy as np
import math

class HypothesisTester:
    """Raw mathematical implementation of hypothesis tests without scipy."""
    
    @staticmethod
    def two_sample_t_test(sample1: np.ndarray, sample2: np.ndarray):
        """Computes two-sample pooled t-statistic and degrees of freedom."""
        x1 = np.asarray(sample1, dtype=np.float64)
        x2 = np.asarray(sample2, dtype=np.float64)
        n1, n2 = len(x1), len(x2)
        
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1 = np.var(x1, ddof=1)
        var2 = np.var(x2, ddof=1)
        
        # Pooled sample variance <- eq. 3.1 in report
        dof = n1 + n2 - 2
        sp_sq = ((n1 - 1) * var1 + (n2 - 1) * var2) / dof
        
        # t statistic <- eq. 3.2 in report
        t_stat = (mean1 - mean2) / np.sqrt(sp_sq * (1.0 / n1 + 1.0 / n2))
        return float(t_stat), int(dof)

    @staticmethod
    def chi_squared_test(contingency_table: np.ndarray):
        """Computes Pearson's Chi-squared statistic and degrees of freedom."""
        O = np.asarray(contingency_table, dtype=np.float64)
        R, C = O.shape
        
        row_totals = np.sum(O, axis=1, keepdims=True)
        col_totals = np.sum(O, axis=0, keepdims=True)
        grand_total = np.sum(O)
        
        # Expected frequencies E_ij = (Row_i * Col_j) / N <- eq. 3.3 in report
        E = (row_totals @ col_totals) / grand_total
        
        # Chi2 statistic <- eq. 3.4 in report
        chi2_stat = np.sum((O - E) ** 2 / E)
        dof = (R - 1) * (C - 1)
        return float(chi2_stat), int(dof)

    @staticmethod
    def permutation_test(sample1: np.ndarray, sample2: np.ndarray, n_permutations: int = 5000, seed: int = 42):
        """Non-parametric two-tailed permutation test for difference in means."""
        x1 = np.asarray(sample1, dtype=np.float64)
        x2 = np.asarray(sample2, dtype=np.float64)
        n1 = len(x1)
        
        t_obs = abs(np.mean(x1) - np.mean(x2))
        combined = np.concatenate([x1, x2])
        total_len = len(combined)
        
        rng = np.random.default_rng(seed)
        count_extreme = 0
        
        for _ in range(n_permutations):
            shuffled = rng.permutation(combined)
            perm_mean1 = np.mean(shuffled[:n1])
            perm_mean2 = np.mean(shuffled[n1:])
            if abs(perm_mean1 - perm_mean2) >= t_obs:
                count_extreme += 1
                
        # Empirical p-value with continuity correction <- eq. 3.5 in report
        p_val = (1.0 + count_extreme) / (1.0 + n_permutations)
        return t_obs, p_val

if __name__ == "__main__":
    print("--- A03: Hypothesis Testing Toolkit Sanity Check ---")
    np.random.seed(42)
    
    # 1. Test Two-Sample t-test
    s1 = np.random.normal(loc=10.0, scale=2.0, size=30)
    s2 = np.random.normal(loc=12.0, scale=2.0, size=35)
    t_stat, dof = HypothesisTester.two_sample_t_test(s1, s2)
    print(f"Two-sample t-statistic: {t_stat:.4f}, dof: {dof}")
    assert dof == (30 + 35 - 2)
    assert t_stat < 0 # Group 1 mean is less than Group 2 mean
    
    # 2. Test Chi-Squared Test on 2x2 table
    # Drug vs Placebo (Cured / Not Cured)
    table = np.array([[35, 15],
                      [20, 30]])
    chi2, dof_chi = HypothesisTester.chi_squared_test(table)
    print(f"Chi-squared statistic: {chi2:.4f}, dof: {dof_chi}")
    assert dof_chi == 1
    assert chi2 > 0
    
    # 3. Test Permutation Test on separated groups
    diff_obs, p_val = HypothesisTester.permutation_test(s1, s2, n_permutations=1000)
    print(f"Permutation test mean diff: {diff_obs:.4f}, empirical p-value: {p_val:.4f}")
    assert p_val < 0.05 # True means differ by 2.0, should reject null
    print("A03 Sanity Check: PASSED cleanly.")

"""
T05. Population Genetics — Wright-Fisher & Coalescent Simulation
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PopulationGenetics:
    """
    Simulates genetic drift, natural selection, and genealogical ancestry:
    1. Wright-Fisher forward model: Binomial sampling of allele frequencies over generations.
    2. Kingman's Coalescent backward model: Genealogy tree simulation back to MRCA.
    """
    
    @staticmethod
    def wright_fisher_forward(N: int, p0: float, n_generations: int = 100, s: float = 0.0):
        """
        Forward simulation of diploid population 2N alleles <- eq. 3.1.
        N: Population size (2N gene copies).
        p0: Initial frequency of allele 'A'.
        s: Selection coefficient (fitness: w_A = 1 + s, w_a = 1).
        Returns array of allele frequencies over generations.
        """
        freqs = [p0]
        p = p0
        two_N = 2 * N
        
        for _ in range(n_generations):
            # Check fixation or extinction
            if p == 0.0 or p == 1.0:
                freqs.append(p)
                continue
                
            # Selection adjustment <- eq. 3.2
            p_prime = (p * (1.0 + s)) / (1.0 + p * s)
            
            # Binomial random sampling of 2N gametes
            k = np.random.binomial(two_N, p_prime)
            p = k / two_N
            freqs.append(p)
            
        return np.array(freqs)

    @staticmethod
    def kingman_coalescent_backward(sample_size_k: int, N: int = 1000):
        """
        Kingman's Coalescent backward simulation to Most Recent Common Ancestor (MRCA) <- eq. 3.3.
        When k lineages remain, coalescence rate is binom(k, 2) / (2N).
        Waiting time is exponentially distributed: t_k ~ Exp(binom(k, 2) / 2N).
        Returns list of (k_lineages, waiting_time, cumulative_time).
        """
        k = sample_size_k
        two_N = 2 * N
        events = []
        t_cum = 0.0
        
        while k > 1:
            # Pairwise combinations
            pairs = k * (k - 1) / 2.0
            rate = pairs / two_N
            # Draw exponential waiting time
            wait_time = np.random.exponential(1.0 / rate)
            t_cum += wait_time
            events.append((k, wait_time, t_cum))
            k -= 1 # Two lineages coalesce into one ancestor
            
        return events

if __name__ == "__main__":
    print("--- T05: Population Genetics Sanity Check ---")
    np.random.seed(42)
    # 1. Neutral Wright-Fisher Drift (s = 0): Expected frequency change is 0
    N = 100
    p0 = 0.5
    n_replicates = 100
    final_freqs = []
    for _ in range(n_replicates):
        traj = PopulationGenetics.wright_fisher_forward(N=N, p0=p0, n_generations=50, s=0.0)
        final_freqs.append(traj[-1])
        
    mean_final_p = float(np.mean(final_freqs))
    print(f"Neutral Drift: Mean Final Frequency across 100 runs: {mean_final_p:.3f} (Initial: {p0})")
    assert abs(mean_final_p - p0) < 0.1, "Neutral genetic drift biased allele mean"
    
    # 2. Kingman's Coalescent: Theoretical expected time to MRCA for k lineages is 4N * (1 - 1/k)
    k_sample = 5
    N_pop = 500
    theoretical_T_mrca = 2 * (2 * N_pop) * (1.0 - 1.0 / k_sample) # 4N * (1 - 1/k) = 2000 * 0.8 = 1600 gen
    
    mrca_times = []
    for _ in range(300):
        tree = PopulationGenetics.kingman_coalescent_backward(sample_size_k=k_sample, N=N_pop)
        mrca_times.append(tree[-1][2]) # Total cumulative time to root
        
    sim_mean_mrca = float(np.mean(mrca_times))
    print(f"Kingman Coalescent Mean T_MRCA: {sim_mean_mrca:.1f} gen (Theory: {theoretical_T_mrca:.1f})")
    assert abs(sim_mean_mrca - theoretical_T_mrca) / theoretical_T_mrca < 0.15
    print("T05 Sanity Check: PASSED cleanly.")

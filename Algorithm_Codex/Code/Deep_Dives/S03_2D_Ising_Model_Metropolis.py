"""
S03. The 2D Ising Model via Metropolis Monte Carlo
From-scratch implementation in pure NumPy.
"""

import numpy as np

class IsingModel2D:
    """
    2D Ferromagnetic Ising Model simulated on an L x L square lattice
    with periodic boundary conditions using Metropolis Monte Carlo.
    Hamiltonian: H = -J * sum_<i,j> (s_i * s_j)
    Undergoes continuous phase transition at Onsager's critical temperature T_c ~ 2.269 J/k_B.
    """
    
    def __init__(self, L: int = 20, J: float = 1.0, k_B: float = 1.0):
        self.L = L
        self.J = J
        self.k_B = k_B
        # Initialize random spin lattice: s_i in {-1, +1}
        self.spins = np.random.choice([-1, 1], size=(L, L))

    def energy(self) -> float:
        """Computes total lattice energy H <- eq. 3.1."""
        L = self.L
        E = 0.0
        for r in range(L):
            for c in range(L):
                s = self.spins[r, c]
                # Right neighbor (c+1) and Down neighbor (r+1) with periodic wrap
                right = self.spins[r, (c + 1) % L]
                down = self.spins[(r + 1) % L, c]
                E += -self.J * s * (right + down)
        return E

    def magnetization(self) -> float:
        """Mean magnetization per spin: m = (1 / N) * sum s_i."""
        return float(np.mean(self.spins))

    def monte_carlo_sweep(self, T: float):
        """
        One full Monte Carlo sweep: attempts L*L single spin flips using Metropolis criterion <- eq. 3.2.
        Delta E = 2 * J * s_i * sum(s_neighbors)
        Accept with probability min(1, exp(-Delta E / k_B T)).
        """
        L = self.L
        beta = 1.0 / (self.k_B * T)
        
        for _ in range(L * L):
            # Pick random lattice site
            r = np.random.randint(0, L)
            c = np.random.randint(0, L)
            s = self.spins[r, c]
            
            # Sum of 4 nearest neighbors (periodic boundaries)
            nb_sum = (self.spins[(r - 1) % L, c] +
                      self.spins[(r + 1) % L, c] +
                      self.spins[r, (c - 1) % L] +
                      self.spins[r, (c + 1) % L])
                      
            # Energy difference if spin is flipped: s -> -s <- eq. 3.2
            delta_E = 2.0 * self.J * s * nb_sum
            
            # Metropolis acceptance
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E * beta):
                self.spins[r, c] = -s

if __name__ == "__main__":
    print("--- S03: 2D Ising Model Sanity Check ---")
    np.random.seed(42)
    # Onsager's critical temperature: T_c = 2 * J / (k_B * ln(1 + sqrt(2))) ~ 2.269
    T_c = 2.269
    
    # 1. Low Temperature regime (T = 0.5 << T_c): Ferromagnetic ordering preserved (|m| -> 1.0)
    ising_low = IsingModel2D(L=16)
    ising_low.spins = np.ones((16, 16)) # Start aligned
    for _ in range(100):
        ising_low.monte_carlo_sweep(T=0.5)
    mag_low = abs(ising_low.magnetization())
    print(f"Low Temp (T=0.5) Mean |Magnetization|: {mag_low:.4f} (Expected near 1.0)")
    assert mag_low > 0.85, "Ferromagnetic alignment failed at low temperature"
    
    # 2. High Temperature regime (T = 5.0 >> T_c): Paramagnetic disorder (|m| -> 0.0)
    for _ in range(100):
        ising_low.monte_carlo_sweep(T=5.0)
    mag_high = abs(ising_low.magnetization())
    print(f"High Temp (T=5.0) Mean |Magnetization|: {mag_high:.4f} (Expected near 0.0)")
    assert mag_high < 0.25, "Paramagnetic disorder failed at high temperature"
    print("S03 Sanity Check: PASSED cleanly.")

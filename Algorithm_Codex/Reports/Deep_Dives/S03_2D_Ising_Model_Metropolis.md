# Deep Dive: S03 — The 2D Ising Model via Metropolis Monte Carlo

## 1. The Beginner's Analogy: Voting in a Friendly Neighborhood
Imagine a grid of 400 neighbors who can vote either Blue (+1) or Red (-1):
- Every neighbor wants to agree with their four immediate adjacent neighbors (ferromagnetic coupling).
- **At Low Temperature (Low Noise / Cold):** Peer pressure dominates completely. A single consensus spreads like wildfire until all 400 neighbors vote unanimous Blue (Spontaneous Magnetization $|m| \approx 1$).
- **At High Temperature (High Noise / Hot):** Everyone behaves completely erratically, voting at random without caring what their neighbors do ($|m| \approx 0$).
- **At the Critical Temperature ($T_c \approx 2.269$):** The system balances on a knife-edge: giant swirling fractal islands of Blue and Red emerge, creating infinite correlation lengths.

---

## 2. The Deep Architecture/Algorithm Breakdown
The Ising model is the cornerstone of statistical mechanics and phase transitions:
1. **Lattice Hamiltonian:**
   $$\mathcal{H}(\mathbf{s}) = -J \sum_{\langle i, j \rangle} s_i s_j$$
   where $s_i \in \{-1, +1\}$ and $\langle i, j \rangle$ denotes nearest-neighbor pairs.
2. **Boltzmann Distribution:**
   $$P(\mathbf{s}) = \frac{1}{Z} e^{-\beta \mathcal{H}(\mathbf{s})}, \quad \beta = \frac{1}{k_B T}$$
3. **Metropolis-Hastings MCMC:**
   Because the partition function $Z = \sum_{\mathbf{s}} e^{-\beta \mathcal{H}}$ requires summing $2^{N}$ configurations (impossible for $N=400$, $2^{400} \approx 10^{120}$), Metropolis MCMC evaluates the local energy difference $\Delta E$ when a single spin is proposed to flip.

---

## 3. The Hardcore Mathematics

### 3.1 Local Flip Energy Difference
When spin $s_i$ flips to $-s_i$, only interactions with its 4 immediate neighbors change:
$$\Delta E = \mathcal{H}(\mathbf{s}^{\text{flipped}}) - \mathcal{H}(\mathbf{s}) = 2 J s_i \sum_{j \in N(i)} s_j$$

### 3.2 Metropolis Acceptance Ratio
$$\alpha(s_i \to -s_i) = \min\left(1, \frac{P(\mathbf{s}^{\text{flipped}})}{P(\mathbf{s})}\right) = \min\left(1, e^{-\beta \Delta E}\right)$$

### 3.3 Onsager's Exact Critical Temperature (1944)
Lars Onsager famously solved the 2D Ising model analytically:
$$k_B T_c = \frac{2J}{\ln(1 + \sqrt{2})} \approx 2.269185 J$$
Below $T_c$, spontaneous magnetization obeys:
$$M(T) = \left[ 1 - \left( \sinh \frac{2J}{k_B T} \right)^{-4} \right]^{1/8}$$

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Create $L \times L$ grid of random spins $\pm 1$.
2. **Sweep:** For $L^2$ iterations:
   - Pick site $(r, c)$ at random.
   - Compute $\Delta E = 2 J s_i \sum s_{\text{nb}}$.
   - If $\Delta E \le 0$, accept flip $s_i \leftarrow -s_i$.
   - Else accept with probability $\exp(-\Delta E / k_B T)$.
3. **Measure:** Calculate mean magnetization $m = \frac{1}{N} \sum s_i$ and energy $E$.

---

## 5. Limitations / When It Breaks
- **Critical Slowing Down:** Near $T_c$, correlation time blows up as $\tau \propto L^z$ with $z \approx 2.1$. Wolff and Swendsen-Wang cluster algorithms flip entire correlated islands in a single step to eliminate slowing down.

---

## 6. Where This Fits
- **Cross-Reference MCMC (E03) & HMC (P01):** The Ising model is the foundational physical benchmark for all Markov Chain Monte Carlo algorithms.
- **Hopfield Networks (K01) & Boltzmann Machines:** Energy-based neural networks directly adapt the Ising spin Hamiltonian into trainable associative memories.

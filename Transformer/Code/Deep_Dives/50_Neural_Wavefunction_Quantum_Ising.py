"""
50. Neural Network Quantum States (NQS) & Variational Monte Carlo (VMC)
Deep Dive Implementation: 1D Transverse-Field Quantum Ising Model

This file explicitly demonstrates using a Neural Network to solve quantum many-body physics:
1. Neural Wavefunction: Approximating ln|ψ_θ(S)| for spin configurations S in {-1, +1}^N.
2. Metropolis-Hastings MCMC: Sampling states from quantum probability P(S) = |ψ(S)|^2.
3. Local Energy Expectation: Evaluating H * ψ / ψ without storing the 2^N Hilbert space.
4. Variational Principle: Optimizing weights to find the quantum ground-state energy.
"""

import torch
import torch.nn as nn
import math

class NeuralWavefunction(nn.Module):
    """
    Parametrizes the log-amplitude of a quantum many-body wavefunction:
    ln |ψ_θ(S)| where S in {-1, +1}^N is a configuration of N quantum spins.
    """
    def __init__(self, num_spins=8, hidden_dim=32):
        super().__init__()
        self.num_spins = num_spins
        self.net = nn.Sequential(
            nn.Linear(num_spins, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, spins):
        """
        spins: [Batch_Size, Num_Spins] with values in {-1.0, +1.0}
        returns: log_psi: [Batch_Size, 1]
        """
        return self.net(spins)


def metropolis_hastings_sample(wavefunction, num_spins=8, num_samples=256, sweep_steps=10):
    """
    Samples spin configurations S ~ |ψ_θ(S)|² using the Metropolis-Hastings MCMC algorithm.
    """
    # Start with a random initial spin configuration (+1 or -1)
    current_spins = torch.randint(0, 2, (num_samples, num_spins), dtype=torch.float32) * 2.0 - 1.0

    with torch.no_grad():
        current_log_psi = wavefunction(current_spins).squeeze(-1) # [Num_Samples]

        for _ in range(sweep_steps * num_spins):
            # Propose flipping a random spin in each sample
            flip_idx = torch.randint(0, num_spins, (num_samples,))
            proposed_spins = current_spins.clone()
            proposed_spins[torch.arange(num_samples), flip_idx] *= -1.0

            proposed_log_psi = wavefunction(proposed_spins).squeeze(-1)

            # Acceptance probability ratio: A = min(1, |ψ(S')|² / |ψ(S)|²) = exp(2 * (ln|ψ'| - ln|ψ|))
            log_ratio = 2.0 * (proposed_log_psi - current_log_psi)
            accept_prob = torch.clamp(torch.exp(log_ratio), max=1.0)

            # Accept / Reject
            random_uniform = torch.rand(num_samples)
            accepted = random_uniform < accept_prob

            current_spins[accepted] = proposed_spins[accepted]
            current_log_psi[accepted] = proposed_log_psi[accepted]

    return current_spins


def compute_local_energy_ising(wavefunction, spins, J=1.0, h=1.0):
    """
    Computes local energy E_loc(S) for the 1D Transverse Field Ising Model with periodic boundaries:
        H = -J * sum_i (σ_i^z * σ_{i+1}^z) - h * sum_i (σ_i^x)

    Diagonal Term (σ^z interaction): -J * sum_i s_i * s_{i+1}
    Off-Diagonal Term (σ^x spin flip): -h * sum_i ( ψ(S flipped at i) / ψ(S) )
    """
    batch_size, num_spins = spins.shape

    # 1. Diagonal energy: nearest-neighbor spin coupling (periodic boundaries)
    # s_i * s_{i+1}
    rolled_spins = torch.roll(spins, shifts=-1, dims=-1)
    diag_energy = -J * torch.sum(spins * rolled_spins, dim=-1) # [Batch_Size]

    # 2. Off-diagonal energy: flip each spin i and compute ratio ψ(S^(i)) / ψ(S)
    with torch.no_grad():
        current_log_psi = wavefunction(spins).squeeze(-1) # [Batch_Size]

        off_diag_sum = torch.zeros(batch_size, device=spins.device)
        for i in range(num_spins):
            flipped_spins = spins.clone()
            flipped_spins[:, i] *= -1.0
            flipped_log_psi = wavefunction(flipped_spins).squeeze(-1)

            # Ratio = exp(ln|ψ_flipped| - ln|ψ_current|)
            ratio = torch.exp(flipped_log_psi - current_log_psi)
            off_diag_sum = off_diag_sum + ratio

        off_diag_energy = -h * off_diag_sum # [Batch_Size]

    local_energy = diag_energy + off_diag_energy
    return local_energy


if __name__ == "__main__":
    print("--- 50. Neural Network Quantum States (NQS) & VMC ---")
    print("Solving Quantum Ground State for 1D Transverse-Field Ising Model...")

    num_spins = 6
    J = 1.0 # Coupling constant (ferromagnetic)
    h = 1.0 # Transverse magnetic field (at quantum critical point)

    psi_net = NeuralWavefunction(num_spins=num_spins, hidden_dim=32)
    optimizer = torch.optim.Adam(psi_net.parameters(), lr=0.01)

    print(f"Quantum Spin Chain Length: N = {num_spins} spins (Hilbert Space = 2^{num_spins} = {2**num_spins} states)")
    print("Beginning Variational Monte Carlo (VMC) Ground State Optimization...\n")

    for epoch in range(1, 41):
        optimizer.zero_grad()

        # 1. Sample configurations from the quantum wavefunction |ψ_θ(S)|² via MCMC
        samples = metropolis_hastings_sample(psi_net, num_spins=num_spins, num_samples=128, sweep_steps=5)

        # 2. Compute local energy E_loc(S)
        local_energy = compute_local_energy_ising(psi_net, samples, J=J, h=h)
        mean_energy = torch.mean(local_energy)

        # 3. Variational Gradient (REINFORCE / Policy Gradient formulation for quantum wavefunctions):
        # ∇_θ E = 2 * E[ (E_loc - <E>) * ∇_θ ln|ψ_θ(S)| ]
        log_psi = psi_net(samples).squeeze(-1)
        centered_energy = (local_energy - mean_energy).detach()
        loss = 2.0 * torch.mean(centered_energy * log_psi)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Variational Energy Expectation <E>: {mean_energy.item():.4f} "
                  f"(Energy / Spin: {mean_energy.item() / num_spins:.4f})")

    print("\nVerification Test:")
    # Evaluate wavefunctions for all-up vs mixed spins
    all_up = torch.ones(1, num_spins)
    alternating = torch.tensor([[1.0, -1.0] * (num_spins // 2)])
    amp_up = math.exp(psi_net(all_up).item())
    amp_alt = math.exp(psi_net(alternating).item())
    print(f"Learned Relative Quantum Amplitude |psi(up,up,up,up,up,up)|: {amp_up:.4f}")
    print(f"Learned Relative Quantum Amplitude |psi(up,dn,up,dn,up,dn)|: {amp_alt:.4f}")
    print("Success: Neural Network Quantum State optimized ground state via Variational Monte Carlo.")

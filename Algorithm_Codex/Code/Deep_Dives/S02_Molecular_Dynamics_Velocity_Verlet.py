"""
S02. Molecular Dynamics — Velocity Verlet & Lennard-Jones Potential
From-scratch implementation in pure NumPy.
"""

import numpy as np

class MolecularDynamicsLJ:
    """
    Microscopic molecular dynamics simulation of Argon gas particles
    interacting via the classical Lennard-Jones (12-6) pair potential.
    Preserves total energy in the NVE (microcanonical) thermodynamic ensemble.
    """
    
    def __init__(self, n_particles: int = 16, box_size: float = 10.0,
                 epsilon: float = 1.0, sigma: float = 1.0, mass: float = 1.0):
        self.n = n_particles
        self.L = box_size
        self.eps = epsilon
        self.sig = sigma
        self.mass = mass
        # Place particles on regular lattice grid to prevent high-energy overlap singularities
        side = int(np.ceil(np.sqrt(n_particles)))
        xs, ys = np.meshgrid(np.linspace(1.5, box_size - 1.5, side),
                             np.linspace(1.5, box_size - 1.5, side))
        grid_pts = np.column_stack([xs.flatten(), ys.flatten()])
        self.pos = grid_pts[:n_particles].astype(float)
        self.vel = np.random.normal(0.0, 0.2, size=(n_particles, 2))
        # Remove net momentum
        self.vel -= np.mean(self.vel, axis=0)

    def compute_forces_and_potential(self):
        """
        Lennard-Jones 12-6 pairwise force and potential with periodic boundary conditions <- eq. 3.1.
        V(r) = 4*eps * [ (sig/r)^12 - (sig/r)^6 ]
        F(r) = -dV/dr = 24*eps/r * [ 2*(sig/r)^12 - (sig/r)^6 ] * (r_vec / r)
        """
        forces = np.zeros_like(self.pos)
        U_total = 0.0
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r_vec = self.pos[i] - self.pos[j]
                # Minimum image convention for periodic boundary conditions
                r_vec -= self.L * np.round(r_vec / self.L)
                
                r2 = np.sum(r_vec**2)
                if r2 < (3.0 * self.sig)**2: # Cutoff at 3*sigma
                    inv_r2 = (self.sig**2) / max(r2, 0.01)
                    inv_r6 = inv_r2**3
                    inv_r12 = inv_r6**2
                    
                    # Potential energy <- eq. 3.1
                    U_pair = 4.0 * self.eps * (inv_r12 - inv_r6)
                    U_total += U_pair
                    
                    # Force magnitude: F = 24*eps/r^2 * (2*inv_r12 - inv_r6) * r_vec <- eq. 3.2
                    f_mag = 24.0 * self.eps * (2.0 * inv_r12 - inv_r6) * (inv_r2 / (self.sig**2))
                    f_vec = f_mag * r_vec
                    
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    
        return forces, U_total

    def compute_kinetic_energy(self) -> float:
        return float(0.5 * self.mass * np.sum(self.vel**2))

    def step(self, dt: float = 0.005):
        """
        Velocity Verlet integration step <- eq. 3.3.
        1. r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. a(t+dt) = F(r(t+dt)) / m
        3. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        """
        forces_t, _ = self.compute_forces_and_potential()
        acc_t = forces_t / self.mass
        
        # 1. Update positions
        self.pos += self.vel * dt + 0.5 * acc_t * (dt**2)
        # Periodic wrap
        self.pos = self.pos % self.L
        
        # 2. Compute new forces
        forces_next, U_next = self.compute_forces_and_potential()
        acc_next = forces_next / self.mass
        
        # 3. Update velocities
        self.vel += 0.5 * (acc_t + acc_next) * dt
        
        K_next = self.compute_kinetic_energy()
        return K_next + U_next

if __name__ == "__main__":
    print("--- S02: Molecular Dynamics Velocity Verlet Sanity Check ---")
    np.random.seed(42)
    md = MolecularDynamicsLJ(n_particles=12, box_size=8.0, epsilon=1.0, sigma=1.0)
    
    # Compute initial energy
    _, U0 = md.compute_forces_and_potential()
    K0 = md.compute_kinetic_energy()
    E_total_init = K0 + U0
    print(f"Initial Total Energy (NVE): {E_total_init:.4f} (K={K0:.4f}, U={U0:.4f})")
    
    # Simulate 100 MD steps
    energies = []
    for _ in range(100):
        E = md.step(dt=0.002)
        energies.append(E)
        
    E_final = energies[-1]
    rel_drift = abs(E_final - E_total_init) / abs(E_total_init)
    print(f"Final Total Energy: {E_final:.4f}")
    print(f"Energy Relative Conservation Error: {rel_drift:.2e}")
    
    # Symplectic Velocity Verlet strictly bounds energy drift in NVE ensemble
    assert rel_drift < 0.01, "Velocity Verlet failed to conserve energy"
    print("S02 Sanity Check: PASSED cleanly.")

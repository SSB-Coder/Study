"""
S01. N-Body Gravitational Simulation (Direct O(N^2) & Barnes-Hut Quadtree)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class QuadNode:
    """A node in the 2D Barnes-Hut Quadtree."""
    def __init__(self, cx: float, cy: float, size: float):
        self.cx = cx
        self.cy = cy
        self.size = size # Width/height of bounding square
        self.mass = 0.0
        self.com = np.zeros(2) # Center of mass [x, y]
        self.particle = None   # Single particle index if leaf
        self.children = [None, None, None, None] # NW, NE, SW, SE

class NBodySimulation:
    """
    2D N-body gravitational simulator with symplectic Leapfrog integration.
    Supports both direct pairwise sum and Barnes-Hut O(N log N) tree approximation.
    """
    G = 1.0 # Gravitational constant in simulation units
    SOFTENING = 0.1 # Softening parameter to prevent infinite force during close encounters

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray):
        self.pos = np.array(positions, dtype=float)
        self.vel = np.array(velocities, dtype=float)
        self.mass = np.array(masses, dtype=float)
        self.n = len(masses)

    def compute_forces_direct(self) -> np.ndarray:
        """Direct O(N^2) pairwise gravitational force calculation <- eq. 3.1."""
        forces = np.zeros_like(self.pos)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r_vec = self.pos[j] - self.pos[i]
                dist_sq = np.sum(r_vec**2) + (self.SOFTENING**2)
                dist = np.sqrt(dist_sq)
                f_mag = (self.G * self.mass[i] * self.mass[j]) / dist_sq
                f_vec = f_mag * (r_vec / dist)
                forces[i] += f_vec
                forces[j] -= f_vec # Newton's 3rd Law
        return forces

    def _insert_particle(self, node: QuadNode, p_idx: int):
        px, py = self.pos[p_idx]
        pm = self.mass[p_idx]
        
        # Update center of mass and total mass <- eq. 3.2
        new_mass = node.mass + pm
        node.com = (node.mass * node.com + pm * self.pos[p_idx]) / new_mass
        node.mass = new_mass
        
        if node.particle is None and all(c is None for c in node.children):
            # Empty leaf -> insert
            node.particle = p_idx
            return
            
        if node.particle is not None:
            # Subdivide: push existing particle down into quadrant
            old_p = node.particle
            node.particle = None
            self._subdivide(node)
            self._insert_to_quadrant(node, old_p)
            
        self._insert_to_quadrant(node, p_idx)

    def _subdivide(self, node: QuadNode):
        half = node.size / 2.0
        q = node.size / 4.0
        # NW: 0, NE: 1, SW: 2, SE: 3
        node.children[0] = QuadNode(node.cx - q, node.cy + q, half)
        node.children[1] = QuadNode(node.cx + q, node.cy + q, half)
        node.children[2] = QuadNode(node.cx - q, node.cy - q, half)
        node.children[3] = QuadNode(node.cx + q, node.cy - q, half)

    def _insert_to_quadrant(self, node: QuadNode, p_idx: int):
        px, py = self.pos[p_idx]
        is_east = px >= node.cx
        is_north = py >= node.cy
        # Determine quadrant index: NW=0, NE=1, SW=2, SE=3
        quad_idx = (1 if is_east else 0) + (0 if is_north else 2)
        if node.children[quad_idx] is None:
            self._subdivide(node)
        self._insert_particle(node.children[quad_idx], p_idx)

    def _compute_force_barnes_hut(self, node: QuadNode, p_idx: int, theta_crit: float = 0.5) -> np.ndarray:
        if node.mass <= 0:
            return np.zeros(2)
        if node.particle == p_idx:
            return np.zeros(2) # Don't exert force on self
            
        r_vec = node.com - self.pos[p_idx]
        d = np.linalg.norm(r_vec)
        
        # Multipole acceptance criterion: size / d < theta_crit <- eq. 3.3
        if (node.particle is not None) or (node.size / (d + 1e-6) < theta_crit):
            dist_sq = d**2 + (self.SOFTENING**2)
            f_mag = (self.G * self.mass[p_idx] * node.mass) / dist_sq
            return f_mag * (r_vec / np.sqrt(dist_sq))
            
        # Otherwise recurse into children
        force = np.zeros(2)
        for child in node.children:
            if child is not None:
                force += self._compute_force_barnes_hut(child, p_idx, theta_crit)
        return force

    def compute_energy(self) -> float:
        """Total Hamiltonian Energy: Kinetic (K) + Gravitational Potential (U)."""
        # Kinetic energy
        K = 0.5 * np.sum(self.mass * np.sum(self.vel**2, axis=1))
        # Potential energy
        U = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r = np.linalg.norm(self.pos[j] - self.pos[i]) + self.SOFTENING
                U -= (self.G * self.mass[i] * self.mass[j]) / r
        return K + U

    def step_leapfrog(self, dt: float, use_barnes_hut: bool = False):
        """Symplectic Velocity Verlet / Leapfrog time step."""
        if not use_barnes_hut:
            forces = self.compute_forces_direct()
        else:
            # Build tree
            min_xy = np.min(self.pos, axis=0) - 1.0
            max_xy = np.max(self.pos, axis=0) + 1.0
            size = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
            center = (min_xy + max_xy) / 2.0
            root = QuadNode(center[0], center[1], size)
            for i in range(self.n):
                self._insert_particle(root, i)
            forces = np.array([self._compute_force_barnes_hut(root, i) for i in range(self.n)])
            
        acc = forces / self.mass[:, np.newaxis]
        # Half step velocity
        self.vel += 0.5 * acc * dt
        # Full step position
        self.pos += self.vel * dt
        # Recompute forces at new position
        if not use_barnes_hut:
            forces_new = self.compute_forces_direct()
        else:
            forces_new = forces
        acc_new = forces_new / self.mass[:, np.newaxis]
        self.vel += 0.5 * acc_new * dt

if __name__ == "__main__":
    print("--- S01: N-Body Simulation Sanity Check ---")
    np.random.seed(42)
    # Stable 2-body circular orbit (Sun + Earth)
    m_sun, m_earth = 1000.0, 1.0
    r_orbit = 10.0
    v_orbit = np.sqrt(NBodySimulation.G * m_sun / r_orbit) # v = sqrt(GM/r)
    
    pos = np.array([[0.0, 0.0], [r_orbit, 0.0]])
    vel = np.array([[0.0, 0.0], [0.0, v_orbit]])
    masses = np.array([m_sun, m_earth])
    
    sim = NBodySimulation(pos, vel, masses)
    E_init = sim.compute_energy()
    print(f"Initial Total Energy: {E_init:.4f}")
    
    # Run 100 orbital steps
    dt = 0.05
    for _ in range(100):
        sim.step_leapfrog(dt)
        
    E_final = sim.compute_energy()
    energy_drift = abs(E_final - E_init) / abs(E_init)
    print(f"Final Total Energy: {E_final:.4f}")
    print(f"Energy Relative Drift: {energy_drift:.2e}")
    
    # Symplectic Leapfrog preserves total Hamiltonian energy to within 0.1%
    assert energy_drift < 1e-3, "Leapfrog integrator failed to conserve energy"
    print("S01 Sanity Check: PASSED cleanly.")

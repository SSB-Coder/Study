"""
T04. Molecular Mechanics Force Fields (Bonds, Angles, Torsions)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class MolecularMechanicsForceField:
    """
    Computes classical molecular mechanics empirical potential energy and forces:
    E_total = E_bonds + E_angles + E_torsions
    Performs geometry optimization via gradient descent to find equilibrium 3D conformations.
    """
    
    def __init__(self, coordinates: np.ndarray, bonds: list, angles: list, torsions: list = None):
        """
        coordinates: (N, 3) Cartesian coordinates.
        bonds: list of (i, j, r0, k_b)
        angles: list of (i, j, k, theta0_rad, k_theta) (j is vertex)
        torsions: list of (i, j, k, l, V_n, n, gamma_rad)
        """
        self.coords = np.array(coordinates, dtype=float)
        self.bonds = bonds
        self.angles = angles
        self.torsions = [] if torsions is None else torsions

    def compute_energy(self, coords: np.ndarray = None) -> float:
        """Evaluates total empirical potential energy <- eq. 3.1."""
        if coords is None:
            coords = self.coords
            
        E_bond = 0.0
        for i, j, r0, kb in self.bonds:
            r = np.linalg.norm(coords[i] - coords[j])
            E_bond += 0.5 * kb * (r - r0)**2
            
        E_angle = 0.0
        for i, j, k, theta0, kt in self.angles:
            # Vectors from central atom j to i and k
            v_ji = coords[i] - coords[j]
            v_jk = coords[k] - coords[j]
            cos_th = np.dot(v_ji, v_jk) / (np.linalg.norm(v_ji) * np.linalg.norm(v_jk) + 1e-12)
            cos_th = np.clip(cos_th, -1.0, 1.0)
            theta = np.arccos(cos_th)
            E_angle += 0.5 * kt * (theta - theta0)**2
            
        E_torsion = 0.0
        for i, j, k, l, Vn, n, gamma in self.torsions:
            # Dihedral angle between planes (i, j, k) and (j, k, l)
            b1 = coords[j] - coords[i]
            b2 = coords[k] - coords[j]
            b3 = coords[l] - coords[k]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)
            if norm_n1 > 1e-8 and norm_n2 > 1e-8:
                cos_phi = np.dot(n1, n2) / (norm_n1 * norm_n2)
                cos_phi = np.clip(cos_phi, -1.0, 1.0)
                phi = np.arccos(cos_phi)
                E_torsion += 0.5 * Vn * (1.0 + np.cos(n * phi - gamma))
                
        return E_bond + E_angle + E_torsion

    def compute_numerical_gradient(self, h: float = 1e-6) -> np.ndarray:
        """Central difference gradient of total energy w.r.t Cartesian coordinates."""
        grad = np.zeros_like(self.coords)
        c = self.coords.copy()
        
        for i in range(len(c)):
            for d in range(3):
                c[i, d] += h
                e_plus = self.compute_energy(c)
                c[i, d] -= 2.0 * h
                e_minus = self.compute_energy(c)
                c[i, d] += h
                grad[i, d] = (e_plus - e_minus) / (2.0 * h)
                
        return grad

    def optimize_geometry(self, lr: float = 0.005, max_steps: int = 200, tol: float = 1e-4):
        """Minimizes conformational energy via gradient descent <- eq. 3.3."""
        for step in range(max_steps):
            grad = self.compute_numerical_gradient()
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                return step + 1, True
            self.coords -= lr * np.clip(grad, -10.0, 10.0)
        return max_steps, False

if __name__ == "__main__":
    print("--- T04: Molecular Mechanics Force Fields Sanity Check ---")
    np.random.seed(42)
    # Model a water molecule (H - O - H):
    # Atom 0: Oxygen at origin (0, 0, 0)
    # Atom 1: Hydrogen 1 at (0.96, 0.0, 0.0)
    # Atom 2: Hydrogen 2 strained at 90 degrees (0.0, 0.96, 0.0)
    # Target: Equilibrium O-H bond length = 0.957 Angstrom, H-O-H bond angle = 104.5 degrees (1.824 rad)
    coords_init = np.array([
        [0.0, 0.0, 0.0],
        [1.10, 0.0, 0.0],  # Stretched bond (1.10 vs 0.957)
        [0.0, 1.10, 0.0]   # Unfavorable 90 deg angle
    ])
    
    r0 = 0.957
    theta0 = np.deg2rad(104.5)
    kb = 450.0  # Bond force constant (kcal / mol A^2)
    kt = 55.0   # Angle force constant (kcal / mol rad^2)
    
    bonds = [(0, 1, r0, kb), (0, 2, r0, kb)]
    angles = [(1, 0, 2, theta0, kt)] # Oxygen 0 is vertex
    
    mm = MolecularMechanicsForceField(coords_init, bonds, angles)
    e_init = mm.compute_energy()
    print(f"Initial Strained Energy: {e_init:.4f} kcal/mol")
    
    steps, success = mm.optimize_geometry(lr=0.002, max_steps=300)
    e_opt = mm.compute_energy()
    print(f"Optimized Equilibrium Energy: {e_opt:.4f} kcal/mol in {steps} steps")
    
    # Calculate optimized bond lengths and angle
    v1 = mm.coords[1] - mm.coords[0]
    v2 = mm.coords[2] - mm.coords[0]
    bond1_len = np.linalg.norm(v1)
    bond2_len = np.linalg.norm(v2)
    opt_angle_deg = np.rad2deg(np.arccos(np.dot(v1, v2) / (bond1_len * bond2_len)))
    
    print(f"Optimized Bond 1: {bond1_len:.3f} A (Target: {r0:.3f})")
    print(f"Optimized Bond 2: {bond2_len:.3f} A (Target: {r0:.3f})")
    print(f"Optimized Angle : {opt_angle_deg:.1f} deg (Target: 104.5 deg)")
    
    assert e_opt < 1e-3, "Energy minimization failed to reach relaxed equilibrium"
    assert abs(bond1_len - r0) < 0.01, "Bond length did not relax to target"
    assert abs(opt_angle_deg - 104.5) < 1.0, "Bond angle did not relax to target"
    print("T04 Sanity Check: PASSED cleanly.")

"""
W11. Topology Optimization — Solid Isotropic Material with Penalization (SIMP)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class TopologyOptimizerSIMP:
    """
    Generative structural design on a 2D cantilever beam using the SIMP method.
    Finds optimal material distribution rho_e in [0, 1] minimizing structural compliance
    under volume fraction constraint sum(rho_e) / n_elements <= V_target.
    """
    
    def __init__(self, nelx: int = 20, nely: int = 10, volfrac: float = 0.5, penal: float = 3.0, rmin: float = 1.5):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.n_elem = nelx * nely
        
        # Element stiffness matrix for 2D plane stress unit square (nu = 0.3) <- eq. 3.1
        self.k0 = self._unit_square_stiffness(nu=0.3)

    @staticmethod
    def _unit_square_stiffness(nu: float = 0.3):
        """Element stiffness matrix for 4-node quad element in plane stress."""
        E = 1.0
        k = np.array([
            1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8,
            1/8 + nu/8, 1/2 - nu/6, 1/8 - 3*nu/8, nu/6, -1/8 - nu/8, -1/4 + nu/12, -1/8 + 3*nu/8, -1/4 - nu/12,
            -1/4 - nu/12, 1/8 - 3*nu/8, 1/2 - nu/6, -1/8 - nu/8, nu/6, -1/8 + 3*nu/8, -1/4 + nu/12, 1/8 + nu/8,
            -1/8 + 3*nu/8, nu/6, -1/8 - nu/8, 1/2 - nu/6, 1/8 - 3*nu/8, -1/4 - nu/12, 1/8 + nu/8, -1/4 + nu/12,
            -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8, 1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
            -1/8 - nu/8, -1/4 + nu/12, -1/8 + 3*nu/8, -1/4 - nu/12, 1/8 + nu/8, 1/2 - nu/6, 1/8 - 3*nu/8, nu/6,
            nu/6, -1/8 + 3*nu/8, -1/4 + nu/12, 1/8 + nu/8, -1/4 - nu/12, 1/8 - 3*nu/8, 1/2 - nu/6, -1/8 - nu/8,
            1/8 - 3*nu/8, -1/4 - nu/12, 1/8 + nu/8, -1/4 + nu/12, -1/8 + 3*nu/8, nu/6, -1/8 - nu/8, 1/2 - nu/6
        ])
        return k.reshape((8, 8))

    def optimize(self, max_iters: int = 30):
        """
        Runs SIMP topology optimization loop using Optimality Criteria (OC) updates.
        Returns final 2D density array (nely x nelx).
        """
        nelx, nely = self.nelx, self.nely
        rho = np.full((nely, nelx), self.volfrac, dtype=float)
        
        # Cantilever beam DOFs: left edge fixed (x=0)
        # Nodal layout: (nelx + 1) * (nely + 1)
        nnodes = (nelx + 1) * (nely + 1)
        ndof = 2 * nnodes
        
        # Load: downward point force at bottom-right corner
        f = np.zeros(ndof)
        f[2 * (nelx * (nely + 1) + nely) + 1] = -1.0 # Bottom-right y-DOF
        
        # Fixed DOFs: left edge (x=0)
        fixed_dofs = []
        for y in range(nely + 1):
            fixed_dofs.extend([2 * y, 2 * y + 1])
        free_dofs = np.array([d for d in range(ndof) if d not in fixed_dofs])
        
        # Precompute mesh distance filter for mesh independency <- eq. 3.2
        H = np.zeros((self.n_elem, self.n_elem), dtype=float)
        for e1 in range(self.n_elem):
            i1, j1 = divmod(e1, nelx)
            for e2 in range(self.n_elem):
                i2, j2 = divmod(e2, nelx)
                dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                if dist < self.rmin:
                    H[e1, e2] = self.rmin - dist
        Hs = np.sum(H, axis=1)

        compliance_history = []

        for it in range(max_iters):
            # 1. Assemble global K with SIMP penalization E_e = E_min + rho_e^p (E_0 - E_min)
            E_min = 1e-3
            K = np.zeros((ndof, ndof))
            
            for elx in range(nelx):
                for ely in range(nely):
                    e = ely * nelx + elx
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof = np.array([
                        2*n1, 2*n1+1, 2*n2, 2*n2+1,
                        2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
                    ])
                    # Penalized stiffness <- eq. 3.1
                    E_e = E_min + (rho[ely, elx] ** self.penal) * (1.0 - E_min)
                    K[np.ix_(edof, edof)] += E_e * self.k0
                    
            # 2. Solve Ku = f
            u = np.zeros(ndof)
            u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], f[free_dofs])
            
            # 3. Compliance and Sensitivity analysis
            c = 0.0
            dc = np.zeros((nely, nelx))
            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof = np.array([
                        2*n1, 2*n1+1, 2*n2, 2*n2+1,
                        2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
                    ])
                    ue = u[edof]
                    ce = float(ue.T @ self.k0 @ ue)
                    c += (E_min + (rho[ely, elx] ** self.penal) * (1.0 - E_min)) * ce
                    dc[ely, elx] = -self.penal * (rho[ely, elx] ** (self.penal - 1)) * ce
            compliance_history.append(c)
            
            # 4. Filter sensitivities to prevent checkerboard patterns <- eq. 3.2
            dc_flat = dc.flatten()
            dc_filtered = (H @ (rho.flatten() * dc_flat)) / (Hs * np.maximum(rho.flatten(), 1e-4))
            dc_filtered = dc_filtered.reshape((nely, nelx))
            
            # 5. Optimality Criteria (OC) update
            l1, l2 = 0.0, 1e9
            move = 0.2
            rho_new = rho.copy()
            while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                B = -dc_filtered / lmid
                rho_cand = np.clip(rho * np.sqrt(np.maximum(B, 1e-10)), rho - move, rho + move)
                rho_cand = np.clip(rho_cand, 0.001, 1.0)
                if np.mean(rho_cand) > self.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid
                rho_new = rho_cand
                
            rho = rho_new
            
        return rho, compliance_history

if __name__ == "__main__":
    print("--- W11: Topology Optimization SIMP Sanity Check ---")
    opt = TopologyOptimizerSIMP(nelx=12, nely=6, volfrac=0.4, penal=3.0)
    final_density, c_hist = opt.optimize(max_iters=15)
    
    print(f"Initial Compliance: {c_hist[0]:.4f}")
    print(f"Final Optimized Compliance: {c_hist[-1]:.4f}")
    print(f"Final Mean Density: {np.mean(final_density):.4f} (Target: 0.40)")
    
    # Assert compliance decreased
    assert c_hist[-1] < c_hist[0], "Compliance must decrease during optimization"
    # Assert volume constraint respected
    assert np.isclose(np.mean(final_density), 0.4, atol=0.02)
    # Binary-tending density (penalization drives values toward 0 or 1)
    print("W11 Sanity Check: PASSED cleanly.")

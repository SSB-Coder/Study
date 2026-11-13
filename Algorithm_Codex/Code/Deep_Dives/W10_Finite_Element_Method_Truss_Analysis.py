"""
W10. Finite Element Method (FEM) — 2D Truss Structural Analysis
From-scratch implementation in pure NumPy.
"""

import numpy as np

class TrussFEM:
    """
    2D Pin-Jointed Elastic Truss Finite Element Solver.
    Assembles global stiffness matrix K from bar elements,
    solves Ku = f for nodal displacements, and computes element axial stresses.
    """
    
    def __init__(self, nodes: np.ndarray, elements: list, E: float = 210e9, A: float = 1e-4):
        """
        nodes: (N, 2) array of nodal (x, y) coordinates.
        elements: list of (node_i, node_j) tuples.
        E: Young's Modulus in Pa (default steel: 210 GPa).
        A: Cross-sectional area in m^2.
        """
        self.nodes = np.array(nodes, dtype=float)
        self.elements = elements
        self.E = E
        self.A = A
        self.n_nodes = len(nodes)
        self.dofs = 2 * self.n_nodes # 2 DOFs (u_x, u_y) per node

    def assemble_stiffness_matrix(self):
        """
        Assembles global stiffness matrix K of size (2N, 2N) <- eq. 3.1.
        """
        K = np.zeros((self.dofs, self.dofs), dtype=float)
        
        for n1, n2 in self.elements:
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            L = np.sqrt(dx**2 + dy**2)
            if L < 1e-8:
                raise ValueError("Degenerate element of zero length")
                
            c = dx / L
            s = dy / L
            
            # Element stiffness in global coordinates <- eq. 3.2
            k_const = (self.E * self.A) / L
            k_elem = k_const * np.array([
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s]
            ])
            
            # Scatter to global K
            idx = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for i in range(4):
                for j in range(4):
                    K[idx[i], idx[j]] += k_elem[i, j]
                    
        return K

    def solve(self, forces: np.ndarray, fixed_dofs: list):
        """
        Solves Ku = f with prescribed Dirichlet displacement boundary conditions u[fixed_dofs] = 0.
        Uses pure NumPy Gaussian elimination / LU partitioning.
        """
        K = self.assemble_stiffness_matrix()
        f = np.array(forces, dtype=float)
        u = np.zeros(self.dofs, dtype=float)
        
        free_dofs = [d for d in range(self.dofs) if d not in fixed_dofs]
        
        # Submatrix partitioning <- eq. 3.3
        K_free = K[np.ix_(free_dofs, free_dofs)]
        f_free = f[free_dofs]
        
        # Solve reduced linear system via Gaussian elimination
        # Hand-rolled forward/backward solve
        n_free = len(free_dofs)
        Aug = np.hstack([K_free, f_free[:, np.newaxis]])
        for i in range(n_free):
            pivot = i + np.argmax(np.abs(Aug[i:, i]))
            if pivot != i:
                Aug[[i, pivot]] = Aug[[pivot, i]]
            Aug[i] /= Aug[i, i]
            for j in range(n_free):
                if i != j:
                    Aug[j] -= Aug[j, i] * Aug[i]
        u_free = Aug[:, -1]
        u[free_dofs] = u_free
        
        # Compute element axial forces & stresses
        stresses = []
        for n1, n2 in self.elements:
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            c = (p2[0] - p1[0]) / L
            s = (p2[1] - p1[1]) / L
            u_elem = np.array([u[2*n1], u[2*n1+1], u[2*n2], u[2*n2+1]])
            # Delta L = T * u_elem
            delta_L = np.dot(np.array([-c, -s, c, s]), u_elem)
            strain = delta_L / L
            stress = self.E * strain # Hooke's Law
            stresses.append(stress)
            
        return u, np.array(stresses)

if __name__ == "__main__":
    print("--- W10: 2D Truss FEM Sanity Check ---")
    # Simple triangular 3-bar truss:
    # Nodes: 0=(0, 0) [fixed], 1=(4, 0) [roller y=0], 2=(2, 3) [loaded downward]
    nodes = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [2.0, 3.0]
    ])
    elements = [(0, 1), (0, 2), (1, 2)]
    fem = TrussFEM(nodes, elements, E=200e9, A=1e-3)
    
    # 10 kN downward vertical load on top node 2: DOF 5 (2 * 2 + 1)
    forces = np.zeros(6)
    forces[5] = -10000.0 # -10 kN
    
    # Boundary conditions:
    # Node 0 fixed in x and y (DOFs 0, 1)
    # Node 1 fixed in y (DOF 3)
    fixed = [0, 1, 3]
    
    u, stresses = fem.solve(forces, fixed)
    print(f"Displacement at loaded node 2: ux={u[4]*1e6:.2f} um, uy={u[5]*1e6:.2f} um")
    print(f"Bar stresses (MPa): {stresses / 1e6}")
    
    # Assert physical sanity:
    # 1. Downward load must cause negative vertical displacement at node 2
    assert u[5] < 0.0, "Vertical displacement must be downward"
    # 2. Left and right diagonal bars (elements 1 and 2) must be under compression (negative stress)
    assert stresses[1] < 0.0, "Element 1 must be under compression"
    assert stresses[2] < 0.0, "Element 2 must be under compression"
    # 3. Bottom bar (element 0) must be under tension (positive stress)
    assert stresses[0] > 0.0, "Bottom bar must be under tension"
    print("W10 Sanity Check: PASSED cleanly.")

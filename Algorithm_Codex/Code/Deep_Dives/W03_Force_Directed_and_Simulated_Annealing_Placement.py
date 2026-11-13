"""
W03. Force-Directed & Simulated-Annealing Component Placement
From-scratch implementation in pure NumPy.
"""

import numpy as np

class ComponentPlacer:
    """
    Simulates component placement on a 2D chip/board die.
    Combines analytical force-directed relaxation with simulated annealing
    to minimize Half-Perimeter Wirelength (HPWL) while discouraging overlaps.
    """
    
    def __init__(self, n_components: int, die_width: float = 100.0, die_height: float = 100.0):
        self.n = n_components
        self.die_w = die_width
        self.die_h = die_height
        self.positions = np.random.uniform(10.0, 90.0, size=(n_components, 2))
        self.fixed = np.zeros(n_components, dtype=bool)

    def compute_hpwl(self, nets: list) -> float:
        """
        Half-Perimeter Wirelength (HPWL) metric <- eq. 3.1:
        HPWL(net) = (max(x) - min(x)) + (max(y) - min(y))
        """
        total_hpwl = 0.0
        for net in nets:
            if len(net) <= 1:
                continue
            pts = self.positions[net]
            dx = np.max(pts[:, 0]) - np.min(pts[:, 0])
            dy = np.max(pts[:, 1]) - np.min(pts[:, 1])
            total_hpwl += (dx + dy)
        return float(total_hpwl)

    def force_directed_relaxation(self, nets: list, n_iters: int = 50, k_spring: float = 0.1, k_repulse: float = 50.0):
        """
        Force-directed relaxation:
        - Hooke's Law attractive spring forces along net connections.
        - Coulomb-like repulsive forces between overlapping/adjacent components.
        """
        for _ in range(n_iters):
            forces = np.zeros_like(self.positions)
            
            # Attractive forces along nets <- eq. 3.2
            for net in nets:
                for i in range(len(net)):
                    u = net[i]
                    for j in range(i + 1, len(net)):
                        v = net[j]
                        delta = self.positions[v] - self.positions[u]
                        dist = np.linalg.norm(delta) + 1e-4
                        f_attr = k_spring * delta
                        forces[u] += f_attr
                        forces[v] -= f_attr
                        
            # Repulsive forces between all component pairs
            for u in range(self.n):
                for v in range(u + 1, self.n):
                    delta = self.positions[u] - self.positions[v]
                    dist = np.linalg.norm(delta) + 1e-4
                    if dist < 20.0: # Repulsion zone
                        f_rep = (k_repulse / (dist**2)) * (delta / dist)
                        forces[u] += f_rep
                        forces[v] -= f_rep
                        
            # Update positions
            for i in range(self.n):
                if not self.fixed[i]:
                    self.positions[i] += np.clip(forces[i], -5.0, 5.0)
                    self.positions[i, 0] = np.clip(self.positions[i, 0], 5.0, self.die_w - 5.0)
                    self.positions[i, 1] = np.clip(self.positions[i, 1], 5.0, self.die_h - 5.0)

    def simulated_annealing_optimize(self, nets: list, initial_temp: float = 100.0, cooling: float = 0.95, n_steps: int = 500):
        """
        Simulated Annealing refinement to escape local minima in HPWL.
        """
        curr_cost = self.compute_hpwl(nets)
        best_cost = curr_cost
        best_pos = self.positions.copy()
        temp = initial_temp
        
        for step in range(n_steps):
            # Pick random movable component and perturb position
            idx = np.random.randint(0, self.n)
            if self.fixed[idx]:
                continue
                
            old_pos = self.positions[idx].copy()
            delta = np.random.uniform(-10.0, 10.0, size=2) * (temp / initial_temp)
            self.positions[idx] += delta
            self.positions[idx, 0] = np.clip(self.positions[idx, 0], 5.0, self.die_w - 5.0)
            self.positions[idx, 1] = np.clip(self.positions[idx, 1], 5.0, self.die_h - 5.0)
            
            new_cost = self.compute_hpwl(nets)
            cost_diff = new_cost - curr_cost
            
            # Metropolis acceptance criterion <- eq. 3.3
            if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / max(temp, 1e-4)):
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_pos = self.positions.copy()
            else:
                self.positions[idx] = old_pos
                
            temp *= cooling
            
        self.positions = best_pos
        return best_cost

if __name__ == "__main__":
    print("--- W03: Force-Directed & SA Placement Sanity Check ---")
    np.random.seed(42)
    placer = ComponentPlacer(n_components=8, die_width=100.0, die_height=100.0)
    
    # Define netlist: pairs and triplets of connected components
    nets = [
        [0, 1], [1, 2], [2, 3], # chain 0-1-2-3
        [4, 5], [5, 6], [6, 7], # chain 4-5-6-7
        [1, 5], [2, 6]          # cross-connects
    ]
    
    initial_hpwl = placer.compute_hpwl(nets)
    print(f"Initial Random HPWL: {initial_hpwl:.2f}")
    
    # Run force-directed relaxation
    placer.force_directed_relaxation(nets, n_iters=40)
    fd_hpwl = placer.compute_hpwl(nets)
    print(f"Post Force-Directed HPWL: {fd_hpwl:.2f}")
    
    # Run simulated annealing refinement
    final_hpwl = placer.simulated_annealing_optimize(nets, n_steps=300)
    print(f"Final Simulated Annealing HPWL: {final_hpwl:.2f}")
    
    assert final_hpwl < initial_hpwl, "Optimization must improve wirelength"
    print("W03 Sanity Check: PASSED cleanly.")

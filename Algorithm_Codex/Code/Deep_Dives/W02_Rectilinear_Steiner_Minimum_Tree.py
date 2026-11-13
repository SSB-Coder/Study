"""
W02. Rectilinear Steiner Minimum Tree (RSMT) Heuristic
From-scratch implementation in pure NumPy.
"""

import numpy as np

class RectilinearSteinerTree:
    """
    Computes a Rectilinear Steiner Minimum Tree (RSMT) connecting a set of pins.
    Uses Hanan grid discretization and minimum spanning tree (MST) heuristics.
    Steiner points reduce total wirelength compared to Rectilinear MST (RMST).
    """
    
    @staticmethod
    def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]))

    @classmethod
    def compute_rmst(cls, points: np.ndarray):
        """
        Computes the Rectilinear Minimum Spanning Tree (RMST) using Prim's algorithm.
        Returns total wirelength and list of edge pairs (idx1, idx2).
        """
        n = len(points)
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        edges = []
        total_length = 0.0
        
        for _ in range(n - 1):
            min_dist = float('inf')
            best_u, best_v = -1, -1
            
            for u in range(n):
                if in_tree[u]:
                    for v in range(n):
                        if not in_tree[v]:
                            d = cls.manhattan_distance(points[u], points[v])
                            if d < min_dist:
                                min_dist = d
                                best_u, best_v = u, v
                                
            in_tree[best_v] = True
            edges.append((best_u, best_v))
            total_length += min_dist
            
        return total_length, edges

    @classmethod
    def compute_rsmt(cls, pins: np.ndarray):
        """
        Computes RSMT by greedily evaluating candidate Steiner points from the Hanan grid.
        Hanan's Theorem: At least one optimal RSMT exists on the grid formed by
        the horizontal and vertical lines passing through all terminal pins.
        """
        pins = np.array(pins, dtype=float)
        n = len(pins)
        if n <= 2:
            rmst_len, edges = cls.compute_rmst(pins)
            return rmst_len, pins.tolist(), edges

        # Generate Hanan grid points <- eq. 3.1
        xs = np.unique(pins[:, 0])
        ys = np.unique(pins[:, 1])
        hanan_points = [np.array([x, y]) for x in xs for y in ys]
        
        current_points = list(pins)
        best_length, best_edges = cls.compute_rmst(np.array(current_points))
        
        # Iterative greedy insertion of Steiner points
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_cand_len = best_length
            
            for cand in hanan_points:
                # Check if already in set
                if any(np.allclose(cand, p) for p in current_points):
                    continue
                test_points = np.array(current_points + [cand])
                test_len, _ = cls.compute_rmst(test_points)
                if test_len < best_cand_len - 1e-5:
                    best_cand_len = test_len
                    best_candidate = cand
                    
            if best_candidate is not None:
                current_points.append(best_candidate)
                best_length = best_cand_len
                improved = True
                
        _, final_edges = cls.compute_rmst(np.array(current_points))
        return best_length, current_points, final_edges

if __name__ == "__main__":
    print("--- W02: Rectilinear Steiner Minimum Tree Sanity Check ---")
    # 3 pins forming an L-shape corner: (0, 0), (0, 2), (2, 0)
    # RMST needs length = 2 + 2 = 4.
    # Steiner point at (0, 0) gives length 4, but for a diagonal triangle:
    # Pins: (0, 2), (2, 0), (2, 2) -> RMST = 4, Steiner at (2, 2) is length 4.
    # 4 corner pins: (0, 0), (0, 4), (4, 0), (4, 4)
    pins = np.array([
        [0.0, 0.0],
        [0.0, 4.0],
        [4.0, 0.0],
        [4.0, 4.0]
    ])
    rmst_len, _ = RectilinearSteinerTree.compute_rmst(pins)
    rsmt_len, all_pts, edges = RectilinearSteinerTree.compute_rsmt(pins)
    
    print(f"Terminal Pins: {len(pins)}")
    print(f"RMST Length (no Steiner points): {rmst_len:.2f}")
    print(f"RSMT Length (with Steiner points): {rsmt_len:.2f}")
    print(f"Steiner points added: {len(all_pts) - len(pins)}")
    
    assert rsmt_len <= rmst_len, "RSMT wirelength cannot exceed RMST"
    # For a square of side 4, perimeter is 16, RMST is 12, RSMT is 12 or better
    assert rsmt_len <= 12.0
    print("W02 Sanity Check: PASSED cleanly.")

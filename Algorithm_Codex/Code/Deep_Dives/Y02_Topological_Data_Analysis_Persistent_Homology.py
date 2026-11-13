"""
Y02: Topological Data Analysis (TDA) & Persistent Homology
Pure NumPy implementation of Vietoris-Rips complex construction, boundary matrices,
standard column-reduction over Z_2, persistence barcodes, and Betti numbers (beta_0, beta_1).
"""

import numpy as np

class VietorisRipsTDA:
    """
    Computes persistent homology of a point cloud via the Vietoris-Rips filtration.
    """
    def __init__(self, max_dim: int = 1):
        self.max_dim = max_dim

    def fit_transform(self, points: np.ndarray, max_edge_length: float | None = None) -> list[tuple[int, float, float]]:
        """
        Builds Rips filtration, computes boundary matrix, and reduces over Z_2.
        Returns list of (dim, birth, death) persistence intervals.
        """
        n_points = len(points)
        # 1. Compute pairwise Euclidean distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        
        if max_edge_length is None:
            max_edge_length = float(np.max(dist_matrix))

        # 2. Collect simplices with their filtration birth values (diameter)
        # 0-simplices (vertices)
        simplices = [((i,), 0.0) for i in range(n_points)]
        
        # 1-simplices (edges)
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = dist_matrix[i, j]
                if d <= max_edge_length:
                    edges.append(((i, j), d))
        simplices.extend(edges)
        
        # 2-simplices (triangles) if max_dim >= 1
        if self.max_dim >= 1:
            triangles = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if dist_matrix[i, j] > max_edge_length:
                        continue
                    for k in range(j + 1, n_points):
                        d_max = max(dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k])
                        if d_max <= max_edge_length:
                            triangles.append(((i, j, k), d_max))
            simplices.extend(triangles)

        # 3. Sort simplices by filtration value (diameter), breaking ties by dimension
        simplices.sort(key=lambda x: (x[1], len(x[0])))
        simplex_to_idx = {s[0]: idx for idx, s in enumerate(simplices)}
        m = len(simplices)

        # 4. Construct sparse boundary matrix columns (list of 1-entry row indices)
        # Over Z_2, boundary of [v0, ..., vk] is sum of [v0, ..., \hat{vi}, ..., vk]
        boundary_cols = []
        for s, _ in simplices:
            k = len(s) - 1
            if k == 0:
                boundary_cols.append(set())
            else:
                faces = set()
                for i in range(len(s)):
                    face = tuple(s[:i] + s[i+1:])
                    if face in simplex_to_idx:
                        faces.add(simplex_to_idx[face])
                boundary_cols.append(faces)

        # 5. Standard Column-Reduction Algorithm over Z_2
        # Low(j) is the maximum row index in column j
        low_to_col = {}
        reduced_cols = [set(col) for col in boundary_cols]

        for j in range(m):
            while reduced_cols[j]:
                low_j = max(reduced_cols[j])
                if low_j in low_to_col:
                    # XOR addition over Z_2: symmetric difference
                    k = low_to_col[low_j]
                    reduced_cols[j] ^= reduced_cols[k]
                else:
                    low_to_col[low_j] = j
                    break

        # 6. Extract Persistence Pairs
        intervals = []
        paired_indices = set()

        for low_val, j in low_to_col.items():
            i = low_val
            paired_indices.add(i)
            paired_indices.add(j)
            dim = len(simplices[i][0]) - 1
            birth = simplices[i][1]
            death = simplices[j][1]
            if death > birth:
                intervals.append((dim, birth, death))

        # Essential classes (infinite death: reduced to 0 and never killed by any higher simplex)
        for i in range(m):
            if not reduced_cols[i] and i not in low_to_col:
                dim = len(simplices[i][0]) - 1
                if dim <= self.max_dim:
                    intervals.append((dim, simplices[i][1], float("inf")))

        return sorted(intervals, key=lambda x: (x[0], x[1]))

if __name__ == "__main__":
    print("--- Y02: Topological Data Analysis (Persistent Homology) Sanity Check ---")
    
    # Generate noisy circle (annulus) in 2D: R = 1.0, noise = 0.04
    np.random.seed(42)
    n_samples = 20
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    noise = np.random.uniform(-0.03, 0.03, size=n_samples)
    r = 1.0 + noise
    circle_points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    tda = VietorisRipsTDA(max_dim=1)
    # Threshold 1.85 allows triangles to form across sqrt(3) ~ 1.732 to cap the 1D cycle
    intervals = tda.fit_transform(circle_points, max_edge_length=1.85)
    
    # Analyze Betti-0 and Betti-1 features
    betti_0 = [it for it in intervals if it[0] == 0]
    betti_1 = [it for it in intervals if it[0] == 1]
    
    print(f"Computed {len(betti_0)} H0 intervals and {len(betti_1)} H1 intervals.")
    
    # 1. Connected components: exactly one essential component surviving to infinity
    inf_h0 = [it for it in betti_0 if np.isinf(it[2])]
    assert len(inf_h0) == 1, "There must be exactly one connected component at large filtration scale"
    
    # 2. 1D Loop detection: Circle must produce at least one prominent H1 feature with long lifetime
    assert len(betti_1) > 0, "Topological circle must produce 1D persistent loop"
    lifetimes = [it[2] - it[1] for it in betti_1 if not np.isinf(it[2])]
    max_h1_lifetime = max(lifetimes) if lifetimes else 0.0
    print(f"Max 1D Loop Persistence Lifetime: {max_h1_lifetime:.4f}")
    assert max_h1_lifetime > 0.80, "Circle must exhibit robust topological persistence separating signal from noise"
    
    print("Y02 Sanity Check: PASSED cleanly.")

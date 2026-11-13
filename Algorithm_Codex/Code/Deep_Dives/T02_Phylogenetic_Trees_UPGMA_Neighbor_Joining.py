"""
T02. Phylogenetic Tree Construction — UPGMA & Neighbor-Joining
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PhylogeneticTree:
    """
    Constructs evolutionary phylogenetic trees from pairwise genetic distance matrices.
    - UPGMA: Ultrametric distance clustering.
    - Neighbor-Joining (NJ): Star-decomposition additive tree reconstruction.
    """
    
    @staticmethod
    def upgma(distance_matrix: np.ndarray, taxa_names: list):
        """
        UPGMA (Unweighted Pair Group Method with Arithmetic Mean) <- eq. 3.1.
        Returns list of tree merges: (cluster1, cluster2, branch_height).
        """
        D = np.array(distance_matrix, dtype=float)
        taxa = list(taxa_names)
        cluster_sizes = {name: 1 for name in taxa}
        merges = []
        
        while len(taxa) > 1:
            n = len(taxa)
            # Find closest pair (min off-diagonal distance)
            min_dist = float('inf')
            best_i, best_j = -1, -1
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i, j] < min_dist:
                        min_dist = D[i, j]
                        best_i, best_j = i, j
                        
            t_i, t_j = taxa[best_i], taxa[best_j]
            branch_height = min_dist / 2.0
            new_cluster = f"({t_i}:{branch_height:.2f},{t_j}:{branch_height:.2f})"
            merges.append((t_i, t_j, branch_height))
            
            # Compute new distances via weighted arithmetic mean
            new_dists = []
            s_i = cluster_sizes[t_i]
            s_j = cluster_sizes[t_j]
            for k in range(n):
                if k != best_i and k != best_j:
                    t_k = taxa[k]
                    d_new_k = (s_i * D[best_i, k] + s_j * D[best_j, k]) / (s_i + s_j)
                    new_dists.append(d_new_k)
                    
            # Rebuild distance matrix
            keep_indices = [k for k in range(n) if k != best_i and k != best_j]
            D = D[np.ix_(keep_indices, keep_indices)]
            
            # Add new cluster row/col
            if len(new_dists) > 0:
                new_row = np.array(new_dists)[np.newaxis, :]
                D = np.vstack([D, new_row])
                col = np.append(new_dists, 0.0)[:, np.newaxis]
                D = np.hstack([D, col])
                
            # Update taxa lists
            cluster_sizes[new_cluster] = s_i + s_j
            taxa = [taxa[k] for k in keep_indices] + [new_cluster]
            
        return taxa[0], merges

    @staticmethod
    def neighbor_joining(distance_matrix: np.ndarray, taxa_names: list):
        """
        Saitou & Nei (1987) Neighbor-Joining (NJ) algorithm <- eq. 3.2.
        Computes Q-matrix to handle varying lineage evolutionary rates (non-ultrametric).
        """
        D = np.array(distance_matrix, dtype=float)
        taxa = list(taxa_names)
        edges = [] # list of (node_u, node_v, length)
        
        node_id_counter = 0
        
        while len(taxa) > 2:
            n = len(taxa)
            # 1. Total distance sums r_i = sum_k D[i, k]
            r = np.sum(D, axis=1)
            
            # 2. Compute Q-matrix <- eq. 3.2: Q(i, j) = (n - 2)*D(i, j) - r_i - r_j
            Q = np.zeros((n, n), dtype=float)
            min_q = float('inf')
            best_i, best_j = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    Q[i, j] = (n - 2) * D[i, j] - r[i] - r[j]
                    Q[j, i] = Q[i, j]
                    if Q[i, j] < min_q:
                        min_q = Q[i, j]
                        best_i, best_j = i, j
                        
            # 3. Branch lengths to new internal node u <- eq. 3.3
            t_i, t_j = taxa[best_i], taxa[best_j]
            delta_r = (r[best_i] - r[best_j]) / (n - 2)
            limb_i = 0.5 * D[best_i, best_j] + 0.5 * delta_r
            limb_j = 0.5 * D[best_i, best_j] - 0.5 * delta_r
            
            new_node = f"Internal_{node_id_counter}"
            node_id_counter += 1
            edges.append((t_i, new_node, max(0.0, limb_i)))
            edges.append((t_j, new_node, max(0.0, limb_j)))
            
            # 4. New distances to node u: D(u, k) = 0.5 * (D(i, k) + D(j, k) - D(i, j))
            new_dists = []
            for k in range(n):
                if k != best_i and k != best_j:
                    d_uk = 0.5 * (D[best_i, k] + D[best_j, k] - D[best_i, best_j])
                    new_dists.append(d_uk)
                    
            keep_indices = [k for k in range(n) if k != best_i and k != best_j]
            D = D[np.ix_(keep_indices, keep_indices)]
            if len(new_dists) > 0:
                D = np.vstack([D, np.array(new_dists)[np.newaxis, :]])
                D = np.hstack([D, np.append(new_dists, 0.0)[:, np.newaxis]])
                
            taxa = [taxa[k] for k in keep_indices] + [new_node]
            
        # Final edge connecting last two nodes
        edges.append((taxa[0], taxa[1], D[0, 1]))
        return edges

if __name__ == "__main__":
    print("--- T02: Phylogenetic Trees Sanity Check ---")
    # Distance matrix between 4 species: Human, Chimp, Gorilla, Orangutan
    taxa = ["Human", "Chimp", "Gorilla", "Orangutan"]
    dist_mat = np.array([
        [0.0, 1.0, 3.0, 5.0],
        [1.0, 0.0, 3.0, 5.0],
        [3.0, 3.0, 0.0, 5.0],
        [5.0, 5.0, 5.0, 0.0]
    ])
    
    # 1. Test UPGMA
    newick_tree, merges = PhylogeneticTree.upgma(dist_mat, taxa)
    print(f"UPGMA Newick Tree: {newick_tree}")
    print(f"UPGMA First Merge: {merges[0][0]} and {merges[0][1]} at height {merges[0][2]}")
    # Human and Chimp must merge first (distance = 1.0 -> height = 0.5)
    assert (merges[0][0] == "Human" and merges[0][1] == "Chimp") or \
           (merges[0][0] == "Chimp" and merges[0][1] == "Human")
    assert np.isclose(merges[0][2], 0.5)
    
    # 2. Test Neighbor-Joining
    nj_edges = PhylogeneticTree.neighbor_joining(dist_mat, taxa)
    print(f"Neighbor-Joining Edges: {len(nj_edges)}")
    for u, v, w in nj_edges:
        print(f"  {u} <---({w:.2f})---> {v}")
    assert len(nj_edges) == 5 # 4 leaves + 2 internal nodes = 5 edges in unrooted tree
    print("T02 Sanity Check: PASSED cleanly.")

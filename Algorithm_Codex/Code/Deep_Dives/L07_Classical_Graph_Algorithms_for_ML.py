"""
L07: Classical Graph Algorithms for ML (Dijkstra, PageRank, Spectral Clustering)
Pure NumPy implementation of graph foundations for machine learning:
1. Dijkstra's shortest path algorithm
2. Google's PageRank via Power Iteration
3. Normalized Spectral Clustering via symmetric graph Laplacian
"""

import numpy as np

def dijkstra_shortest_path(adj_matrix: np.ndarray, source: int) -> tuple:
    """
    Computes shortest path distances from source to all nodes on non-negative weighted graph.
    adj_matrix: (N, N) where 0 indicates no edge.
    """
    n = adj_matrix.shape[0]
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=bool)

    for _ in range(n):
        # Pick unvisited node with minimal distance
        unvisited_dists = np.where(visited, np.inf, dist)
        u = int(np.argmin(unvisited_dists))
        if np.isinf(unvisited_dists[u]):
            break
        visited[u] = True

        # Relax outgoing edges
        for v in range(n):
            weight = adj_matrix[u, v]
            if weight > 0 and not visited[v]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

    return dist


def pagerank(adj_matrix: np.ndarray, damping: float = 0.85, tol: float = 1e-7, max_iter: int = 150) -> np.ndarray:
    """
    Google's PageRank algorithm via Power Iteration on Google Matrix:
    G = d * M + (1 - d) / N * 1_N 1_N^T
    """
    n = adj_matrix.shape[0]
    out_degree = np.sum(adj_matrix, axis=1)

    # Column-stochastic transition matrix M: M[j, i] = 1 / out_degree(i)
    M = np.zeros((n, n))
    for i in range(n):
        if out_degree[i] > 0:
            M[:, i] = adj_matrix[i, :] / out_degree[i]
        else:
            M[:, i] = 1.0 / n # Dangling node redistribution

    # Google matrix G
    G = damping * M + ((1.0 - damping) / n) * np.ones((n, n))

    # Power iteration
    p = np.ones(n) / n
    for _ in range(max_iter):
        p_next = G @ p
        if np.sum(np.abs(p_next - p)) < tol:
            break
        p = p_next

    return p


class SpectralClustering:
    """
    Normalized Spectral Clustering:
    Constructs normalized symmetric Laplacian L_sym = I - D^-1/2 W D^-1/2,
    extracts bottom k eigenvectors, and runs Lloyd's k-means on normalized rows.
    """
    def __init__(self, n_clusters: int = 2, gamma: float = 1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.labels = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        # 1. Construct Gaussian affinity/adjacency matrix W
        dists_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        W = np.exp(-self.gamma * dists_sq)
        np.fill_diagonal(W, 0.0)

        # 2. Degree matrix D and normalized Laplacian L_sym = I - D^-1/2 W D^-1/2
        d = np.sum(W, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        D_inv_sqrt = np.diag(d_inv_sqrt)

        L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # 3. Eigendecomposition of symmetric Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        # Select eigenvectors corresponding to smallest k eigenvalues
        U = eigenvectors[:, :self.n_clusters] # Shape (n, k)

        # Normalize rows of U to unit length
        norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        T = U / norm

        # 4. Standard k-means on normalized eigenvector rows
        # Initialize centroids randomly
        centroids = T[np.random.choice(n, self.n_clusters, replace=False)]
        labels = np.zeros(n, dtype=int)

        for _ in range(20):
            # Assign points to closest centroid
            dists = np.sum((T[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
            labels = np.argmin(dists, axis=1)
            # Update centroids
            for c in range(self.n_clusters):
                if np.sum(labels == c) > 0:
                    centroids[c] = np.mean(T[labels == c], axis=0)

        self.labels = labels
        return labels


if __name__ == "__main__":
    print("--- L07: Classical Graph Algorithms for ML Sanity Check ---")
    np.random.seed(42)

    # 1. Test Dijkstra
    # Graph with 4 nodes:
    # 0 -> 1 (w=2), 0 -> 2 (w=5)
    # 1 -> 2 (w=1), 1 -> 3 (w=7)
    # 2 -> 3 (w=2)
    adj = np.zeros((4, 4))
    adj[0, 1] = 2; adj[0, 2] = 5
    adj[1, 2] = 1; adj[1, 3] = 7
    adj[2, 3] = 2

    dists = dijkstra_shortest_path(adj, source=0)
    print(f"Dijkstra Shortest Paths from Node 0: {dists}")
    # Shortest to 2: 0->1->2 (cost 3)
    # Shortest to 3: 0->1->2->3 (cost 5)
    assert np.isclose(dists[2], 3.0), f"Expected distance to 2 is 3.0, got {dists[2]}"
    assert np.isclose(dists[3], 5.0), f"Expected distance to 3 is 5.0, got {dists[3]}"

    # 2. Test PageRank
    # Hub graph: node 3 receives in-links from all nodes (0, 1, 2)
    web_adj = np.zeros((4, 4))
    web_adj[0, 3] = 1.0; web_adj[1, 3] = 1.0; web_adj[2, 3] = 1.0
    web_adj[3, 0] = 1.0 # Back-link

    ranks = pagerank(web_adj, damping=0.85)
    print(f"PageRank Probabilities: {ranks.round(4)}")
    top_node = int(np.argmax(ranks))
    assert top_node == 3, f"Node 3 must receive highest PageRank authority, got node {top_node}"

    # 3. Test Spectral Clustering on Concentric Circles (fails with linear k-means)
    n_ring = 30
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    inner_ring = np.column_stack([np.cos(theta), np.sin(theta)])
    outer_ring = np.column_stack([3.0 * np.cos(theta), 3.0 * np.sin(theta)])
    X_circles = np.vstack([inner_ring, outer_ring])

    sc = SpectralClustering(n_clusters=2, gamma=2.0)
    cluster_labels = sc.fit_predict(X_circles)

    # Check that inner ring and outer ring receive separate cluster IDs
    inner_labels = cluster_labels[:n_ring]
    outer_labels = cluster_labels[n_ring:]
    pure_inner = len(np.unique(inner_labels)) == 1
    pure_outer = len(np.unique(outer_labels)) == 1
    print(f"Spectral Clustering Ring Separation: Inner Pure={pure_inner}, Outer Pure={pure_outer}")
    assert pure_inner and pure_outer and inner_labels[0] != outer_labels[0], "Spectral clustering must perfectly separate non-linear concentric rings"

    print("L07 Sanity Check: PASSED cleanly.")

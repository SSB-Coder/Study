"""
C02. Hierarchical Agglomerative Clustering
From-scratch implementation in pure NumPy.
"""

import numpy as np

class AgglomerativeClusteringScratch:
    """Bottom-up hierarchical clustering supporting single, complete, average, and ward linkage."""
    def __init__(self, n_clusters: int = 2, linkage: str = "average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.merge_history_ = [] # (cluster1, cluster2, distance, new_size)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        N = len(X)
        
        # Track cluster membership: dict mapping cluster_id -> list of sample indices
        clusters = {i: [i] for i in range(N)}
        next_cluster_id = N

        # Pairwise distance matrix
        dist_matrix = np.full((2 * N, 2 * N), np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        active_clusters = list(range(N))

        while len(active_clusters) > self.n_clusters:
            # Find closest pair among active clusters
            best_d = np.inf
            pair = (None, None)

            for i_idx in range(len(active_clusters)):
                c1 = active_clusters[i_idx]
                for j_idx in range(i_idx + 1, len(active_clusters)):
                    c2 = active_clusters[j_idx]
                    d = dist_matrix[c1, c2]
                    if d < best_d:
                        best_d = d
                        pair = (c1, c2)

            c1, c2 = pair
            # Merge c1 and c2 into new cluster
            new_cluster = next_cluster_id
            next_cluster_id += 1
            new_members = clusters[c1] + clusters[c2]
            clusters[new_cluster] = new_members
            self.merge_history_.append((c1, c2, best_d, len(new_members)))

            active_clusters.remove(c1)
            active_clusters.remove(c2)

            # Compute distance from new_cluster to remaining active clusters
            for c_other in active_clusters:
                pts_new = X[new_members]
                pts_other = X[clusters[c_other]]
                
                # Pairwise distances between all points in both clusters
                diffs = pts_new[:, np.newaxis, :] - pts_other[np.newaxis, :, :]
                pairwise_dists = np.linalg.norm(diffs, axis=-1)

                if self.linkage == "single":
                    # Single linkage: min distance <- eq. 3.1
                    d_val = np.min(pairwise_dists)
                elif self.linkage == "complete":
                    # Complete linkage: max distance <- eq. 3.1
                    d_val = np.max(pairwise_dists)
                elif self.linkage == "average":
                    # Average linkage: mean distance <- eq. 3.1
                    d_val = np.mean(pairwise_dists)
                elif self.linkage == "ward":
                    # Ward's criterion: variance increase <- eq. 3.2
                    n1, n2 = len(new_members), len(clusters[c_other])
                    mu1 = np.mean(pts_new, axis=0)
                    mu2 = np.mean(pts_other, axis=0)
                    d_val = np.sqrt((n1 * n2 / (n1 + n2)) * np.sum((mu1 - mu2) ** 2))
                else:
                    raise ValueError(f"Unknown linkage {self.linkage}")

                dist_matrix[new_cluster, c_other] = d_val
                dist_matrix[c_other, new_cluster] = d_val

            active_clusters.append(new_cluster)

        # Assign final flat labels
        self.labels_ = np.zeros(N, dtype=int)
        for label_idx, c_id in enumerate(active_clusters):
            for sample_idx in clusters[c_id]:
                self.labels_[sample_idx] = label_idx
        return self

if __name__ == "__main__":
    print("--- C02: Hierarchical Clustering Sanity Check ---")
    np.random.seed(42)
    # 2 well separated clusters
    X_hc = np.vstack([np.random.randn(20, 2) + np.array([-3, 0]), np.random.randn(20, 2) + np.array([3, 0])])
    y_true = np.array([0]*20 + [1]*20)

    hac = AgglomerativeClusteringScratch(n_clusters=2, linkage="average").fit(X_hc)
    acc = max(np.mean(hac.labels_ == y_true), np.mean(hac.labels_ == (1 - y_true)))
    print(f"Hierarchical Clustering Accuracy: {acc * 100:.2f}%")
    print(f"Number of merges executed: {len(hac.merge_history_)}")
    assert acc > 0.95
    assert len(hac.merge_history_) == (40 - 2)
    print("C02 Sanity Check: PASSED cleanly.")

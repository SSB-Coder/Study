"""
M01: Hierarchical Navigable Small World (HNSW)
Pure NumPy implementation of multi-layer skip-graph approximate nearest neighbor search.
Powers modern vector databases and high-dimensional semantic search.
"""

import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Set


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) graph for sub-linear similarity search.
    """
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 64, ef_search: int = 32, ml: Optional[float] = None):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # Layer 0 can hold twice as many neighbors
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml if ml is not None else 1.0 / np.log(M)
        
        # Multi-layer graph storage: layer_id -> {node_id -> set(neighbor_ids)}
        self.graphs: List[Dict[int, Set[int]]] = []
        self.data: Dict[int, np.ndarray] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1

    def _distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Euclidean L2 metric distance."""
        diff = u - v
        return float(np.dot(diff, diff))

    def _assign_random_level(self) -> int:
        """Draws layer height from exponential decay distribution."""
        unif = np.random.uniform(1e-9, 1.0)
        return int(np.floor(-np.log(unif) * self.ml))

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        """
        Greedy beam search on a single layer graph.
        Returns list of (distance, node_id) sorted ascending.
        """
        v_dist = {ep: self._distance(query, self.data[ep]) for ep in entry_points}
        visited = set(entry_points)
        
        # Min-heap of candidates to explore: (dist, node_id)
        candidates = [(v_dist[ep], ep) for ep in entry_points]
        heapq.heapify(candidates)
        
        # Max-heap of current best results: (-dist, node_id)
        w_results = [(-v_dist[ep], ep) for ep in entry_points]
        heapq.heapify(w_results)
        
        graph_level = self.graphs[level]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -w_results[0][0]
            
            if c_dist > furthest_dist:
                break
                
            for neighbor in graph_level.get(c_id, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._distance(query, self.data[neighbor])
                    furthest_dist = -w_results[0][0]
                    
                    if n_dist < furthest_dist or len(w_results) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(w_results, (-n_dist, neighbor))
                        if len(w_results) > ef:
                            heapq.heappop(w_results)
                            
        # Sort ascending by distance
        res = [(-item[0], item[1]) for item in w_results]
        res.sort(key=lambda x: x[0])
        return res

    def insert(self, node_id: int, vector: np.ndarray) -> None:
        """Inserts a new vector into the HNSW index."""
        self.data[node_id] = vector.astype(float)
        level = self._assign_random_level()

        # Ensure graphs list is deep enough
        while len(self.graphs) <= max(level, self.max_level):
            self.graphs.append({})

        # Initialize neighbor sets for the new node on all its active layers
        for l in range(level + 1):
            if node_id not in self.graphs[l]:
                self.graphs[l][node_id] = set()

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        curr_ep = self.entry_point
        # Phase 1: Zoom through upper layers to find closest entry point at level
        for l in range(self.max_level, level, -1):
            w = self._search_layer(vector, [curr_ep], ef=1, level=l)
            curr_ep = w[0][1]

        # Phase 2: From min(level, max_level) down to 0, connect bidirectional neighbors
        for l in range(min(level, self.max_level), -1, -1):
            m_max = self.M0 if l == 0 else self.M
            w = self._search_layer(vector, [curr_ep], ef=self.ef_construction, level=l)
            
            # Select neighbors (closest up to m_max)
            neighbors = [node for _, node in w[:m_max]]
            self.graphs[l][node_id] = set(neighbors)
            
            # Reciprocal bidirectional links
            for nbr in neighbors:
                if nbr not in self.graphs[l]:
                    self.graphs[l][nbr] = set()
                self.graphs[l][nbr].add(node_id)
                if len(self.graphs[l][nbr]) > m_max:
                    # Prune furthest neighbor
                    nbr_list = list(self.graphs[l][nbr])
                    nbr_list.sort(key=lambda x: self._distance(self.data[nbr], self.data[x]))
                    self.graphs[l][nbr] = set(nbr_list[:m_max])

            curr_ep = w[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, int]]:
        """Searches for top-k nearest neighbors."""
        if self.entry_point is None:
            return []

        curr_ep = self.entry_point
        # Top-down zoom
        for l in range(self.max_level, 0, -1):
            w = self._search_layer(query, [curr_ep], ef=1, level=l)
            curr_ep = w[0][1]

        # Layer 0 search
        ef = max(self.ef_search, k)
        results = self._search_layer(query, [curr_ep], ef=ef, level=0)
        return results[:k]


if __name__ == "__main__":
    np.random.seed(42)
    n_points = 250
    dim = 16
    X = np.random.randn(n_points, dim)

    hnsw = HNSW(dim=dim, M=8, ef_construction=32, ef_search=32)
    for i in range(n_points):
        hnsw.insert(i, X[i])

    # Run query benchmarks
    n_queries = 20
    k = 5
    recalls = []

    for q_idx in range(n_queries):
        query = np.random.randn(dim)
        
        # Ground truth exact brute force
        dists = np.sum((X - query) ** 2, axis=1)
        exact_top_k = set(np.argsort(dists)[:k])
        
        # HNSW search
        hnsw_results = hnsw.search(query, k=k)
        hnsw_top_k = set([node_id for _, node_id in hnsw_results])
        
        recall = len(exact_top_k.intersection(hnsw_top_k)) / k
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    print(f"HNSW Top-{k} Average Recall across {n_queries} queries: {avg_recall * 100:.2f}%")
    assert avg_recall >= 0.85, f"Expected HNSW recall >= 85%, got {avg_recall * 100:.2f}%"
    print("M01 Sanity Check: PASSED cleanly.")

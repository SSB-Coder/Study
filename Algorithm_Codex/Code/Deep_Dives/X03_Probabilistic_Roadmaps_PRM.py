"""
X03. Probabilistic Roadmaps (PRM) for Multi-Query Motion Planning
From-scratch implementation in pure NumPy.
"""

import numpy as np
import heapq

class PRMPlanner:
    """
    Probabilistic Roadmap (PRM) motion planner.
    Constructs an undirected roadmap graph in continuous C-space during an offline phase,
    enabling ultra-fast multi-query online planning via A* / Dijkstra.
    """
    
    def __init__(self, bounds: list, n_samples: int = 150, k_neighbors: int = 8):
        self.bounds = bounds # [(min_x, max_x), (min_y, max_y)]
        self.n_samples = n_samples
        self.k = k_neighbors
        self.points = []     # List of (x, y)
        self.adj = {}        # node_id -> list of (neighbor_id, distance)
        self.obstacles = []

    def add_obstacle(self, ox: float, oy: float, radius: float):
        self.obstacles.append((ox, oy, radius))

    def _is_point_valid(self, x: float, y: float) -> bool:
        for ox, oy, r in self.obstacles:
            if (x - ox)**2 + (y - oy)**2 <= r**2:
                return False
        return True

    def _is_edge_valid(self, p1: tuple, p2: tuple, n_checks: int = 10) -> bool:
        for alpha in np.linspace(0.0, 1.0, n_checks):
            px = p1[0] + alpha * (p2[0] - p1[0])
            py = p1[1] + alpha * (p2[1] - p1[1])
            if not self._is_point_valid(px, py):
                return False
        return True

    def build_roadmap(self):
        """
        Phase 1 (Learning): Uniformly samples collision-free points and connects k-NN graph <- eq. 3.1.
        """
        self.points = []
        # Sample points in free space
        while len(self.points) < self.n_samples:
            rx = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            ry = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            if self._is_point_valid(rx, ry):
                self.points.append((rx, ry))
                
        self.adj = {i: [] for i in range(len(self.points))}
        
        # Connect to k-nearest neighbors
        for i in range(len(self.points)):
            p_i = self.points[i]
            dists = []
            for j in range(len(self.points)):
                if i != j:
                    p_j = self.points[j]
                    d = np.sqrt((p_i[0] - p_j[0])**2 + (p_i[1] - p_j[1])**2)
                    dists.append((d, j))
            dists.sort(key=lambda x: x[0])
            
            # Try connecting to k nearest
            for d, j in dists[:self.k]:
                if self._is_edge_valid(p_i, self.points[j]):
                    if (j, d) not in self.adj[i]:
                        self.adj[i].append((j, d))
                    if (i, d) not in self.adj[j]:
                        self.adj[j].append((i, d))

    def query(self, start: tuple, goal: tuple):
        """
        Phase 2 (Query): Connects start and goal to roadmap, solves shortest path via Dijkstra.
        """
        if not self._is_point_valid(start[0], start[1]) or not self._is_point_valid(goal[0], goal[1]):
            return None
            
        start_id = len(self.points)
        goal_id = len(self.points) + 1
        
        # Temporary adjacency copy
        temp_adj = {k: list(v) for k, v in self.adj.items()}
        temp_adj[start_id] = []
        temp_adj[goal_id] = []
        
        # Connect start and goal to roadmap points
        for pt_id, pt in [(start_id, start), (goal_id, goal)]:
            dists = []
            for i in range(len(self.points)):
                p = self.points[i]
                d = np.sqrt((p[0] - pt[0])**2 + (p[1] - pt[1])**2)
                dists.append((d, i))
            dists.sort(key=lambda x: x[0])
            for d, i in dists[:self.k]:
                if self._is_edge_valid(pt, self.points[i]):
                    temp_adj[pt_id].append((i, d))
                    temp_adj[i].append((pt_id, d))
                    
        # Dijkstra search <- eq. 3.2
        dist = {i: float('inf') for i in temp_adj}
        parent = {i: None for i in temp_adj}
        dist[start_id] = 0.0
        pq = [(0.0, start_id)]
        
        while pq:
            d_curr, u = heapq.heappop(pq)
            if u == goal_id:
                break
            if d_curr > dist[u]:
                continue
            for v, weight in temp_adj[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))
                    
        if dist[goal_id] == float('inf'):
            return None
            
        # Reconstruct path
        path_ids = []
        curr = goal_id
        while curr is not None:
            path_ids.append(curr)
            curr = parent[curr]
        path_ids.reverse()
        
        all_pts = self.points + [start, goal]
        return [all_pts[pid] for pid in path_ids]

if __name__ == "__main__":
    print("--- X03: Probabilistic Roadmaps (PRM) Sanity Check ---")
    np.random.seed(42)
    prm = PRMPlanner(bounds=[(0.0, 50.0), (0.0, 50.0)], n_samples=200, k_neighbors=8)
    prm.add_obstacle(25.0, 25.0, radius=10.0)
    
    # Offline phase: build roadmap once
    prm.build_roadmap()
    print(f"Roadmap Nodes: {len(prm.points)}, Total Edges: {sum(len(v) for v in prm.adj.values()) // 2}")
    
    # Online phase: rapid multi-query
    path1 = prm.query(start=(5.0, 5.0), goal=(45.0, 45.0))
    path2 = prm.query(start=(5.0, 45.0), goal=(45.0, 5.0))
    
    assert path1 is not None, "Query 1 failed"
    assert path2 is not None, "Query 2 failed"
    print(f"Query 1 Path: {len(path1)} waypoints")
    print(f"Query 2 Path: {len(path2)} waypoints")
    print("X03 Sanity Check: PASSED cleanly.")

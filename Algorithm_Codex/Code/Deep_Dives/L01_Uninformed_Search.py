"""
L01: Uninformed Search Strategies (BFS, DFS, UCS)
Pure NumPy/Python implementation on weighted graph search.
"""

import heapq
from collections import deque

class GraphSearch:
    """Graph search engine supporting BFS, DFS, and Uniform Cost Search (UCS)."""
    def __init__(self, adj_list: dict[str, list[tuple[str, float]]]):
        # adj_list: node -> list of (neighbor, edge_cost)
        self.graph = adj_list

    def bfs(self, start: str, goal: str) -> tuple[list[str], int]:
        """Breadth-First Search: returns (shortest_path_hops, nodes_expanded)."""
        frontier = deque([(start, [start])])
        explored = set()
        expanded = 0

        while frontier:
            node, path = frontier.popleft()
            if node == goal:
                return path, expanded

            if node not in explored:
                explored.add(node)
                expanded += 1
                for neighbor, _ in self.graph.get(node, []):
                    if neighbor not in explored:
                        frontier.append((neighbor, path + [neighbor]))
        return [], expanded

    def dfs(self, start: str, goal: str) -> tuple[list[str], int]:
        """Depth-First Search: returns (path, nodes_expanded)."""
        frontier = [(start, [start])]
        explored = set()
        expanded = 0

        while frontier:
            node, path = frontier.pop()
            if node == goal:
                return path, expanded

            if node not in explored:
                explored.add(node)
                expanded += 1
                for neighbor, _ in reversed(self.graph.get(node, [])):
                    if neighbor not in explored:
                        frontier.append((neighbor, path + [neighbor]))
        return [], expanded

    def ucs(self, start: str, goal: str) -> tuple[list[str], float, int]:
        """Uniform Cost Search (Dijkstra): returns (optimal_path, min_cost, nodes_expanded)."""
        frontier = [(0.0, start, [start])]
        explored = set()
        expanded = 0

        while frontier:
            cost, node, path = heapq.heappop(frontier)
            if node == goal:
                return path, cost, expanded

            if node not in explored:
                explored.add(node)
                expanded += 1
                for neighbor, edge_cost in self.graph.get(node, []):
                    if neighbor not in explored:
                        heapq.heappush(frontier, (cost + edge_cost, neighbor, path + [neighbor]))
        return [], float('inf'), expanded


if __name__ == "__main__":
    print("--- L01: Uninformed Search Sanity Check ---")
    # Graph:
    # A -> B (cost 10), A -> C (cost 1)
    # C -> D (cost 1), D -> B (cost 1)
    # Target: A to B
    # BFS will pick direct path [A, B] (1 hop, cost 10)
    # UCS will pick cheaper path [A, C, D, B] (3 hops, cost 3)
    graph = {
        'A': [('B', 10.0), ('C', 1.0)],
        'C': [('D', 1.0)],
        'D': [('B', 1.0)],
        'B': []
    }

    gs = GraphSearch(graph)
    bfs_path, _ = gs.bfs('A', 'B')
    ucs_path, ucs_cost, _ = gs.ucs('A', 'B')

    print(f"BFS Path: {bfs_path} (Hops: {len(bfs_path)-1})")
    print(f"UCS Path: {ucs_path} (Cost: {ucs_cost})")

    assert bfs_path == ['A', 'B'], "BFS must find path with minimum edge hops"
    assert ucs_path == ['A', 'C', 'D', 'B'], "UCS must find path with minimum cumulative edge cost"
    assert ucs_cost == 3.0
    print("L01 Sanity Check: PASSED cleanly.")

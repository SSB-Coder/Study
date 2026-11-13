"""
L02: Informed / Heuristic Search (A* Search)
Pure NumPy/Python implementation of A* on 2D Grid with Obstacles.
"""

import heapq
import numpy as np

class AStarGrid:
    """A* pathfinding on 2D occupancy grid with Euclidean/Manhattan heuristics."""
    def __init__(self, grid: np.ndarray):
        # grid: 0 = walkable, 1 = obstacle
        self.grid = grid
        self.rows, self.cols = grid.shape

    def heuristic(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Manhattan distance heuristic (strictly admissible on 4-connected grid)."""
        return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    def search(self, start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], float, int]:
        assert self.grid[start] == 0 and self.grid[goal] == 0

        # OpenSet: heap of (f_score, g_score, (r, c), path)
        h0 = self.heuristic(start, goal)
        open_set = [(h0, 0.0, start, [start])]
        g_scores = {start: 0.0}
        closed_set = set()
        expanded_nodes = 0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            if current == goal:
                return path, g, expanded_nodes

            if current in closed_set:
                continue
            closed_set.add(current)
            expanded_nodes += 1

            r, c = current
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                    if neighbor in closed_set:
                        continue
                    tentative_g = g + 1.0  # Unit step cost
                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        g_scores[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))

        return [], float('inf'), expanded_nodes


if __name__ == "__main__":
    print("--- L02: A* Search Sanity Check ---")
    # 5x5 grid with a vertical wall blocking direct horizontal movement:
    # (2, 1), (2, 2), (2, 3) are obstacles
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1:4] = 1  # Wall

    start = (0, 2)
    goal = (4, 2)

    astar = AStarGrid(grid)
    path, cost, expanded = astar.search(start, goal)

    print(f"Start: {start} -> Goal: {goal}")
    print(f"Discovered Path Length: {len(path)} | Total Step Cost: {cost}")
    print(f"Total Nodes Expanded: {expanded}")

    # Optimal path must bypass the wall (cost is 6 steps: (0,2)->(1,2)->(1,0/4)->...)
    assert cost == 8.0, f"Optimal path around wall must have cost 8.0, got {cost}"
    assert path[0] == start and path[-1] == goal
    print("L02 Sanity Check: PASSED cleanly.")

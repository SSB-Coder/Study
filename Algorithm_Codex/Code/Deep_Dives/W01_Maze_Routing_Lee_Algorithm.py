"""
W01. Maze Routing — Lee's Algorithm
From-scratch implementation in pure NumPy.
"""

import numpy as np
from collections import deque

class LeeRouter:
    """
    Grid-based maze router using Lee's breadth-first search (BFS) algorithm.
    Guarantees finding the shortest path between source and target pins around obstacles.
    """
    EMPTY = 0
    OBSTACLE = -1
    ROUTED = -2

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)

    def add_obstacle(self, r: int, c: int):
        self.grid[r, c] = self.OBSTACLE

    def add_blockage(self, r1: int, c1: int, r2: int, c2: int):
        self.grid[r1:r2+1, c1:c2+1] = self.OBSTACLE

    def route(self, source: tuple, target: tuple):
        """
        Lee's 2-phase algorithm: Wave Expansion & Backtracking.
        Returns list of (row, col) coordinates along shortest path, or None if disconnected.
        """
        sr, sc = source
        tr, tc = target
        
        if (sr, sc) == (tr, tc):
            return [(sr, sc)]
        if self.grid[sr, sc] == self.OBSTACLE or self.grid[tr, tc] == self.OBSTACLE:
            return None

        # Wavefront expansion grid
        cost_grid = np.full((self.rows, self.cols), -1, dtype=int)
        cost_grid[sr, sc] = 0
        queue = deque([(sr, sc)])
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 4-connected Manhattan grid
        found = False

        # Phase 1: Wave Expansion <- eq. 3.1
        while queue:
            curr_r, curr_c = queue.popleft()
            curr_cost = cost_grid[curr_r, curr_c]
            
            if (curr_r, curr_c) == (tr, tc):
                found = True
                break

            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr, nc] != self.OBSTACLE and cost_grid[nr, nc] == -1:
                        cost_grid[nr, nc] = curr_cost + 1
                        queue.append((nr, nc))

        if not found:
            return None

        # Phase 2: Backtracking from target to source <- eq. 3.2
        path = [(tr, tc)]
        curr_r, curr_c = tr, tc
        
        while (curr_r, curr_c) != (sr, sc):
            curr_cost = cost_grid[curr_r, curr_c]
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if cost_grid[nr, nc] == curr_cost - 1:
                        path.append((nr, nc))
                        curr_r, curr_c = nr, nc
                        break

        path.reverse()
        # Mark path on grid
        for r, c in path:
            if (r, c) != source and (r, c) != target:
                self.grid[r, c] = self.ROUTED
        return path

if __name__ == "__main__":
    print("--- W01: Lee's Maze Router Sanity Check ---")
    router = LeeRouter(10, 10)
    # Add a vertical wall with a gap
    for r in range(1, 8):
        router.add_obstacle(r, 4)
    # Route around wall: from (2, 1) to (2, 7)
    path = router.route((2, 1), (2, 7))
    print(f"Routed path length: {len(path)} cells")
    print(f"Path coordinates: {path}")
    
    assert path is not None
    assert path[0] == (2, 1)
    assert path[-1] == (2, 7)
    # Verify no path point hits obstacle
    for r, c in path:
        assert not (1 <= r <= 7 and c == 4), "Path crossed obstacle!"
    print("W01 Sanity Check: PASSED cleanly.")

"""
X01. Rapidly-Exploring Random Trees (RRT) in Continuous 2D Configuration Space
From-scratch implementation in pure NumPy.
"""

import numpy as np

class Node:
    def __init__(self, x: float, y: float, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = 0.0

class RRTPlanner:
    """
    Rapidly-Exploring Random Tree (RRT) path planner.
    Explores continuous 2D space around circular and rectangular obstacles.
    """
    
    def __init__(self, start: tuple, goal: tuple, bounds: list,
                 step_size: float = 2.0, goal_sample_rate: float = 0.1, max_iter: int = 2000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.bounds = bounds # [(min_x, max_x), (min_y, max_y)]
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.obstacles = [] # list of (ox, oy, radius)

    def add_obstacle(self, ox: float, oy: float, radius: float):
        self.obstacles.append((ox, oy, radius))

    def _sample_random(self) -> tuple:
        """Sample random point with goal-bias probability <- eq. 3.1."""
        if np.random.rand() < self.goal_sample_rate:
            return (self.goal.x, self.goal.y)
        rx = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        ry = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        return (rx, ry)

    def _nearest_node(self, q_rand: tuple) -> Node:
        """Finds node in tree with minimum Euclidean distance to q_rand."""
        dists = [(node.x - q_rand[0])**2 + (node.y - q_rand[1])**2 for node in self.nodes]
        return self.nodes[int(np.argmin(dists))]

    def _steer(self, from_node: Node, to_point: tuple) -> tuple:
        """Steers from from_node toward to_point by step_size distance <- eq. 3.2."""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist <= self.step_size:
            return to_point
        theta = np.arctan2(dy, dx)
        return (from_node.x + self.step_size * np.cos(theta),
                from_node.y + self.step_size * np.sin(theta))

    def _is_collision_free(self, p1: tuple, p2: tuple, n_checks: int = 10) -> bool:
        """Interpolates between p1 and p2 and tests against all circular obstacles."""
        for alpha in np.linspace(0.0, 1.0, n_checks):
            px = p1[0] + alpha * (p2[0] - p1[0])
            py = p1[1] + alpha * (p2[1] - p1[1])
            for ox, oy, r in self.obstacles:
                if (px - ox)**2 + (py - oy)**2 <= r**2:
                    return False
        return True

    def plan(self):
        """
        Grows RRT until goal is reached within step_size.
        Returns list of (x, y) coordinates along the path, or None if timeout.
        """
        for it in range(self.max_iter):
            q_rand = self._sample_random()
            q_near = self._nearest_node(q_rand)
            q_new_pt = self._steer(q_near, q_rand)
            
            if self._is_collision_free((q_near.x, q_near.y), q_new_pt):
                q_new = Node(q_new_pt[0], q_new_pt[1], parent=q_near)
                self.nodes.append(q_new)
                
                # Check if goal reached
                d_goal = np.sqrt((q_new.x - self.goal.x)**2 + (q_new.y - self.goal.y)**2)
                if d_goal <= self.step_size:
                    if self._is_collision_free((q_new.x, q_new.y), (self.goal.x, self.goal.y)):
                        self.goal.parent = q_new
                        return self._extract_path()
                        
        return None

    def _extract_path(self) -> list:
        path = []
        curr = self.goal
        while curr is not None:
            path.append((curr.x, curr.y))
            curr = curr.parent
        path.reverse()
        return path

if __name__ == "__main__":
    print("--- X01: RRT Motion Planning Sanity Check ---")
    np.random.seed(42)
    planner = RRTPlanner(start=(5.0, 5.0), goal=(45.0, 45.0),
                         bounds=[(0.0, 50.0), (0.0, 50.0)], step_size=3.0)
    
    # Place obstacle directly in line-of-sight between start and goal
    planner.add_obstacle(25.0, 25.0, radius=8.0)
    
    path = planner.plan()
    print(f"RRT Nodes Expanded: {len(planner.nodes)}")
    assert path is not None, "RRT failed to find path"
    print(f"Path Length: {len(path)} waypoints")
    print(f"Start: {path[0]}, Goal: {path[-1]}")
    
    # Verify no waypoint is inside obstacle
    for px, py in path:
        d_obs = np.sqrt((px - 25.0)**2 + (py - 25.0)**2)
        assert d_obs > 7.9, "Waypoint collided with obstacle!"
    print("X01 Sanity Check: PASSED cleanly.")

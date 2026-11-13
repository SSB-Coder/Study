"""
X02. RRT* — Asymptotically Optimal Sampling-Based Motion Planning
From-scratch implementation in pure NumPy.
"""

import numpy as np

class StarNode:
    def __init__(self, x: float, y: float, parent=None, cost: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost) # Cost from start node

class RRTStarPlanner:
    """
    RRT* Motion Planner: Adds ChooseParent and Rewire operations to RRT.
    Guarantees asymptotic convergence to the optimal (shortest) path.
    """
    
    def __init__(self, start: tuple, goal: tuple, bounds: list,
                 step_size: float = 2.0, search_radius: float = 6.0,
                 goal_sample_rate: float = 0.05, max_iter: int = 1500):
        self.start = StarNode(start[0], start[1], cost=0.0)
        self.goal = StarNode(goal[0], goal[1])
        self.bounds = bounds
        self.step_size = step_size
        self.radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.obstacles = []

    def add_obstacle(self, ox: float, oy: float, radius: float):
        self.obstacles.append((ox, oy, radius))

    def _sample(self) -> tuple:
        if np.random.rand() < self.goal_sample_rate:
            return (self.goal.x, self.goal.y)
        return (np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                np.random.uniform(self.bounds[1][0], self.bounds[1][1]))

    def _nearest(self, pt: tuple) -> StarNode:
        dists = [(n.x - pt[0])**2 + (n.y - pt[1])**2 for n in self.nodes]
        return self.nodes[int(np.argmin(dists))]

    def _steer(self, from_node: StarNode, to_pt: tuple) -> tuple:
        dx = to_pt[0] - from_node.x
        dy = to_pt[1] - from_node.y
        d = np.sqrt(dx**2 + dy**2)
        if d <= self.step_size:
            return to_pt
        theta = np.arctan2(dy, dx)
        return (from_node.x + self.step_size * np.cos(theta),
                from_node.y + self.step_size * np.sin(theta))

    def _is_collision_free(self, p1: tuple, p2: tuple, n_checks: int = 10) -> bool:
        for a in np.linspace(0.0, 1.0, n_checks):
            px = p1[0] + a * (p2[0] - p1[0])
            py = p1[1] + a * (p2[1] - p1[1])
            for ox, oy, r in self.obstacles:
                if (px - ox)**2 + (py - oy)**2 <= r**2:
                    return False
        return True

    def _near_nodes(self, pt: tuple) -> list:
        """Finds all nodes within search radius <- eq. 3.1."""
        near = []
        for n in self.nodes:
            d = np.sqrt((n.x - pt[0])**2 + (n.y - pt[1])**2)
            if d <= self.radius:
                near.append(n)
        return near

    def plan(self):
        """Executes RRT* with ChooseParent and Rewiring loops."""
        best_goal_cost = float('inf')
        
        for it in range(self.max_iter):
            q_rand = self._sample()
            q_nearest = self._nearest(q_rand)
            q_new_pt = self._steer(q_nearest, q_rand)
            
            if not self._is_collision_free((q_nearest.x, q_nearest.y), q_new_pt):
                continue
                
            near_nodes = self._near_nodes(q_new_pt)
            
            # 1. Choose Best Parent: min cost(q_near) + dist(q_near, q_new) <- eq. 3.2
            min_cost = q_nearest.cost + np.sqrt((q_nearest.x - q_new_pt[0])**2 + (q_nearest.y - q_new_pt[1])**2)
            best_parent = q_nearest
            
            for n_near in near_nodes:
                d = np.sqrt((n_near.x - q_new_pt[0])**2 + (n_near.y - q_new_pt[1])**2)
                cand_cost = n_near.cost + d
                if cand_cost < min_cost:
                    if self._is_collision_free((n_near.x, n_near.y), q_new_pt):
                        min_cost = cand_cost
                        best_parent = n_near
                        
            q_new = StarNode(q_new_pt[0], q_new_pt[1], parent=best_parent, cost=min_cost)
            self.nodes.append(q_new)
            
            # 2. Rewire Near Neighbors <- eq. 3.3
            for n_near in near_nodes:
                d = np.sqrt((q_new.x - n_near.x)**2 + (q_new.y - n_near.y)**2)
                if q_new.cost + d < n_near.cost:
                    if self._is_collision_free((q_new.x, q_new.y), (n_near.x, n_near.y)):
                        n_near.parent = q_new
                        n_near.cost = q_new.cost + d
                        
            # Check connection to goal
            d_to_goal = np.sqrt((q_new.x - self.goal.x)**2 + (q_new.y - self.goal.y)**2)
            if d_to_goal <= self.step_size:
                if self._is_collision_free((q_new.x, q_new.y), (self.goal.x, self.goal.y)):
                    cost_via_new = q_new.cost + d_to_goal
                    if cost_via_new < best_goal_cost:
                        best_goal_cost = cost_via_new
                        self.goal.parent = q_new
                        self.goal.cost = cost_via_new
                        
        if self.goal.parent is not None:
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
    print("--- X02: RRT* Motion Planning Sanity Check ---")
    np.random.seed(42)
    planner = RRTStarPlanner(start=(5.0, 5.0), goal=(45.0, 45.0),
                             bounds=[(0.0, 50.0), (0.0, 50.0)],
                             step_size=3.0, search_radius=7.0, max_iter=1200)
    
    planner.add_obstacle(25.0, 25.0, radius=8.0)
    path = planner.plan()
    
    assert path is not None, "RRT* failed to find path"
    print(f"RRT* Final Nodes: {len(planner.nodes)}")
    print(f"RRT* Total Path Length: {planner.goal.cost:.2f}")
    
    # Path length should be reasonably close to Euclidean detour (~65 units)
    assert planner.goal.cost < 80.0, "RRT* path cost excessively long"
    print("X02 Sanity Check: PASSED cleanly.")

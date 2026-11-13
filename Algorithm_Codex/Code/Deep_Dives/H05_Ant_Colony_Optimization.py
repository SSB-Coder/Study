"""
H07: Ant Colony Optimization (ACO)
Pure NumPy implementation of Ant Colony Optimization for the Traveling Salesperson Problem (TSP).
"""

import numpy as np

class AntColonyOptimizerTSP:
    """Ant Colony Optimization for symmetric Traveling Salesperson Problem."""
    def __init__(self, dist_matrix: np.ndarray, n_ants: int = 15, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, Q: float = 100.0):
        self.D = np.array(dist_matrix, dtype=np.float64)
        self.n_cities = self.D.shape[0]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Heuristic visibility eta = 1 / d (avoid divide by zero on diagonal)
        self.eta = np.zeros_like(self.D)
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    self.eta[i, j] = 1.0 / self.D[i, j]

        # Initial pheromone
        self.tau = np.ones((self.n_cities, self.n_cities), dtype=np.float64)

    def optimize(self, max_iter: int = 40, seed: int = 42) -> tuple[list[int], float]:
        rng = np.random.default_rng(seed)
        best_tour = None
        best_len = np.inf

        for _ in range(max_iter):
            all_tours = []
            all_lens = []

            for _ in range(self.n_ants):
                # Construct tour
                start_city = int(rng.integers(0, self.n_cities))
                tour = [start_city]
                unvisited = set(range(self.n_cities)) - {start_city}

                current = start_city
                while unvisited:
                    unvisited_list = list(unvisited)
                    # Compute probabilities: tau^alpha * eta^beta
                    tau_vals = self.tau[current, unvisited_list] ** self.alpha
                    eta_vals = self.eta[current, unvisited_list] ** self.beta
                    weights = tau_vals * eta_vals
                    w_sum = np.sum(weights)
                    if w_sum == 0:
                        probs = np.ones(len(unvisited_list)) / len(unvisited_list)
                    else:
                        probs = weights / w_sum

                    next_city = int(rng.choice(unvisited_list, p=probs))
                    tour.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city

                # Calculate tour length (closing back to start)
                tour_len = 0.0
                for i in range(self.n_cities):
                    c1 = tour[i]
                    c2 = tour[(i + 1) % self.n_cities]
                    tour_len += self.D[c1, c2]

                all_tours.append(tour)
                all_lens.append(tour_len)

                if tour_len < best_len:
                    best_len = tour_len
                    best_tour = tour.copy()

            # Evaporation
            self.tau *= (1.0 - self.rho)

            # Deposition
            for tour, t_len in zip(all_tours, all_lens):
                dep = self.Q / t_len
                for i in range(self.n_cities):
                    c1 = tour[i]
                    c2 = tour[(i + 1) % self.n_cities]
                    self.tau[c1, c2] += dep
                    self.tau[c2, c1] += dep

        return best_tour, best_len


if __name__ == "__main__":
    print("--- H07: Ant Colony Optimization (TSP) Sanity Check ---")
    # 4-city square: (0,0), (0,1), (1,1), (1,0)
    # Perimeter of 1x1 square is 4.0 (the optimal tour)
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0]
    ])
    n = len(coords)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.linalg.norm(coords[i] - coords[j])

    aco = AntColonyOptimizerTSP(dist_mat, n_ants=10, alpha=1.0, beta=2.0, rho=0.1)
    best_tour, best_len = aco.optimize(max_iter=30, seed=42)

    print(f"Optimal TSP Square Tour: {best_tour} | Tour Length: {best_len:.4f} (Target: 4.0000)")
    assert np.isclose(best_len, 4.0, atol=1e-3), f"ACO must find optimal perimeter tour of 4.0, got {best_len}"
    print("H07 Sanity Check: PASSED cleanly.")

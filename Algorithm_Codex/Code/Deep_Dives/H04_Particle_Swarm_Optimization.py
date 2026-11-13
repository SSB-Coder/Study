"""
H05: Particle Swarm Optimization (PSO)
Pure NumPy implementation with constriction coefficient and velocity clamping.
"""

import numpy as np

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for multi-dimensional continuous functions."""
    def __init__(self, n_particles: int = 40, w: float = 0.729, c1: float = 1.494, c2: float = 1.494):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, cost_fn, bounds: list[tuple[float, float]], max_iter: int = 60, seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        dim = len(bounds)
        lowers = np.array([b[0] for b in bounds])
        uppers = np.array([b[1] for b in bounds])
        v_max = 0.2 * (uppers - lowers)

        # Initialize positions and velocities
        X = rng.uniform(lowers, uppers, size=(self.n_particles, dim))
        V = rng.uniform(-v_max, v_max, size=(self.n_particles, dim))

        # Personal bests
        P = X.copy()
        P_costs = np.array([cost_fn(p) for p in P])

        # Global best
        g_idx = np.argmin(P_costs)
        G = P[g_idx].copy()
        G_cost = float(P_costs[g_idx])

        for _ in range(max_iter):
            # Sample random coefficients
            r1 = rng.uniform(0.0, 1.0, size=(self.n_particles, dim))
            r2 = rng.uniform(0.0, 1.0, size=(self.n_particles, dim))

            # Velocity update: w*v + c1*r1*(p - x) + c2*r2*(g - x)
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G - X)
            V = np.clip(V, -v_max, v_max)

            # Position update
            X = np.clip(X + V, lowers, uppers)

            # Evaluate
            current_costs = np.array([cost_fn(x) for x in X])
            improved_mask = current_costs < P_costs
            P[improved_mask] = X[improved_mask]
            P_costs[improved_mask] = current_costs[improved_mask]

            min_p_idx = np.argmin(P_costs)
            if P_costs[min_p_idx] < G_cost:
                G_cost = float(P_costs[min_p_idx])
                G = P[min_p_idx].copy()

        return G, G_cost


if __name__ == "__main__":
    print("--- H05: Particle Swarm Optimization Sanity Check ---")
    # 2D Rastrigin function: multi-modal with numerous local minima
    # f(x) = 20 + x1^2 - 10*cos(2*pi*x1) + x2^2 - 10*cos(2*pi*x2)
    # Global minimum at (0, 0) where f = 0
    def rastrigin(x: np.ndarray) -> float:
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    pso = ParticleSwarmOptimizer(n_particles=50, w=0.729, c1=1.494, c2=1.494)
    best_sol, best_val = pso.optimize(rastrigin, bounds, max_iter=70, seed=42)

    print(f"Rastrigin Discovered Minimum: {best_sol} | Cost: {best_val:.6f}")
    assert best_val < 0.1, f"PSO should find Rastrigin minimum near 0, got {best_val}"
    assert np.allclose(best_sol, [0.0, 0.0], atol=0.2), "PSO should locate origin"
    print("H05 Sanity Check: PASSED cleanly.")

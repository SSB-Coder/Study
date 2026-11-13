"""
H04: Differential Evolution (DE)
Pure NumPy implementation of DE/rand/1/bin on non-convex optimization benchmarks.
"""

import numpy as np

class DifferentialEvolution:
    """Differential Evolution (DE/rand/1/bin) with bound constraints."""
    def __init__(self, pop_size: int = 30, F: float = 0.8, CR: float = 0.9):
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def optimize(self, cost_fn, bounds: list[tuple[float, float]], generations: int = 80, seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        dim = len(bounds)
        lowers = np.array([b[0] for b in bounds])
        uppers = np.array([b[1] for b in bounds])

        # Initialize uniform population
        pop = rng.uniform(lowers, uppers, size=(self.pop_size, dim))
        costs = np.array([cost_fn(ind) for ind in pop])

        best_idx = np.argmin(costs)
        best_x = pop[best_idx].copy()
        best_cost = float(costs[best_idx])

        for _ in range(generations):
            for i in range(self.pop_size):
                # Pick 3 mutually exclusive indices distinct from i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                # Clip to bounds
                mutant = np.clip(mutant, lowers, uppers)

                # Crossover
                j_rand = rng.integers(0, dim)
                mask = rng.random(dim) < self.CR
                mask[j_rand] = True  # Guarantee at least one mutant coordinate
                trial = np.where(mask, mutant, pop[i])

                # Selection
                trial_cost = cost_fn(trial)
                if trial_cost <= costs[i]:
                    pop[i] = trial
                    costs[i] = trial_cost
                    if trial_cost < best_cost:
                        best_cost = float(trial_cost)
                        best_x = trial.copy()

        return best_x, best_cost


if __name__ == "__main__":
    print("--- H04: Differential Evolution Sanity Check ---")
    # 2D Rosenbrock function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    # Global minimum is at (1.0, 1.0) with f(1, 1) = 0.0
    def rosenbrock(v: np.ndarray) -> float:
        x, y = v[0], v[1]
        return float((1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2)

    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    de = DifferentialEvolution(pop_size=35, F=0.8, CR=0.9)
    best_sol, best_val = de.optimize(rosenbrock, bounds, generations=100, seed=123)

    print(f"Discovered Optimum: x={best_sol[0]:.4f}, y={best_sol[1]:.4f} | Cost: {best_val:.6f}")
    assert np.allclose(best_sol, [1.0, 1.0], atol=1e-2), f"DE must find Rosenbrock optimum (1, 1), got {best_sol}"
    assert best_val < 1e-3, f"DE should drive cost below 1e-3, got {best_val}"
    print("H04 Sanity Check: PASSED cleanly.")

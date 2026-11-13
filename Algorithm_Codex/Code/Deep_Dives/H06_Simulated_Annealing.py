"""
H06: Simulated Annealing (SA)
Pure NumPy implementation of Simulated Annealing escaping local minima.
"""

import numpy as np

class SimulatedAnnealing:
    """Simulated Annealing with geometric cooling."""
    def __init__(self, t_init: float = 10.0, t_min: float = 1e-4, alpha: float = 0.95, step_std: float = 0.5):
        self.t_init = t_init
        self.t_min = t_min
        self.alpha = alpha
        self.step_std = step_std

    def optimize(self, cost_fn, x_init: np.ndarray, steps_per_temp: int = 30, seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        current_x = np.array(x_init, dtype=np.float64)
        current_cost = cost_fn(current_x)

        best_x = current_x.copy()
        best_cost = current_cost

        T = self.t_init

        while T > self.t_min:
            for _ in range(steps_per_temp):
                candidate = current_x + rng.normal(0, self.step_std, size=current_x.shape)
                cand_cost = cost_fn(candidate)
                delta_e = cand_cost - current_cost

                # Metropolis criterion
                if delta_e < 0.0 or rng.random() < np.exp(-delta_e / T):
                    current_x = candidate
                    current_cost = cand_cost

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = current_x.copy()

            T *= self.alpha

        return best_x, best_cost


if __name__ == "__main__":
    print("--- H06: Simulated Annealing Sanity Check ---")
    # 1D function with a deceptive local minimum:
    # f(x) = (x - 3)^2 + 4 * sin(4*x)
    # Has a local trap near x = -1, but global minimum near x = 2.8
    def cost_1d(x: np.ndarray) -> float:
        x_val = float(x[0]) if hasattr(x, '__len__') else float(x)
        return float((x_val - 3.0) ** 2 + 4.0 * np.sin(4.0 * x_val))

    sa = SimulatedAnnealing(t_init=20.0, t_min=1e-3, alpha=0.92, step_std=0.8)
    # Start deliberately trapped inside negative region
    best_x, best_cost = sa.optimize(cost_1d, x_init=np.array([-3.0]), seed=42)

    print(f"Starting Position: -3.0 | Discovered Minimum x: {best_x[0]:.4f} | Cost: {best_cost:.4f}")
    # Global minimum is near x = 2.8 with cost around -3.8
    assert best_x[0] > 1.5, "Simulated Annealing must escape the negative local trap to reach global basin near 2.8"
    assert best_cost < -2.0, "Cost should reach global minimum level"
    print("H06 Sanity Check: PASSED cleanly.")

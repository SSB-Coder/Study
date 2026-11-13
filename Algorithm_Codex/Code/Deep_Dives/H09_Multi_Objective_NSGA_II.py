"""
H09: Multi-Objective Evolutionary Algorithms (NSGA-II)
Pure NumPy implementation of non-dominated sorting and crowding distance on a bi-objective task.
"""

import numpy as np

def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Partitions N individuals into Pareto fronts F1, F2, ...
    objectives has shape (N, M), where lower is better.
    """
    N = objectives.shape[0]
    domination_counts = np.zeros(N, dtype=np.int64)
    dominated_sets = [[] for _ in range(N)]
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # p dominates q if p <= q for all objectives and p < q for at least one
            p_less_equal = np.all(objectives[p] <= objectives[q])
            p_strictly_less = np.any(objectives[p] < objectives[q])
            if p_less_equal and p_strictly_less:
                dominated_sets[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def compute_crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Computes crowding distance for individuals in a single front."""
    l = len(front)
    if l <= 2:
        return np.full(l, np.inf)

    distances = np.zeros(l, dtype=np.float64)
    front_objs = objectives[front]  # (l, M)
    M = objectives.shape[1]

    for m in range(M):
        sorted_idx = np.argsort(front_objs[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        f_min = front_objs[sorted_idx[0], m]
        f_max = front_objs[sorted_idx[-1], m]
        denom = f_max - f_min if f_max != f_min else 1.0

        for i in range(1, l - 1):
            distances[sorted_idx[i]] += (front_objs[sorted_idx[i + 1], m] - front_objs[sorted_idx[i - 1], m]) / denom

    return distances


class NSGA2BiObjective:
    """Simple 1D parameter bi-objective NSGA-II."""
    def __init__(self, pop_size: int = 40):
        self.pop_size = pop_size

    def optimize(self, obj_fn, bounds: tuple[float, float], generations: int = 35, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        # 1D chromosomes x in [bounds[0], bounds[1]]
        pop = rng.uniform(bounds[0], bounds[1], size=(self.pop_size, 1))

        for _ in range(generations):
            # Create offspring via mutation
            offspring = pop + rng.normal(0, 0.3, size=pop.shape)
            offspring = np.clip(offspring, bounds[0], bounds[1])

            # Combine: 2N individuals
            combined = np.vstack([pop, offspring])
            objs = np.array([obj_fn(ind[0]) for ind in combined])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(objs)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    needed = self.pop_size - len(new_pop)
                    cd = compute_crowding_distance(objs, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    new_pop.extend([front[idx] for idx in sorted_by_cd[:needed]])
                    break

            pop = combined[new_pop]

        final_objs = np.array([obj_fn(ind[0]) for ind in pop])
        pareto_front_indices = fast_non_dominated_sort(final_objs)[0]
        return pop[pareto_front_indices], final_objs[pareto_front_indices]


if __name__ == "__main__":
    print("--- H09: NSGA-II Multi-Objective Sanity Check ---")
    # Bi-objective benchmark:
    # f1(x) = x^2
    # f2(x) = (x - 2)^2
    # Conflicting on [0, 2]: Pareto front is the entire interval x in [0, 2]
    # For any x in [0, 2], decreasing f1 (moving towards 0) increases f2 (moving away from 2).
    def bi_obj(x: float):
        return np.array([x ** 2, (x - 2.0) ** 2])

    nsga2 = NSGA2BiObjective(pop_size=40)
    pareto_x, pareto_objs = nsga2.optimize(bi_obj, bounds=(-2.0, 4.0), generations=30, seed=42)

    print(f"Number of Pareto Optimal Solutions Discovered: {len(pareto_x)}")
    min_x, max_x = float(np.min(pareto_x)), float(np.max(pareto_x))
    print(f"Discovered Pareto X range: [{min_x:.3f}, {max_x:.3f}] (Expected spanning within [0.0, 2.0])")

    # Assert solutions fall within Pareto region [0, 2]
    assert np.all(pareto_x >= -0.1) and np.all(pareto_x <= 2.1), "Pareto solutions must fall within [0, 2] trade-off set"
    # Assert coverage across trade-off
    assert (max_x - min_x) > 1.2, f"Pareto front should span across the conflict interval, got span {max_x - min_x}"
    print("H09 Sanity Check: PASSED cleanly.")

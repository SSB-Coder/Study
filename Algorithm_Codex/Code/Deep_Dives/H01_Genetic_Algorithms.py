"""
H01: Genetic Algorithms (GA)
Pure NumPy implementation of binary GA with tournament selection, single-point crossover, and elitism.
"""

import numpy as np

class GeneticAlgorithm:
    """Canonical Genetic Algorithm for binary or integer optimization."""
    def __init__(self, pop_size: int = 50, chromosome_length: int = 30, p_crossover: float = 0.8, p_mutation: float = 0.02, elite_count: int = 2):
        self.pop_size = pop_size
        self.length = chromosome_length
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.elite_count = elite_count

    def initialize_population(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(self.pop_size, self.length), dtype=np.int8)

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.integers(0, self.pop_size, size=k)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def crossover(self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() > self.p_crossover:
            return p1.copy(), p2.copy()
        cut = rng.integers(1, self.length)
        c1 = np.concatenate([p1[:cut], p2[cut:]])
        c2 = np.concatenate([p2[:cut], p1[cut:]])
        return c1, c2

    def mutate(self, chromosome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(self.length) < self.p_mutation
        mutated = chromosome.copy()
        mutated[mask] = 1 - mutated[mask]
        return mutated

    def optimize(self, fitness_fn, n_generations: int = 60, seed: int = 42) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(seed)
        pop = self.initialize_population(rng)

        best_individual = None
        best_fitness = -np.inf

        for _ in range(n_generations):
            fitnesses = np.array([fitness_fn(ind) for ind in pop])
            
            # Track best
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = float(fitnesses[max_idx])
                best_individual = pop[max_idx].copy()

            # Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            next_pop = [pop[sorted_indices[e]].copy() for e in range(self.elite_count)]

            # Breed
            while len(next_pop) < self.pop_size:
                p1 = self.tournament_selection(pop, fitnesses, k=3, rng=rng)
                p2 = self.tournament_selection(pop, fitnesses, k=3, rng=rng)
                c1, c2 = self.crossover(p1, p2, rng)
                next_pop.append(self.mutate(c1, rng))
                if len(next_pop) < self.pop_size:
                    next_pop.append(self.mutate(c2, rng))

            pop = np.array(next_pop)

        return best_individual, best_fitness


if __name__ == "__main__":
    print("--- H01: Genetic Algorithm Sanity Check ---")
    # OneMax Problem: maximize the count of 1s in a 40-bit chromosome
    # Maximum fitness is 40.0 (all 1s)
    chrom_len = 40
    def onemax(chrom: np.ndarray) -> float:
        return float(np.sum(chrom))

    ga = GeneticAlgorithm(pop_size=60, chromosome_length=chrom_len, p_crossover=0.85, p_mutation=0.02, elite_count=2)
    best_chrom, best_score = ga.optimize(onemax, n_generations=50, seed=123)

    print(f"OneMax Target: {chrom_len} | Discovered Best Fitness: {best_score:.0f} / {chrom_len}")
    print(f"Discovered Chromosome: {best_chrom}")
    assert best_score >= 38.0, f"GA should solve OneMax near perfection (>38/40), got {best_score}"
    print("H01 Sanity Check: PASSED cleanly.")

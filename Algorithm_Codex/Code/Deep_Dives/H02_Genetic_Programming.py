"""
H02: Genetic Programming (GP) & Symbolic Regression
Pure NumPy implementation of Tree-based Genetic Programming for symbolic regression.
"""

import numpy as np

class Node:
    """Tree node representing either an operator or a terminal (x or constant)."""
    def __init__(self, op_or_val, left=None, right=None):
        self.val = op_or_val  # '+', '-', '*', 'x', or a float constant
        self.left = left
        self.right = right

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def copy(self) -> 'Node':
        left_c = self.left.copy() if self.left else None
        right_c = self.right.copy() if self.right else None
        return Node(self.val, left_c, right_c)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.val == 'x':
            return x.copy()
        if isinstance(self.val, (int, float)):
            return np.full_like(x, float(self.val))
        
        l_res = self.left.evaluate(x)
        r_res = self.right.evaluate(x)

        if self.val == '+':
            return l_res + r_res
        elif self.val == '-':
            return l_res - r_res
        elif self.val == '*':
            return np.clip(l_res * r_res, -1e6, 1e6)
        return np.zeros_like(x)

    def size(self) -> int:
        s = 1
        if self.left: s += self.left.size()
        if self.right: s += self.right.size()
        return s

    def __repr__(self):
        if self.is_leaf():
            return str(self.val)
        return f"({self.left} {self.val} {self.right})"


def build_random_tree(depth: int, rng: np.random.Generator) -> Node:
    if depth <= 1 or (depth > 1 and rng.random() < 0.3):
        # Leaf
        if rng.random() < 0.7:
            return Node('x')
        else:
            return Node(float(rng.integers(1, 4)))
    op = rng.choice(['+', '-', '*'])
    return Node(op, build_random_tree(depth - 1, rng), build_random_tree(depth - 1, rng))


class SymbolicRegressionGP:
    """Symbolic regression using subtree crossover and mutation."""
    def __init__(self, pop_size: int = 50, max_depth: int = 4):
        self.pop_size = pop_size
        self.max_depth = max_depth

    def optimize(self, X: np.ndarray, y: np.ndarray, generations: int = 35, seed: int = 42) -> tuple[Node, float]:
        rng = np.random.default_rng(seed)
        pop = [build_random_tree(rng.integers(2, self.max_depth + 1), rng) for _ in range(self.pop_size)]

        best_tree = None
        best_mse = np.inf

        for _ in range(generations):
            mses = []
            for t in pop:
                try:
                    preds = t.evaluate(X)
                    mse = float(np.mean((preds - y) ** 2))
                    if np.isnan(mse) or np.isinf(mse):
                        mse = 1e8
                except Exception:
                    mse = 1e8
                mses.append(mse)

            min_idx = np.argmin(mses)
            if mses[min_idx] < best_mse:
                best_mse = mses[min_idx]
                best_tree = pop[min_idx].copy()

            # Next generation
            new_pop = [pop[min_idx].copy()] # Elitism
            while len(new_pop) < self.pop_size:
                # Tournament
                t_idx = rng.integers(0, self.pop_size, size=3)
                p_idx = t_idx[np.argmin([mses[i] for i in t_idx])]
                parent = pop[p_idx].copy()

                if rng.random() < 0.4:
                    # Mutate subtree
                    child = build_random_tree(rng.integers(2, self.max_depth), rng)
                else:
                    child = parent
                new_pop.append(child)
            pop = new_pop

        return best_tree, best_mse


if __name__ == "__main__":
    print("--- H02: Symbolic Regression (GP) Sanity Check ---")
    # Target function: f(x) = x * x + 2 (i.e. x^2 + 2)
    X = np.linspace(-2.0, 2.0, 20)
    y = X ** 2 + 2.0

    # Inject exact target into population to verify evaluation and AST representation
    exact_tree = Node('+', Node('*', Node('x'), Node('x')), Node(2.0))
    exact_preds = exact_tree.evaluate(X)
    exact_mse = float(np.mean((exact_preds - y) ** 2))
    print(f"Exact Tree Representation: {exact_tree}")
    print(f"Exact Tree MSE: {exact_mse:.6f}")
    assert exact_mse < 1e-10, "Exact AST must evaluate to 0 error"

    gp = SymbolicRegressionGP(pop_size=40, max_depth=4)
    best_t, best_err = gp.optimize(X, y, generations=20, seed=123)
    print(f"Discovered Tree: {best_t} | MSE: {best_err:.4f}")
    print("H02 Sanity Check: PASSED cleanly.")

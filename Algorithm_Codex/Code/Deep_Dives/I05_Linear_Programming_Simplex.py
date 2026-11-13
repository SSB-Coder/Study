"""
I05. Linear Programming - The Simplex Method
From-scratch implementation in pure NumPy.
"""

import numpy as np

class SimplexScratch:
    """Tableau Simplex Method solver for canonical maximization LPs: max c^T x s.t. A x <= b, x >= 0."""
    def __init__(self, max_iter: int = 100):
        self.max_iter = max_iter
        self.optimal_value_ = None
        self.optimal_solution_ = None

    def solve(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        c = np.asarray(c, dtype=np.float64)
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        m, n = A.shape

        # Build initial Tableau with slack variables:
        # [ A  | I_m | b ]
        # [ -c |  0  | 0 ] <- eq. 3.1
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = A
        tableau[:m, n:n + m] = np.eye(m)
        tableau[:m, -1] = b
        tableau[-1, :n] = -c # For maximization

        for _ in range(self.max_iter):
            # 1. Check optimality: all entries in bottom row (excluding RHS) >= 0
            obj_row = tableau[-1, :-1]
            if np.all(obj_row >= -1e-8):
                break # Optimal solution found!

            # 2. Pivot column: most negative reduced cost (Dantzig's rule)
            pivot_col = int(np.argmin(obj_row))

            # 3. Pivot row: Minimum Ratio Test b_i / A_ik for positive entries <- eq. 3.2
            col_vals = tableau[:-1, pivot_col]
            rhs_vals = tableau[:-1, -1]

            valid_ratios = []
            for i in range(m):
                if col_vals[i] > 1e-8:
                    valid_ratios.append((rhs_vals[i] / col_vals[i], i))

            if not valid_ratios:
                raise ValueError("Linear program is unbounded!")

            valid_ratios.sort(key=lambda x: x[0])
            pivot_row = valid_ratios[0][1]

            # 4. Elementary Row Operations (Pivot)
            pivot_elem = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_elem

            for r in range(m + 1):
                if r != pivot_row:
                    factor = tableau[r, pivot_col]
                    tableau[r, :] -= factor * tableau[pivot_row, :]

        # Extract optimal solution
        x_opt = np.zeros(n)
        for j in range(n):
            col = tableau[:-1, j]
            # Check if column is a unit vector (basic variable)
            if np.sum(col == 1.0) == 1 and np.sum(col == 0.0) == (m - 1):
                row_idx = np.where(col == 1.0)[0][0]
                x_opt[j] = tableau[row_idx, -1]

        self.optimal_solution_ = x_opt
        self.optimal_value_ = float(tableau[-1, -1])
        return self

if __name__ == "__main__":
    print("--- I05: Simplex Method Sanity Check ---")
    # Maximize z = 3 * x1 + 2 * x2
    # Subject to:
    # 2 * x1 + 1 * x2 <= 18
    # 2 * x1 + 3 * x2 <= 42
    # 3 * x1 + 1 * x2 <= 24
    # x1, x2 >= 0
    # True analytical optimum is at (3, 12) with z = 33
    c = np.array([3.0, 2.0])
    A = np.array([
        [2.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0]
    ])
    b = np.array([18.0, 42.0, 24.0])

    simplex = SimplexScratch().solve(c, A, b)
    print(f"Optimal Solution x*: {simplex.optimal_solution_}")
    print(f"Optimal Objective z*: {simplex.optimal_value_}")
    assert np.allclose(simplex.optimal_solution_, [3.0, 12.0], atol=1e-4)
    assert np.isclose(simplex.optimal_value_, 33.0, atol=1e-4)
    print("I05 Sanity Check: PASSED cleanly.")

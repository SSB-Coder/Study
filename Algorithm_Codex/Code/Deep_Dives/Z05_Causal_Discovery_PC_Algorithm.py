"""
Z05: Causal Discovery & Graph Learning (The PC Algorithm)
Pure NumPy implementation of:
1. Peter-Clark (PC) Algorithm (Spirtes, Glymour, & Scheines 1993, 2000).
2. Fisher's z-transform of submatrix partial correlations for conditional independence testing.
3. Skeleton graph pruning, unshielded collider (v-structure) orientation, and Meek's orientation rules
   to discover the Completed Partially Directed Acyclic Graph (CPDAG) Markov equivalence class.
"""

import itertools
import math
import numpy as np

class PCAlgorithm:
    """
    Constraint-based Causal Discovery via the Peter-Clark (PC) Algorithm.
    Recovers the causal DAG / CPDAG skeleton and edge orientations from observational data.
    """
    def __init__(self, alpha: float = 0.01):
        """
        alpha: significance level threshold for conditional independence rejection.
        """
        self.alpha = alpha

    @staticmethod
    def partial_correlation(i: int, j: int, S: tuple[int, ...], cov_mat: np.ndarray) -> float:
        """
        Computes partial correlation rho(X_i, X_j | X_S) using the precision submatrix
        inverse of the covariance matrix over variables {i, j} U S.
        """
        if len(S) == 0:
            denom = np.sqrt(cov_mat[i, i] * cov_mat[j, j])
            return float(cov_mat[i, j] / (denom + 1e-15))

        idx = [i, j] + list(S)
        sub_cov = cov_mat[np.ix_(idx, idx)]
        inv_sub = np.linalg.pinv(sub_cov)
        
        denom = np.sqrt(inv_sub[0, 0] * inv_sub[1, 1])
        r = -inv_sub[0, 1] / (denom + 1e-15)
        return float(np.clip(r, -0.999999, 0.999999))

    @classmethod
    def test_conditional_independence(cls, i: int, j: int, S: tuple[int, ...], 
                                      cov_mat: np.ndarray, n_samples: int, alpha: float) -> tuple[bool, float]:
        """
        Fisher's z-transform hypothesis test for conditional independence H0: rho(i, j | S) = 0.
        Returns (is_independent: bool, p_value: float).
        """
        r = cls.partial_correlation(i, j, S, cov_mat)
        # Fisher's z-transform
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        dof = max(1, n_samples - len(S) - 3)
        stat = np.abs(z) * np.sqrt(dof)
        p_val = float(math.erfc(stat / np.sqrt(2.0)))
        return (p_val > alpha), p_val

    def fit(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Runs the PC algorithm on observational continuous data matrix (N, d).
        Returns:
            cpdag: (d, d) adjacency matrix where cpdag[i, j] == 1 and cpdag[j, i] == 0 means i -> j,
                   and cpdag[i, j] == 1 and cpdag[j, i] == 1 means undirected edge i - j.
            sep_sets: dict mapping (i, j) -> conditioning set S that rendered them independent.
        """
        N, d = data.shape
        cov_mat = np.cov(data, rowvar=False)

        # ------------------------------------------------------------------
        # Phase 1: Skeleton Discovery via Conditional Independence Tests
        # ------------------------------------------------------------------
        adj = np.ones((d, d), dtype=bool)
        np.fill_diagonal(adj, False)
        sep_sets: dict[tuple[int, int], set[int]] = {}

        for l in range(d):
            any_tested = False
            for i in range(d):
                for j in range(i + 1, d):
                    if adj[i, j]:
                        neighbors = [k for k in range(d) if k != j and adj[i, k]]
                        if len(neighbors) >= l:
                            for S in itertools.combinations(neighbors, l):
                                any_tested = True
                                is_indep, _ = self.test_conditional_independence(i, j, S, cov_mat, N, self.alpha)
                                if is_indep:
                                    adj[i, j] = adj[j, i] = False
                                    sep_sets[(i, j)] = set(S)
                                    sep_sets[(j, i)] = set(S)
                                    break
            if not any_tested:
                break

        # ------------------------------------------------------------------
        # Phase 2: Unshielded Collider (v-structure) Orientation
        # For every unshielded triple X - Z - Y where X and Y are not adjacent:
        # If Z not in SepSet(X, Y), orient as X -> Z <- Y.
        # ------------------------------------------------------------------
        directed = np.zeros((d, d), dtype=bool)
        undirected = adj.copy()

        for i in range(d):
            for j in range(i + 1, d):
                if not adj[i, j]:  # X and Y non-adjacent
                    common_neighbors = [k for k in range(d) if adj[i, k] and adj[j, k]]
                    for k in common_neighbors:
                        # Unshielded triple i - k - j
                        if k not in sep_sets.get((i, j), set()):
                            directed[i, k] = True
                            directed[j, k] = True
                            undirected[i, k] = undirected[k, i] = False
                            undirected[j, k] = undirected[k, j] = False

        # ------------------------------------------------------------------
        # Phase 3: Meek's Orientation Rules (Rule 1 & Rule 2)
        # ------------------------------------------------------------------
        changed = True
        while changed:
            changed = False
            # Rule 1: If i -> k - j and i not adjacent to j, orient k -> j
            for i in range(d):
                for k in range(d):
                    if directed[i, k]:
                        for j in range(d):
                            if undirected[k, j] and not adj[i, j] and j != i:
                                directed[k, j] = True
                                undirected[k, j] = undirected[j, k] = False
                                changed = True

            # Rule 2: If i -> k -> j and i - j, orient i -> j to prevent directed cycle
            for i in range(d):
                for k in range(d):
                    if directed[i, k]:
                        for j in range(d):
                            if directed[k, j] and undirected[i, j]:
                                directed[i, j] = True
                                undirected[i, j] = undirected[j, i] = False
                                changed = True

        # Combine into CPDAG format
        cpdag = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                if directed[i, j]:
                    cpdag[i, j] = 1
                elif undirected[i, j]:
                    cpdag[i, j] = 1
                    cpdag[j, i] = 1

        return cpdag, sep_sets


if __name__ == "__main__":
    print("--- Z05: Causal Discovery via PC Algorithm Sanity Check ---")

    # Generate synthetic observational data from a known Ground-Truth Causal SCM:
    #   X0 -> X2 <- X1   (v-structure / collider at X2)
    #   X2 -> X3         (downstream effect of collider)
    # Total Nodes: 4
    # Ground truth edges: (0 -> 2), (1 -> 2), (2 -> 3)
    # Conditional Independencies:
    #   X0 _||_ X1 | {}
    #   X0 _||_ X3 | {X2}
    #   X1 _||_ X3 | {X2}
    rng = np.random.default_rng(42)
    N = 10000
    x0 = rng.normal(loc=0.0, scale=1.0, size=N)
    x1 = rng.normal(loc=0.0, scale=1.0, size=N)
    x2 = 0.8 * x0 + 0.7 * x1 + rng.normal(loc=0.0, scale=0.4, size=N)
    x3 = 0.9 * x2 + rng.normal(loc=0.0, scale=0.4, size=N)

    dataset = np.vstack([x0, x1, x2, x3]).T

    # Run PC Algorithm
    pc = PCAlgorithm(alpha=0.01)
    cpdag, sep_sets = pc.fit(dataset)

    print("Discovered CPDAG Adjacency Matrix:")
    print(cpdag)

    # Inspect discovered separating sets
    print(f"Separating Set between X0 and X1: {sep_sets.get((0, 1), set())}")
    print(f"Separating Set between X0 and X3: {sep_sets.get((0, 3), set())}")
    print(f"Separating Set between X1 and X3: {sep_sets.get((1, 3), set())}")

    # Assert correct causal skeleton
    assert 2 in sep_sets.get((0, 3), set()), "X0 and X3 must be separated by intermediate mediator X2"
    assert 2 in sep_sets.get((1, 3), set()), "X1 and X3 must be separated by intermediate mediator X2"
    assert len(sep_sets.get((0, 1), set())) == 0, "X0 and X1 must be unconditionally independent"

    # Assert exact causal DAG direction recovery:
    # 0 -> 2, 1 -> 2, 2 -> 3
    assert cpdag[0, 2] == 1 and cpdag[2, 0] == 0, "Must recover directed causal edge 0 -> 2"
    assert cpdag[1, 2] == 1 and cpdag[2, 1] == 0, "Must recover directed causal edge 1 -> 2"
    assert cpdag[2, 3] == 1 and cpdag[3, 2] == 0, "Must recover directed causal edge 2 -> 3 via Meek's Rule 1"

    print("Causal v-structure and downstream edge orientations: 100% MATCH GROUND TRUTH.")
    print("Z05 Sanity Check: PASSED cleanly.")

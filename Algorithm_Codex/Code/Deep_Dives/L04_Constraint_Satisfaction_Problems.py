"""
L04: Constraint Satisfaction Problems (CSP)
Pure Python/NumPy implementation of AC-3 Arc Consistency and Backtracking with MRV.
"""

from collections import deque

class CSP:
    """Constraint Satisfaction Problem with AC-3 and Backtracking."""
    def __init__(self, variables: list[str], domains: dict[str, list], constraints: dict[tuple[str, str], callable]):
        self.variables = variables
        self.domains = {v: list(domains[v]) for v in variables}
        self.constraints = constraints  # (X_i, X_j) -> function(val_i, val_j) -> bool

    def is_consistent(self, var: str, val, assignment: dict) -> bool:
        for other_var, other_val in assignment.items():
            if (var, other_var) in self.constraints:
                if not self.constraints[(var, other_var)](val, other_val):
                    return False
            if (other_var, var) in self.constraints:
                if not self.constraints[(other_var, var)](other_val, val):
                    return False
        return True

    def ac3(self) -> bool:
        """AC-3 Arc Consistency algorithm."""
        queue = deque(list(self.constraints.keys()))

        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                # Re-add neighbors
                for xk in self.variables:
                    if xk != xi and xk != xj and (xk, xi) in self.constraints:
                        queue.append((xk, xi))
        return True

    def _revise(self, xi: str, xj: str) -> bool:
        revised = False
        to_remove = []
        for x in self.domains[xi]:
            # Check if there exists y in D(xj) satisfying constraint
            has_support = any(self.constraints[(xi, xj)](x, y) for y in self.domains[xj])
            if not has_support:
                to_remove.append(x)
                revised = True
        for rem in to_remove:
            self.domains[xi].remove(rem)
        return revised

    def backtrack_search(self) -> dict | None:
        """Backtracking search with MRV heuristic."""
        return self._backtrack({})

    def _backtrack(self, assignment: dict) -> dict | None:
        if len(assignment) == len(self.variables):
            return assignment

        # MRV: Choose unassigned variable with smallest domain
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(self.domains[v]))

        for val in self.domains[var]:
            if self.is_consistent(var, val, assignment):
                assignment[var] = val
                result = self._backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]

        return None


if __name__ == "__main__":
    print("--- L04: CSP & AC-3 Sanity Check ---")
    # Map Coloring: 3 regions (WA, NT, SA) with 2 colors (Red, Green)
    # WA must differ from NT, WA differs from SA, NT differs from SA (Triangle Clique)
    # Coloring a 3-clique with 2 colors is IMPOSSIBLE. AC-3 should detect failure or backtracking will fail.
    vars_3 = ['WA', 'NT', 'SA']
    domains_2 = {'WA': ['R', 'G'], 'NT': ['R', 'G'], 'SA': ['R', 'G']}
    diff_fn = lambda a, b: a != b
    constraints_3 = {
        ('WA', 'NT'): diff_fn, ('NT', 'WA'): diff_fn,
        ('WA', 'SA'): diff_fn, ('SA', 'WA'): diff_fn,
        ('NT', 'SA'): diff_fn, ('SA', 'NT'): diff_fn
    }

    csp_impossible = CSP(vars_3, domains_2, constraints_3)
    sol_imp = csp_impossible.backtrack_search()
    print(f"3-Region 2-Color Solution (Should be None): {sol_imp}")
    assert sol_imp is None, "2 colors cannot color a 3-clique"

    # Now provide 3 colors (Red, Green, Blue) -> Valid solution exists!
    domains_3 = {'WA': ['R', 'G', 'B'], 'NT': ['R', 'G', 'B'], 'SA': ['R', 'G', 'B']}
    csp_possible = CSP(vars_3, domains_3, constraints_3)
    assert csp_possible.ac3() is True, "AC-3 should succeed on valid 3-colorable graph"
    sol_pos = csp_possible.backtrack_search()
    print(f"3-Region 3-Color Discovered Assignment: {sol_pos}")

    assert sol_pos is not None
    assert sol_pos['WA'] != sol_pos['NT'] and sol_pos['WA'] != sol_pos['SA'] and sol_pos['NT'] != sol_pos['SA']
    print("L04 Sanity Check: PASSED cleanly.")

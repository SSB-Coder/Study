"""
W09. SAT Solving — Davis-Putnam-Logemann-Loveland (DPLL)
From-scratch implementation in pure NumPy.
"""

class DPLLSolver:
    """
    Exact Boolean Satisfiability (SAT) solver for formulas in Conjunctive Normal Form (CNF).
    Implements the recursive DPLL algorithm with:
    1. Unit Propagation (Unit Clause rule) <- eq. 3.1
    2. Pure Literal Elimination <- eq. 3.2
    3. Backtracking search with Most Constrained Variable heuristic
    """
    
    @staticmethod
    def simplify(clauses: list, literal: int):
        """
        Assigns literal=True (meaning variable abs(literal) is set to sgn(literal) > 0).
        Removes satisfied clauses; strips negation of literal from remaining clauses.
        """
        new_clauses = []
        for clause in clauses:
            if literal in clause:
                continue # Clause is satisfied -> drop
            if -literal in clause:
                new_clause = [lit for lit in clause if lit != -literal]
                new_clauses.append(new_clause) # False literal removed
            else:
                new_clauses.append(list(clause))
        return new_clauses

    @classmethod
    def dpll(cls, clauses: list, assignment: dict):
        """
        Recursive DPLL solver.
        clauses: list of lists of integers (e.g. [[1, -2], [2, 3]])
        assignment: dict mapping var_int -> bool
        Returns (is_sat, final_assignment)
        """
        # Base case 1: Empty clause in formula -> contradiction/conflict
        if any(len(c) == 0 for c in clauses):
            return False, {}
            
        # Base case 2: No clauses left -> satisfied!
        if len(clauses) == 0:
            return True, assignment

        # 1. Unit Propagation: A clause with a single literal MUST be true <- eq. 3.1
        unit_clauses = [c[0] for c in clauses if len(c) == 1]
        if unit_clauses:
            lit = unit_clauses[0]
            var = abs(lit)
            val = (lit > 0)
            new_assign = dict(assignment)
            new_assign[var] = val
            simplified = cls.simplify(clauses, lit)
            return cls.dpll(simplified, new_assign)

        # 2. Pure Literal Elimination: Literals appearing with only one polarity <- eq. 3.2
        all_literals = [lit for c in clauses for lit in c]
        lit_set = set(all_literals)
        pure_literals = [lit for lit in lit_set if -lit not in lit_set]
        if pure_literals:
            lit = pure_literals[0]
            var = abs(lit)
            val = (lit > 0)
            new_assign = dict(assignment)
            new_assign[var] = val
            simplified = cls.simplify(clauses, lit)
            return cls.dpll(simplified, new_assign)

        # 3. Branching: Choose variable using Maximum Occurrences heuristic
        counts = {}
        for lit in all_literals:
            var = abs(lit)
            counts[var] = counts.get(var, 0) + 1
        branch_var = max(counts.keys(), key=lambda v: counts[v])

        # Try True branch first
        assign_true = dict(assignment)
        assign_true[branch_var] = True
        sat_t, res_t = cls.dpll(cls.simplify(clauses, branch_var), assign_true)
        if sat_t:
            return True, res_t

        # Backtrack: Try False branch
        assign_false = dict(assignment)
        assign_false[branch_var] = False
        sat_f, res_f = cls.dpll(cls.simplify(clauses, -branch_var), assign_false)
        if sat_f:
            return True, res_f

        return False, {}

    @classmethod
    def solve(cls, clauses: list):
        return cls.dpll(clauses, {})

if __name__ == "__main__":
    print("--- W09: DPLL SAT Solver Sanity Check ---")
    # Formula 1: Satisfiable
    # (x1 or x2) and (not x1 or x2) and (x1 or not x2)
    # CNF: [[1, 2], [-1, 2], [1, -2]]
    cnf_sat = [[1, 2], [-1, 2], [1, -2]]
    is_sat1, model1 = DPLLSolver.solve(cnf_sat)
    print(f"CNF 1 Satisfiable: {is_sat1}, Model: {model1}")
    assert is_sat1
    # Verify model satisfies all clauses
    for c in cnf_sat:
        assert any((lit > 0 and model1.get(abs(lit)) is True) or \
                   (lit < 0 and model1.get(abs(lit)) is False) for lit in c)

    # Formula 2: Unsatisfiable (p and not p)
    # (x1) and (not x1)
    cnf_unsat = [[1], [-1]]
    is_sat2, _ = DPLLSolver.solve(cnf_unsat)
    print(f"CNF 2 Satisfiable: {is_sat2}")
    assert not is_sat2, "Expected UNSAT for x1 and not x1"
    print("W09 Sanity Check: PASSED cleanly.")

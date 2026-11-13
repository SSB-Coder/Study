"""
E01. Bayesian Networks (Exact Inference via Variable Elimination)
From-scratch implementation in pure Python/NumPy.
"""

import numpy as np

class Factor:
    """Discrete multi-dimensional probabilistic factor."""
    def __init__(self, variables: list, values: np.ndarray):
        self.variables = variables # List of string variable names
        self.values = np.asarray(values, dtype=np.float64) # Array of probabilities

    def observe(self, var: str, value: int):
        """Fixes evidence variable to a specific discrete state <- eq. 3.1"""
        if var not in self.variables:
            return self
        axis = self.variables.index(var)
        # Slice along axis
        new_values = np.take(self.values, indices=value, axis=axis)
        new_variables = [v for v in self.variables if v != var]
        return Factor(new_variables, new_values)

    def multiply(self, other):
        """Pointwise multiplication of two factors over union of variables <- eq. 3.1"""
        all_vars = list(self.variables)
        for v in other.variables:
            if v not in all_vars:
                all_vars.append(v)

        # Expand self and other to shape of all_vars
        shape = []
        self_slices = []
        other_slices = []

        for v in all_vars:
            dim_self = self.values.shape[self.variables.index(v)] if v in self.variables else 1
            dim_other = other.values.shape[other.variables.index(v)] if v in other.variables else 1
            dim = max(dim_self, dim_other)
            shape.append(dim)

            self_slices.append(slice(None) if v in self.variables else np.newaxis)
            other_slices.append(slice(None) if v in other.variables else np.newaxis)

        # Reshape self values
        self_expanded = self.values.reshape(
            [self.values.shape[self.variables.index(v)] if v in self.variables else 1 for v in all_vars]
        )
        other_expanded = other.values.reshape(
            [other.values.shape[other.variables.index(v)] if v in other.variables else 1 for v in all_vars]
        )

        return Factor(all_vars, self_expanded * other_expanded)

    def marginalize(self, var: str):
        """Sums out a variable <- eq. 3.1"""
        if var not in self.variables:
            return self
        axis = self.variables.index(var)
        new_values = np.sum(self.values, axis=axis)
        new_variables = [v for v in self.variables if v != var]
        return Factor(new_variables, new_values)

class BayesianNetwork:
    """Exact inference engine for discrete Bayesian networks."""
    def __init__(self):
        self.factors = []

    def add_factor(self, factor: Factor):
        self.factors.append(factor)

    def query(self, target_var: str, evidence: dict = None, elimination_order: list = None):
        """Runs Variable Elimination to compute P(target_var | evidence)."""
        if evidence is None:
            evidence = {}

        # 1. Instantiate evidence on all factors
        current_factors = []
        for f in self.factors:
            f_reduced = f
            for var, val in evidence.items():
                f_reduced = f_reduced.observe(var, val)
            current_factors.append(f_reduced)

        # 2. Determine elimination order (all hidden variables except target and evidence)
        all_vars = set()
        for f in current_factors:
            all_vars.update(f.variables)
        hidden_vars = [v for v in all_vars if v != target_var and v not in evidence]

        if elimination_order is not None:
            hidden_vars = [v for v in elimination_order if v in hidden_vars]

        # 3. Eliminate hidden variables
        for h in hidden_vars:
            # Collect factors mentioning h
            factors_with_h = [f for f in current_factors if h in f.variables]
            factors_without_h = [f for f in current_factors if h not in f.variables]

            if not factors_with_h:
                continue

            # Multiply all factors containing h
            prod_factor = factors_with_h[0]
            for f in factors_with_h[1:]:
                prod_factor = prod_factor.multiply(f)

            # Sum out h
            new_factor = prod_factor.marginalize(h)
            factors_without_h.append(new_factor)
            current_factors = factors_without_h

        # 4. Multiply remaining factors containing target_var
        final_factor = current_factors[0]
        for f in current_factors[1:]:
            final_factor = final_factor.multiply(f)

        # Normalize
        probs = final_factor.values / np.sum(final_factor.values)
        return probs

if __name__ == "__main__":
    print("--- E01: Bayesian Networks (Variable Elimination) Sanity Check ---")
    # Classic Alarm Network: Burglary (B) -> Alarm (A) <- Earthquake (E)
    # B in {0, 1}, E in {0, 1}, A in {0, 1}
    # P(B=1) = 0.01, P(E=1) = 0.02
    # P(A=1 | B, E): A=1 if B=1 or E=1 with high prob
    f_B = Factor(["B"], np.array([0.99, 0.01]))
    f_E = Factor(["E"], np.array([0.98, 0.02]))
    # P(A | B, E) shape: B x E x A
    cpt_A = np.zeros((2, 2, 2))
    cpt_A[0, 0, :] = [0.999, 0.001] # No B, No E -> Alarm very rare
    cpt_A[0, 1, :] = [0.71, 0.29]   # Earthquake alone
    cpt_A[1, 0, :] = [0.06, 0.94]   # Burglary alone
    cpt_A[1, 1, :] = [0.05, 0.95]   # Both
    f_A = Factor(["B", "E", "A"], cpt_A)

    bn = BayesianNetwork()
    bn.add_factor(f_B)
    bn.add_factor(f_E)
    bn.add_factor(f_A)

    # Query 1: What is the probability of Burglary given Alarm sounded (A=1)?
    p_b_given_a = bn.query(target_var="B", evidence={"A": 1})
    print(f"P(Burglary=1 | Alarm=1): {p_b_given_a[1]:.4f}")

    # Query 2: "Explaining Away" effect: What happens if we also know an Earthquake occurred (A=1, E=1)?
    p_b_given_a_e = bn.query(target_var="B", evidence={"A": 1, "E": 1})
    print(f"P(Burglary=1 | Alarm=1, Earthquake=1): {p_b_given_a_e[1]:.4f}")

    # The earthquake must explain away the burglary (probability should drop)
    assert p_b_given_a_e[1] < p_b_given_a[1]
    assert np.isclose(np.sum(p_b_given_a), 1.0)
    print("Explaining-away phenomenon verified successfully!")
    print("E01 Sanity Check: PASSED cleanly.")

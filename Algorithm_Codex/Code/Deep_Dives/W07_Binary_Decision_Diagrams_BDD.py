"""
W07. Binary Decision Diagrams (ROBDDs) & Formal Equivalence Checking
From-scratch implementation in pure NumPy.
"""

class BDDNode:
    def __init__(self, var: str, low: int, high: int):
        self.var = var      # Variable name (e.g. 'x0', 'x1')
        self.low = low      # Node index when var == 0
        self.high = high    # Node index when var == 1
    def __repr__(self):
        return f"Node({self.var}, low={self.low}, high={self.high})"

class ROBDD:
    """
    Reduced Ordered Binary Decision Diagram (ROBDD).
    Guarantees a canonical representation for any Boolean function under a fixed variable ordering.
    Two functions are logically equivalent if and only if their ROBDD root node IDs are identical!
    """
    
    def __init__(self, var_order: list):
        self.var_order = var_order
        self.var_level = {v: i for i, v in enumerate(var_order)}
        
        # Terminal nodes: 0 (False) and 1 (True)
        self.nodes = [BDDNode('0', 0, 0), BDDNode('1', 1, 1)]
        # Unique table: (var, low, high) -> node_id
        self.unique_table = {}
        # Computed table for Apply cache: (op, id1, id2) -> result_id
        self.computed_table = {}

    def get_node(self, var: str, low: int, high: int) -> int:
        """
        Reduction rules:
        1. Elimination rule: If low == high, redundant test -> return low <- eq. 3.1
        2. Isomorphism rule: If identical node exists in unique table, return existing id <- eq. 3.2
        """
        if low == high:
            return low
        key = (var, low, high)
        if key in self.unique_table:
            return self.unique_table[key]
        node_id = len(self.nodes)
        self.nodes.append(BDDNode(var, low, high))
        self.unique_table[key] = node_id
        return node_id

    def variable(self, var: str) -> int:
        """Creates BDD for a single literal x: if x=0 -> 0, if x=1 -> 1."""
        return self.get_node(var, low=0, high=1)

    def constant(self, val: bool) -> int:
        return 1 if val else 0

    def apply(self, op: str, u1: int, u2: int) -> int:
        """
        Bryant's Apply operation: Combines two BDDs using boolean operator
        op in {'AND', 'OR', 'XOR', 'NAND', 'IFF'}.
        """
        key = (op, u1, u2)
        if key in self.computed_table:
            return self.computed_table[key]

        # Base cases: both terminals
        if u1 in (0, 1) and u2 in (0, 1):
            b1 = bool(u1)
            b2 = bool(u2)
            if op == 'AND': res = b1 and b2
            elif op == 'OR': res = b1 or b2
            elif op == 'XOR': res = b1 != b2
            elif op == 'IFF': res = b1 == b2
            elif op == 'NAND': res = not (b1 and b2)
            else: raise ValueError(f"Unknown op {op}")
            return self.constant(res)

        # Determine top variable
        v1 = self.nodes[u1].var if u1 not in (0, 1) else None
        v2 = self.nodes[u2].var if u2 not in (0, 1) else None
        
        lvl1 = self.var_level[v1] if v1 else float('inf')
        lvl2 = self.var_level[v2] if v2 else float('inf')
        
        if lvl1 <= lvl2:
            top_var = v1
        else:
            top_var = v2
            
        # Shannon cofactor expansion <- eq. 3.3
        if v1 == top_var:
            u1_low, u1_high = self.nodes[u1].low, self.nodes[u1].high
        else:
            u1_low, u1_high = u1, u1
            
        if v2 == top_var:
            u2_low, u2_high = self.nodes[u2].low, self.nodes[u2].high
        else:
            u2_low, u2_high = u2, u2

        low = self.apply(op, u1_low, u2_low)
        high = self.apply(op, u1_high, u2_high)
        res_id = self.get_node(top_var, low, high)
        
        self.computed_table[key] = res_id
        return res_id

    def are_equivalent(self, u1: int, u2: int) -> bool:
        """Due to canonicity, two BDDs are logically equivalent iff their root IDs match."""
        return u1 == u2

if __name__ == "__main__":
    print("--- W07: Reduced Ordered BDD Sanity Check ---")
    # Variable ordering: x0 < x1 < x2
    bdd = ROBDD(var_order=['x0', 'x1', 'x2'])
    
    x0 = bdd.variable('x0')
    x1 = bdd.variable('x1')
    x2 = bdd.variable('x2')
    
    # Prove De Morgan's Law via ROBDD canonicity:
    # Not (A and B) == (Not A) or (Not B)
    # 1. f1 = NAND(x0, x1)
    f1 = bdd.apply('NAND', x0, x1)
    
    # 2. f2 = NOT(x0) OR NOT(x1)
    not_x0 = bdd.apply('XOR', x0, bdd.constant(True))
    not_x1 = bdd.apply('XOR', x1, bdd.constant(True))
    f2 = bdd.apply('OR', not_x0, not_x1)
    
    print(f"BDD Root ID for f1 = NAND(x0, x1): {f1}")
    print(f"BDD Root ID for f2 = (~x0 | ~x1)  : {f2}")
    
    # Strict canonical identity check!
    assert bdd.are_equivalent(f1, f2), "De Morgan equivalence failed!"
    
    # Check tautology: f XOR f == 0
    taut = bdd.apply('XOR', f1, f2)
    assert taut == 0, "Self-XOR must yield 0 terminal"
    print("W07 Sanity Check: PASSED cleanly.")

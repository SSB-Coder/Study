"""
U05. Circuit Nodal Analysis — Modified Nodal Analysis (MNA)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class CircuitMNA:
    """
    Solves linear DC electric circuits using Modified Nodal Analysis (MNA).
    Handles:
    - Resistors (R)
    - Independent Current Sources (I)
    - Independent Voltage Sources (V)
    Formulates and solves: [G, B; B^T, D] * [v; j] = [i; e]
    """
    
    def __init__(self, num_nodes: int):
        self.n_nodes = num_nodes # Node 0 is Ground reference
        self.resistors = []       # (node1, node2, resistance)
        self.current_sources = [] # (node_from, node_to, current)
        self.voltage_sources = [] # (node_pos, node_neg, voltage)

    def add_resistor(self, n1: int, n2: int, value: float):
        self.resistors.append((n1, n2, float(value)))

    def add_current_source(self, n_from: int, n_to: int, value: float):
        self.current_sources.append((n_from, n_to, float(value)))

    def add_voltage_source(self, n_pos: int, n_neg: int, value: float):
        self.voltage_sources.append((n_pos, n_neg, float(value)))

    def solve(self):
        """
        Assembles MNA block matrices and solves for node voltages and branch currents.
        """
        n = self.n_nodes - 1 # Non-ground nodes (1 ... n_nodes-1)
        m = len(self.voltage_sources)
        
        G = np.zeros((n, n), dtype=float) # Conductance matrix <- eq. 3.1
        B = np.zeros((n, m), dtype=float) # Voltage source incidence matrix <- eq. 3.2
        D = np.zeros((m, m), dtype=float) # Zero for ideal independent voltage sources
        i_rhs = np.zeros(n, dtype=float)
        e_rhs = np.zeros(m, dtype=float)
        
        # 1. Fill Conductance Matrix G
        for n1, n2, R in self.resistors:
            g = 1.0 / R
            # Node 0 is ground (omitted from equations)
            if n1 != 0:
                G[n1 - 1, n1 - 1] += g
            if n2 != 0:
                G[n2 - 1, n2 - 1] += g
            if n1 != 0 and n2 != 0:
                G[n1 - 1, n2 - 1] -= g
                G[n2 - 1, n1 - 1] -= g
                
        # 2. Fill Current Sources RHS
        for n_from, n_to, I in self.current_sources:
            if n_from != 0:
                i_rhs[n_from - 1] -= I # Current leaves n_from
            if n_to != 0:
                i_rhs[n_to - 1] += I   # Current enters n_to
                
        # 3. Fill Voltage Sources B and e_rhs
        for k, (n_pos, n_neg, V) in enumerate(self.voltage_sources):
            if n_pos != 0:
                B[n_pos - 1, k] = 1.0
            if n_neg != 0:
                B[n_neg - 1, k] = -1.0
            e_rhs[k] = V
            
        # 4. Assemble Global MNA Matrix
        A_mna = np.block([
            [G, B],
            [B.T, D]
        ])
        z_rhs = np.concatenate([i_rhs, e_rhs])
        
        # 5. Solve linear system via Gaussian elimination
        sol = np.linalg.solve(A_mna, z_rhs)
        
        node_voltages = {0: 0.0} # Ground
        for i in range(n):
            node_voltages[i + 1] = sol[i]
            
        voltage_currents = sol[n:]
        return node_voltages, voltage_currents

if __name__ == "__main__":
    print("--- U05: Circuit Modified Nodal Analysis Sanity Check ---")
    # Wheatstone bridge circuit:
    # Nodes: 0 (GND), 1 (Vin), 2 (Left tap), 3 (Right tap)
    # 10V Source between Node 1 (+) and Node 0 (-)
    # R1 between 1 and 2 (100 ohm)
    # R2 between 2 and 0 (100 ohm) -> V2 = 5V
    # R3 between 1 and 3 (100 ohm)
    # R4 between 3 and 0 (100 ohm) -> V3 = 5V
    # Bridge resistor R5 between 2 and 3 (50 ohm) -> balanced bridge, V2 - V3 = 0, current = 0
    ckt = CircuitMNA(num_nodes=4)
    ckt.add_voltage_source(n_pos=1, n_neg=0, value=10.0)
    ckt.add_resistor(1, 2, value=100.0)
    ckt.add_resistor(2, 0, value=100.0)
    ckt.add_resistor(1, 3, value=100.0)
    ckt.add_resistor(3, 0, value=100.0)
    ckt.add_resistor(2, 3, value=50.0)
    
    voltages, v_currents = ckt.solve()
    print(f"Node Voltages: {voltages}")
    print(f"Total Current drawn from 10V Source: {abs(v_currents[0]):.4f} A")
    
    # Asserts:
    # 1. Node 1 voltage must be 10.0V
    assert np.isclose(voltages[1], 10.0, atol=1e-8)
    # 2. Balanced bridge: Node 2 and Node 3 must both be exactly 5.0V
    assert np.isclose(voltages[2], 5.0, atol=1e-8)
    assert np.isclose(voltages[3], 5.0, atol=1e-8)
    # 3. Total resistance = 100 ohm || 100 ohm = 100 ohm total equivalent -> I = 10V / 100 ohm = 0.1A
    assert np.isclose(abs(v_currents[0]), 0.1, atol=1e-6)
    print("U05 Sanity Check: PASSED cleanly.")

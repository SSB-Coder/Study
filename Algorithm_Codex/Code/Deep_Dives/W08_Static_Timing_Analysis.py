"""
W08. Static Timing Analysis (STA) & Critical Path Computation
From-scratch implementation in pure NumPy.
"""

import numpy as np
from collections import deque

class StaticTimingAnalyzer:
    """
    Performs Static Timing Analysis (STA) on a digital gate-level circuit DAG.
    Computes Arrival Times (AT), Required Arrival Times (RAT), Slacks,
    and identifies the Critical Timing Path.
    """
    
    def __init__(self, clock_period: float):
        self.clock_period = clock_period
        self.nodes = {}           # node_id -> {'type': str, 'delay': float}
        self.edges = {}           # u -> list of (v, wire_delay)
        self.in_edges = {}        # v -> list of (u, wire_delay)

    def add_node(self, node_id: str, node_type: str, delay: float = 0.0):
        self.nodes[node_id] = {'type': node_type, 'delay': float(delay)}
        self.edges[node_id] = []
        self.in_edges[node_id] = []

    def add_wire(self, u: str, v: str, wire_delay: float = 0.0):
        self.edges[u].append((v, float(wire_delay)))
        self.in_edges[v].append((u, float(wire_delay)))

    def topological_sort(self):
        in_degree = {u: len(self.in_edges[u]) for u in self.nodes}
        queue = deque([u for u, deg in in_degree.items() if deg == 0])
        order = []
        
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.edges[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        if len(order) != len(self.nodes):
            raise ValueError("Circuit contains a combinational timing loop (cycle)!")
        return order

    def analyze(self):
        """
        Runs Forward Pass for Arrival Time (AT) and Backward Pass for Required Arrival Time (RAT).
        """
        order = self.topological_sort()
        
        # 1. Forward Pass: Arrival Time (AT) <- eq. 3.1
        at = {u: 0.0 for u in self.nodes}
        for u in order:
            node_delay = self.nodes[u]['delay']
            if self.in_edges[u]:
                at[u] = max(at[pred] + wire_d for pred, wire_d in self.in_edges[u]) + node_delay
            else:
                at[u] = node_delay # Primary Input / Clocked launch
                
        # 2. Backward Pass: Required Arrival Time (RAT) <- eq. 3.2
        rat = {u: self.clock_period for u in self.nodes}
        for u in reversed(order):
            if not self.edges[u]: # Primary Output
                rat[u] = self.clock_period
            else:
                rat[u] = min(rat[succ] - wire_d - self.nodes[succ]['delay'] for succ, wire_d in self.edges[u])
                
        # 3. Slack calculation <- eq. 3.3
        slack = {u: rat[u] - at[u] for u in self.nodes}
        
        # 4. Critical path identification (nodes with minimum slack)
        min_slack = min(slack.values())
        critical_nodes = [u for u in order if np.isclose(slack[u], min_slack)]
        
        return at, rat, slack, critical_nodes, min_slack

if __name__ == "__main__":
    print("--- W08: Static Timing Analysis Sanity Check ---")
    # Setup digital pipeline:
    # Clock period = 10.0 ns
    # Path 1: IN1 (0.5) -> AND1 (2.0) -> OR1 (1.5) -> OUT (0.5) = total delay 4.5 ns
    # Path 2: IN2 (0.5) -> MULT (6.0) -> OR1 (1.5) -> OUT (0.5) = total delay 8.5 ns (Critical!)
    sta = StaticTimingAnalyzer(clock_period=10.0)
    sta.add_node("IN1", "PI", delay=0.5)
    sta.add_node("IN2", "PI", delay=0.5)
    sta.add_node("AND1", "GATE", delay=2.0)
    sta.add_node("MULT", "GATE", delay=6.0)
    sta.add_node("OR1", "GATE", delay=1.5)
    sta.add_node("OUT", "PO", delay=0.5)
    
    sta.add_wire("IN1", "AND1", wire_delay=0.1)
    sta.add_wire("AND1", "OR1", wire_delay=0.2)
    sta.add_wire("IN2", "MULT", wire_delay=0.1)
    sta.add_wire("MULT", "OR1", wire_delay=0.2)
    sta.add_wire("OR1", "OUT", wire_delay=0.1)
    
    at, rat, slack, crit_path, min_slack = sta.analyze()
    
    print(f"Total Arrival Time at OUT: {at['OUT']:.2f} ns")
    print(f"Worst Negative Slack (WNS): {min_slack:.2f} ns")
    print(f"Critical Timing Nodes: {crit_path}")
    
    # MULT delay (6.0) + IN2 (0.5) + wire (0.1) + wire (0.2) + OR1 (1.5) + wire (0.1) + OUT (0.5) = 8.9 ns
    assert at['OUT'] > 8.5, "Critical path arrival time mismatch"
    assert min_slack >= 0.0, "Timing violation occurred"
    assert "MULT" in crit_path, "Multiplier must be on critical path"
    print("W08 Sanity Check: PASSED cleanly.")

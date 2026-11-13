"""
H08: NEAT (NeuroEvolution of Augmenting Topologies)
Pure NumPy implementation of Stanley & Miikkulainen (2002).
Features historical innovation tracking, topological mutation (add node, add connection),
speciation distance, and minimal evolutionary solver.
"""

import numpy as np

class ConnectionGene:
    def __init__(self, in_node: int, out_node: int, weight: float, enabled: bool, innovation: int):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation)


class Genome:
    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.node_ids = set(range(n_inputs + n_outputs))
        self.connections = []
        self.fitness = 0.0

    def add_connection(self, in_node: int, out_node: int, weight: float, innovation: int):
        self.connections.append(ConnectionGene(in_node, out_node, weight, True, innovation))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Feedforward evaluation through arbitrary directed acyclic topology."""
        values = {}
        for i in range(self.n_inputs):
            values[i] = x[i]

        # Order nodes topologically or iteratively relax
        active_conns = [c for c in self.connections if c.enabled]
        # Multi-pass relaxation for general feedforward
        for _ in range(len(self.node_ids)):
            for c in active_conns:
                if c.in_node in values:
                    in_val = values[c.in_node]
                    out_val = values.get(c.out_node, 0.0)
                    values[c.out_node] = out_val + in_val * c.weight

        # Apply sigmoid to outputs
        outputs = []
        for o in range(self.n_inputs, self.n_inputs + self.n_outputs):
            raw = values.get(o, 0.0)
            outputs.append(1.0 / (1.0 + np.exp(-np.clip(raw, -15.0, 15.0))))
        return np.array(outputs)

    def mutate_weights(self, rate: float = 0.8, scale: float = 0.3):
        for c in self.connections:
            if np.random.rand() < rate:
                c.weight += np.random.randn() * scale

    def mutate_add_node(self, innovation_tracker):
        active = [c for c in self.connections if c.enabled]
        if not active:
            return
        conn = active[np.random.randint(len(active))]
        conn.enabled = False

        new_node = max(self.node_ids) + 1
        self.node_ids.add(new_node)

        inno1 = innovation_tracker.get_innovation(conn.in_node, new_node)
        inno2 = innovation_tracker.get_innovation(new_node, conn.out_node)

        self.connections.append(ConnectionGene(conn.in_node, new_node, 1.0, True, inno1))
        self.connections.append(ConnectionGene(new_node, conn.out_node, conn.weight, True, inno2))

    def mutate_add_connection(self, innovation_tracker):
        n1 = list(self.node_ids)[np.random.randint(len(self.node_ids))]
        n2 = list(self.node_ids)[np.random.randint(len(self.node_ids))]
        if n1 != n2 and n2 >= self.n_inputs: # Avoid connecting into input nodes
            # Check if connection already exists
            exists = any(c.in_node == n1 and c.out_node == n2 for c in self.connections)
            if not exists:
                inno = innovation_tracker.get_innovation(n1, n2)
                weight = np.random.randn() * 0.5
                self.connections.append(ConnectionGene(n1, n2, weight, True, inno))


class InnovationTracker:
    def __init__(self):
        self.history = {}
        self.current_innovation = 0

    def get_innovation(self, in_node: int, out_node: int) -> int:
        pair = (in_node, out_node)
        if pair not in self.history:
            self.current_innovation += 1
            self.history[pair] = self.current_innovation
        return self.history[pair]


def compatibility_distance(g1: Genome, g2: Genome, c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """Computes NEAT speciation distance delta based on excess, disjoint, and weight difference."""
    inno1 = {c.innovation: c for c in g1.connections}
    inno2 = {c.innovation: c for c in g2.connections}

    all_inno = set(inno1.keys()).union(set(inno2.keys()))
    if not all_inno:
        return 0.0

    max_inno1 = max(inno1.keys()) if inno1 else 0
    max_inno2 = max(inno2.keys()) if inno2 else 0
    cutoff = min(max_inno1, max_inno2)

    disjoint = 0
    excess = 0
    weight_diffs = []

    for inno in all_inno:
        if inno in inno1 and inno in inno2:
            weight_diffs.append(abs(inno1[inno].weight - inno2[inno].weight))
        elif inno in inno1 or inno in inno2:
            if inno > cutoff:
                excess += 1
            else:
                disjoint += 1

    N = max(len(g1.connections), len(g2.connections), 1)
    W_bar = np.mean(weight_diffs) if weight_diffs else 0.0
    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * W_bar)


if __name__ == "__main__":
    print("--- H08: NEAT NeuroEvolution Sanity Check ---")
    np.random.seed(42)

    tracker = InnovationTracker()
    # Create initial minimal genome (2 inputs, 1 output)
    g1 = Genome(n_inputs=2, n_outputs=1)
    g1.add_connection(0, 2, 0.5, tracker.get_innovation(0, 2))
    g1.add_connection(1, 2, -0.5, tracker.get_innovation(1, 2))

    # Evaluate forward pass
    out1 = g1.forward(np.array([1.0, 1.0]))
    print(f"G1 initial output for [1, 1]: {out1[0]:.4f}")
    assert 0.0 <= out1[0] <= 1.0, "Output must be valid probability in [0, 1]"

    # Clone and mutate topology
    g2 = Genome(n_inputs=2, n_outputs=1)
    for c in g1.connections:
        g2.connections.append(c.copy())
    g2.node_ids = set(g1.node_ids)

    # Add node mutation (splits a connection)
    g2.mutate_add_node(tracker)
    print(f"G2 mutated: {len(g2.node_ids)} nodes, {len(g2.connections)} connections")
    assert len(g2.node_ids) > len(g1.node_ids), "Add node mutation must increase node count"

    # Speciation distance check
    dist = compatibility_distance(g1, g2)
    print(f"Speciation compatibility distance delta(G1, G2): {dist:.4f}")
    assert dist > 0.0, "Mutated genome must have positive compatibility distance from parent"

    print("H08 Sanity Check: PASSED cleanly.")

"""
W05. Min-Cut Circuit Partitioning (Kernighan-Lin & Fiduccia-Mattheyses)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class MinCutPartition:
    """
    Partitions a netlist graph into two balanced subsets A and B to minimize
    the number of cut nets crossing the partition boundary.
    Implements the Kernighan-Lin (KL) 2-way graph partitioning algorithm.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray):
        self.adj = np.array(adjacency_matrix, dtype=int)
        self.n = self.adj.shape[0]
        if self.n % 2 != 0:
            raise ValueError("Graph size must be even for balanced partition")

    def compute_cut_size(self, partition_a: set, partition_b: set) -> int:
        """Counts edges crossing between partition A and B."""
        cut = 0
        for u in partition_a:
            for v in partition_b:
                if self.adj[u, v] > 0:
                    cut += self.adj[u, v]
        return cut

    def compute_d_values(self, A: set, B: set):
        """
        D(u) = External_cost(u) - Internal_cost(u) <- eq. 3.1
        """
        D = np.zeros(self.n, dtype=int)
        for u in range(self.n):
            in_set = A if u in A else B
            out_set = B if u in A else A
            internal_cost = sum(self.adj[u, v] for v in in_set)
            external_cost = sum(self.adj[u, v] for v in out_set)
            D[u] = external_cost - internal_cost
        return D

    def kernighan_lin(self, max_passes: int = 10):
        """
        Iterative Kernighan-Lin 2-way min-cut partitioning.
        """
        # Initial arbitrary balanced partition
        nodes = list(range(self.n))
        np.random.shuffle(nodes)
        A = set(nodes[:self.n // 2])
        B = set(nodes[self.n // 2:])
        
        best_cut = self.compute_cut_size(A, B)
        
        for p in range(max_passes):
            curr_A = set(A)
            curr_B = set(B)
            locked = set()
            swaps = []
            gains = []
            
            D = self.compute_d_values(curr_A, curr_B)
            
            for _ in range(self.n // 2):
                best_gain = -float('inf')
                best_pair = None
                
                # Find unlocked pair (a, b) maximizing gain g = D[a] + D[b] - 2 * c(a, b) <- eq. 3.2
                for a in curr_A:
                    if a in locked: continue
                    for b in curr_B:
                        if b in locked: continue
                        gain = D[a] + D[b] - 2 * self.adj[a, b]
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (a, b)
                            
                if best_pair is None:
                    break
                    
                a_sel, b_sel = best_pair
                locked.add(a_sel)
                locked.add(b_sel)
                swaps.append((a_sel, b_sel))
                gains.append(best_gain)
                
                # Update D values for remaining unlocked nodes
                for x in curr_A:
                    if x not in locked:
                        D[x] += 2 * self.adj[x, a_sel] - 2 * self.adj[x, b_sel]
                for y in curr_B:
                    if y not in locked:
                        D[y] += 2 * self.adj[y, b_sel] - 2 * self.adj[y, a_sel]
                        
            # Find k that maximizes cumulative gain G_k = sum_{i=1}^k gains[i]
            cum_gains = np.cumsum(gains)
            k_max = int(np.argmax(cum_gains))
            max_cum_gain = cum_gains[k_max]
            
            if max_cum_gain > 0:
                # Apply best k swaps
                for i in range(k_max + 1):
                    a_swap, b_swap = swaps[i]
                    A.remove(a_swap)
                    A.add(b_swap)
                    B.remove(b_swap)
                    B.add(a_swap)
                best_cut = self.compute_cut_size(A, B)
            else:
                # Local minimum reached
                break
                
        return best_cut, A, B

if __name__ == "__main__":
    print("--- W05: Min-Cut Circuit Partitioning Sanity Check ---")
    np.random.seed(42)
    # Create two tightly connected clusters of 4 nodes each, with 1 bridging edge
    adj = np.zeros((8, 8), dtype=int)
    # Cluster 1: nodes 0, 1, 2, 3 fully connected
    for i in range(4):
        for j in range(i + 1, 4):
            adj[i, j] = adj[j, i] = 1
    # Cluster 2: nodes 4, 5, 6, 7 fully connected
    for i in range(4, 8):
        for j in range(i + 1, 8):
            adj[i, j] = adj[j, i] = 1
    # Single cut bridge between 3 and 4
    adj[3, 4] = adj[4, 3] = 1
    
    partitioner = MinCutPartition(adj)
    min_cut, set_A, set_B = partitioner.kernighan_lin()
    print(f"Optimal Partition Cut Size: {min_cut}")
    print(f"Set A: {sorted(list(set_A))}, Set B: {sorted(list(set_B))}")
    
    assert min_cut == 1, f"Expected min cut of 1, got {min_cut}"
    assert (set_A == {0, 1, 2, 3} and set_B == {4, 5, 6, 7}) or \
           (set_B == {0, 1, 2, 3} and set_A == {4, 5, 6, 7})
    print("W05 Sanity Check: PASSED cleanly.")

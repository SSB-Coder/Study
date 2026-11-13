"""
T03. Molecular Fingerprints & Chemical Tanimoto Similarity (ECFP / Morgan)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class MolecularFingerprint:
    """
    Computes circular Extended-Connectivity Fingerprints (ECFP / Morgan-style)
    on hand-represented molecular graphs (atoms as nodes, bonds as edges).
    Computes Tanimoto / Jaccard chemical similarity between molecules.
    """
    
    def __init__(self, n_bits: int = 128, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius

    @staticmethod
    def _hash_combine(seed: int, val: int) -> int:
        """64-bit integer hash combination function."""
        # Murmur-like mixing
        seed = int(seed) ^ (int(val) + 0x9e3779b9 + ((seed << 6) & 0xFFFFFFFF) + (seed >> 2))
        return seed & 0xFFFFFFFF

    def compute_ecfp(self, atomic_numbers: list, bonds: list) -> np.ndarray:
        """
        Computes folded binary Morgan fingerprint <- eq. 3.1.
        atomic_numbers: list of atomic numbers (e.g. C=6, O=8, N=7).
        bonds: list of (atom1, atom2, bond_order) tuples (1=single, 2=double).
        """
        n_atoms = len(atomic_numbers)
        adj = {i: [] for i in range(n_atoms)}
        for u, v, order in bonds:
            adj[u].append((v, order))
            adj[v].append((u, order))
            
        # Initial atom invariants (atomic number + heavy degree) <- eq. 3.2
        atom_identifiers = []
        for i in range(n_atoms):
            degree = len(adj[i])
            h = self._hash_combine(atomic_numbers[i], degree)
            atom_identifiers.append(h)
            
        feature_hashes = set(atom_identifiers)
        
        # Iterative circular neighborhood expansion (Morgan radius)
        for r in range(1, self.radius + 1):
            next_identifiers = []
            for i in range(n_atoms):
                # Collect neighbor descriptors: (bond_order, neighbor_hash)
                neighbor_info = []
                for v, order in adj[i]:
                    neighbor_info.append((order, atom_identifiers[v]))
                # Sort neighbors to ensure canonical invariance to atom ordering
                neighbor_info.sort()
                
                # Hash current atom + radius + sorted neighbors
                curr_h = self._hash_combine(atom_identifiers[i], r)
                for order, n_hash in neighbor_info:
                    curr_h = self._hash_combine(curr_h, order)
                    curr_h = self._hash_combine(curr_h, n_hash)
                    
                next_identifiers.append(curr_h)
                feature_hashes.add(curr_h)
                
            atom_identifiers = next_identifiers
            
        # Fold feature hashes into fixed binary bit vector via modulo
        bit_vector = np.zeros(self.n_bits, dtype=int)
        for h in feature_hashes:
            bit_idx = h % self.n_bits
            bit_vector[bit_idx] = 1
            
        return bit_vector

    @staticmethod
    def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Tanimoto / Jaccard coefficient between two binary fingerprints <- eq. 3.3:
        T(A, B) = |A and B| / |A or B|
        """
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        if union == 0:
            return 1.0
        return float(intersection) / float(union)

if __name__ == "__main__":
    print("--- T03: Molecular Fingerprints & Tanimoto Similarity Sanity Check ---")
    fp_gen = MolecularFingerprint(n_bits=128, radius=2)
    
    # Molecule 1: Ethanol (C - C - O)
    # Atoms: 0: C(6), 1: C(6), 2: O(8)
    # Bonds: (0, 1, 1), (1, 2, 1)
    atoms_ethanol = [6, 6, 8]
    bonds_ethanol = [(0, 1, 1), (1, 2, 1)]
    fp_ethanol = fp_gen.compute_ecfp(atoms_ethanol, bonds_ethanol)
    
    # Molecule 2: Isopropanol (C - C(OH) - C)
    # Atoms: 0: C(6), 1: C(6), 2: C(6), 3: O(8)
    # Bonds: (0, 1, 1), (1, 2, 1), (1, 3, 1)
    atoms_isopropanol = [6, 6, 6, 8]
    bonds_isopropanol = [(0, 1, 1), (1, 2, 1), (1, 3, 1)]
    fp_isopropanol = fp_gen.compute_ecfp(atoms_isopropanol, bonds_isopropanol)
    
    # Molecule 3: Benzene ring (6 Carbon ring with alternating single/double bonds)
    atoms_benzene = [6, 6, 6, 6, 6, 6]
    bonds_benzene = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1)]
    fp_benzene = fp_gen.compute_ecfp(atoms_benzene, bonds_benzene)
    
    # Compute Tanimoto similarities
    sim_self = MolecularFingerprint.tanimoto_similarity(fp_ethanol, fp_ethanol)
    sim_eth_iso = MolecularFingerprint.tanimoto_similarity(fp_ethanol, fp_isopropanol)
    sim_eth_benz = MolecularFingerprint.tanimoto_similarity(fp_ethanol, fp_benzene)
    
    print(f"Self-similarity (Ethanol vs Ethanol): {sim_self:.4f}")
    print(f"Alcohol similarity (Ethanol vs Isopropanol): {sim_eth_iso:.4f}")
    print(f"Cross-class similarity (Ethanol vs Benzene): {sim_eth_benz:.4f}")
    
    # Asserts:
    assert np.isclose(sim_self, 1.0), "Self-similarity must be 1.0"
    # Ethanol and Isopropanol share the C(sp3) and O(OH) motifs -> higher similarity than benzene
    assert sim_eth_iso > 0.12, "Related alcohols must share substructure features"
    # Ethanol and Benzene are structurally distinct -> lower similarity
    assert sim_eth_benz < sim_eth_iso, "Benzene cannot be more similar to Ethanol than Isopropanol"
    print("T03 Sanity Check: PASSED cleanly.")

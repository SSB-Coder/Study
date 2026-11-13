"""
R07: Tensor Networks and Matrix Product States (MPS)
Pure NumPy implementation of Matrix Product States (MPS) for 1D quantum many-body systems.
Includes Schmidt decomposition, entanglement entropy, SVD bond truncation, and canonical forms.
"""

import numpy as np

class MatrixProductState:
    """
    Open-boundary Matrix Product State for L sites of physical dimension d=2.
    Tensors: list of 3D arrays A[i] with shape (chi_left, d, chi_right).
    """
    def __init__(self, tensors: list[np.ndarray]):
        self.tensors = tensors
        self.L = len(tensors)

    @classmethod
    def from_full_state(cls, psi: np.ndarray, L: int, max_chi: int = 16) -> "MatrixProductState":
        """
        Decompose a full 2^L state vector into an MPS via successive SVDs.
        """
        d = 2
        psi = np.asarray(psi, dtype=complex).reshape([d] * L)
        tensors = []
        
        # Iterative left-to-right SVD Schmidt decomposition
        chi_left = 1
        curr = psi
        for site in range(L - 1):
            # Reshape into matrix: (chi_left * d, remaining)
            curr = curr.reshape(chi_left * d, -1)
            U, S, Vh = np.linalg.svd(curr, full_matrices=False)
            
            # Truncate bond dimension
            chi_right = min(len(S), max_chi)
            U = U[:, :chi_right]
            S = S[:chi_right]
            Vh = Vh[:chi_right, :]
            
            # Save A[site] with shape (chi_left, d, chi_right)
            A = U.reshape(chi_left, d, chi_right)
            tensors.append(A)
            
            # Update curr for next site: absorb singular values into remaining tensor
            curr = np.diag(S) @ Vh
            chi_left = chi_right
            
        # Last tensor has shape (chi_left, d, 1)
        tensors.append(curr.reshape(chi_left, d, 1))
        return cls(tensors)

    def to_full_state(self) -> np.ndarray:
        """Contract MPS tensors to reconstruct full 2^L state vector."""
        curr = self.tensors[0]  # shape (1, d, chi_1)
        for i in range(1, self.L):
            next_t = self.tensors[i]  # shape (chi_i, d, chi_{i+1})
            # Contract curr along its right bond with next_t along its left bond
            curr = np.tensordot(curr, next_t, axes=(-1, 0))
            # curr now has shape (..., d_curr, d_next, chi_{i+1})
        # Last bond dimension is 1
        return curr.flatten()

    def norm(self) -> float:
        """Compute state norm <psi|psi> by contracting the tensor network."""
        vec = self.to_full_state()
        return float(np.linalg.norm(vec))

    def entanglement_entropy(self, bipartition_site: int) -> float:
        """
        Compute von Neumann entanglement entropy S = -sum lambda_i^2 ln(lambda_i^2)
        across bipartition [0:bipartition_site] and [bipartition_site:L].
        """
        # Reshape full state across the cut
        psi = self.to_full_state().reshape(1 << bipartition_site, -1)
        _, S, _ = np.linalg.svd(psi, full_matrices=False)
        # Normalize singular values
        lambdas = S / np.linalg.norm(S)
        lambdas = lambdas[lambdas > 1e-12]
        return float(-np.sum(lambdas**2 * np.log(lambdas**2)))

if __name__ == "__main__":
    print("--- R07: Matrix Product States (MPS) Sanity Check ---")
    
    # 1. Product state |0000> (L=4): Zero entanglement everywhere
    state_product = np.zeros(16, dtype=complex)
    state_product[0] = 1.0
    mps_prod = MatrixProductState.from_full_state(state_product, L=4, max_chi=4)
    s_prod = mps_prod.entanglement_entropy(bipartition_site=2)
    print(f"Product state entanglement entropy: {s_prod:.6f} (Expected: 0.0)")
    assert np.isclose(s_prod, 0.0), "Product state must have zero entanglement entropy"
    
    # 2. Reconstructed state fidelity
    reconstructed_prod = mps_prod.to_full_state()
    assert np.allclose(state_product, reconstructed_prod), "Reconstructed product state mismatch"
    
    # 3. Entangled GHZ state (|0000> + |1111>)/sqrt(2)
    state_ghz = np.zeros(16, dtype=complex)
    state_ghz[0] = 1.0 / np.sqrt(2.0)
    state_ghz[15] = 1.0 / np.sqrt(2.0)
    mps_ghz = MatrixProductState.from_full_state(state_ghz, L=4, max_chi=4)
    s_ghz = mps_ghz.entanglement_entropy(bipartition_site=2)
    # Theoretical entropy: ln(2) = 0.693147
    print(f"GHZ state entanglement entropy: {s_ghz:.6f} (Expected: {np.log(2):.6f})")
    assert np.isclose(s_ghz, np.log(2), atol=1e-5), "GHZ bipartition entropy must equal ln(2)"
    
    # 4. Bond truncation accuracy on random state
    np.random.seed(42)
    rand_psi = np.random.randn(16) + 1.0j * np.random.randn(16)
    rand_psi /= np.linalg.norm(rand_psi)
    mps_rand = MatrixProductState.from_full_state(rand_psi, L=4, max_chi=16)
    rand_recon = mps_rand.to_full_state()
    assert np.allclose(rand_psi, rand_recon, atol=1e-8), "Exact SVD MPS must reproduce state exactly"
    
    print("Matrix Product States verifications PASSED cleanly.")

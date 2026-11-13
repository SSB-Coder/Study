"""
O02: Entropic Regularized Optimal Transport (Sinkhorn-Knopp Algorithm)
Pure NumPy implementation of fast entropic optimal transport:
1. Standard matrix scaling Sinkhorn-Knopp iteration
2. Log-domain stabilized Sinkhorn for low-temperature epsilon
"""

import numpy as np
from typing import Tuple


def sinkhorn_knopp(
    p: np.ndarray, 
    q: np.ndarray, 
    cost_matrix: np.ndarray, 
    epsilon: float = 0.05, 
    max_iter: int = 500, 
    tol: float = 1e-7
) -> Tuple[float, np.ndarray]:
    """
    Solves entropic OT: min_{P in Pi(p, q)} <C, P> - epsilon * H(P)
    via alternating matrix scaling: u = p / (K @ v), v = q / (K.T @ u)
    """
    n, m = cost_matrix.shape
    # Gibbs kernel
    K = np.exp(-cost_matrix / epsilon)

    u = np.ones(n) / n
    v = np.ones(m) / m

    for it in range(max_iter):
        u = p / (K @ v + 1e-16)
        v = q / (K.T @ u + 1e-16)
        if it % 10 == 0:
            curr_P = u[:, None] * K * v[None, :]
            err = np.sum(np.abs(np.sum(curr_P, axis=1) - p))
            if err < tol:
                break

    # Final scaling to guarantee row marginal satisfaction
    u = p / (K @ v + 1e-16)
    P = u[:, None] * K * v[None, :]
    cost = float(np.sum(P * cost_matrix))
    return cost, P


def sinkhorn_log_domain(
    p: np.ndarray, 
    q: np.ndarray, 
    cost_matrix: np.ndarray, 
    epsilon: float = 0.01, 
    max_iter: int = 500, 
    tol: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """
    Log-domain stabilized Sinkhorn iteration using Log-Sum-Exp to prevent underflow.
    alpha_i = -eps * logsumexp((beta_j - C_ij) / eps) + eps * log(p_i)
    """
    n, m = cost_matrix.shape
    log_p = np.log(p + 1e-16)
    log_q = np.log(q + 1e-16)

    alpha = np.zeros(n)
    beta = np.zeros(m)

    def logsumexp(matrix, axis):
        max_val = np.max(matrix, axis=axis, keepdims=True)
        return np.squeeze(max_val + np.log(np.sum(np.exp(matrix - max_val), axis=axis, keepdims=True)), axis=axis)

    for it in range(max_iter):
        # Update alpha
        m_mat = (beta[None, :] - cost_matrix) / epsilon
        alpha = epsilon * (log_p - logsumexp(m_mat, axis=1))

        # Update beta
        m_mat = (alpha[:, None] - cost_matrix) / epsilon
        beta = epsilon * (log_q - logsumexp(m_mat, axis=0))

        if it % 10 == 0:
            log_P = (alpha[:, None] + beta[None, :] - cost_matrix) / epsilon
            curr_P = np.exp(log_P)
            err = np.sum(np.abs(np.sum(curr_P, axis=1) - p))
            if err < tol:
                break

    # Final pass on alpha to ensure row marginals
    m_mat = (beta[None, :] - cost_matrix) / epsilon
    alpha = epsilon * (log_p - logsumexp(m_mat, axis=1))

    # Recover coupling matrix
    log_P = (alpha[:, None] + beta[None, :] - cost_matrix) / epsilon
    P = np.exp(log_P)
    cost = float(np.sum(P * cost_matrix))
    return cost, P


if __name__ == "__main__":
    np.random.seed(42)

    # Spatial 2D distribution transport: 10 source points to 10 target points
    n = 10
    sources = np.random.randn(n, 2)
    targets = np.random.randn(n, 2) + 2.0  # Shifted cluster

    p = np.ones(n) / n
    q = np.ones(n) / n

    # Standard OT practice: scale cost matrix by mean distance
    C = np.sum((sources[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
    C_norm = C / np.mean(C)

    # 1. Standard Sinkhorn
    cost_std, P_std = sinkhorn_knopp(p, q, C_norm, epsilon=0.1)
    print(f"Standard Sinkhorn Cost: {cost_std:.4f}")

    # Mass conservation checks
    assert np.allclose(np.sum(P_std, axis=1), p, atol=1e-3), "Row marginals violated"
    assert np.allclose(np.sum(P_std, axis=0), q, atol=1e-3), "Column marginals violated"

    # 2. Log-Domain Stabilized Sinkhorn with lower temperature epsilon
    cost_log, P_log = sinkhorn_log_domain(p, q, C_norm, epsilon=0.04)
    print(f"Log-Domain Stabilized Sinkhorn Cost (eps=0.04): {cost_log:.4f}")
    assert np.allclose(np.sum(P_log, axis=1), p, atol=1e-3), "Log-domain row marginals violated"
    assert np.allclose(np.sum(P_log, axis=0), q, atol=1e-3), "Log-domain col marginals violated"

    # Lower epsilon produces sharper transport coupling
    print(f"P_std max entry: {np.max(P_std):.4f}, P_log max entry: {np.max(P_log):.4f}")
    assert np.max(P_log) > np.max(P_std), "Lower epsilon must produce sharper, sparser transport coupling"

    print("O02 Sanity Check: PASSED cleanly.")

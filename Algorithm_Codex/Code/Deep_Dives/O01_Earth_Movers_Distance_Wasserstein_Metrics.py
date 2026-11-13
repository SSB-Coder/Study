"""
O01: Earth Mover's Distance & Wasserstein Metrics
Pure NumPy implementation of optimal transport metrics:
1. Exact 1D Wasserstein-1 metric via sorted quantile difference
2. Discrete multi-dimensional Kantorovich optimal transport via dual coordinate ascent
"""

import numpy as np
from typing import Tuple


def wasserstein_1d(u_samples: np.ndarray, v_samples: np.ndarray) -> float:
    """
    Computes exact 1D Wasserstein-1 distance (Earth Mover's Distance).
    W_1(u, v) = integral_0^1 |F_u^(-1)(t) - F_v^(-1)(t)| dt
    """
    u_sorted = np.sort(u_samples)
    v_sorted = np.sort(v_samples)

    if len(u_sorted) == len(v_sorted):
        return float(np.mean(np.abs(u_sorted - v_sorted)))

    # Empirical CDF step interpolation for unequal lengths
    all_vals = np.concatenate([u_sorted, v_sorted])
    all_vals.sort()
    
    # Evaluate empirical CDFs on unique intervals
    deltas = np.diff(all_vals)
    midpoints = all_vals[:-1] + deltas / 2.0

    cdf_u = np.searchsorted(u_sorted, midpoints, side='right') / len(u_sorted)
    cdf_v = np.searchsorted(v_sorted, midpoints, side='right') / len(v_sorted)

    return float(np.sum(np.abs(cdf_u - cdf_v) * deltas))


def discrete_optimal_transport_dual(
    p: np.ndarray, 
    q: np.ndarray, 
    cost_matrix: np.ndarray, 
    max_iter: int = 500, 
    lr: float = 0.05
) -> Tuple[float, np.ndarray]:
    """
    Solves discrete Kantorovich OT problem via dual projected subgradient ascent:
    max_{alpha, beta} alpha^T p + beta^T q  s.t. alpha_i + beta_j <= C_{ij}
    """
    n, m = cost_matrix.shape
    alpha = np.zeros(n)
    beta = np.zeros(m)

    for it in range(max_iter):
        # Coordinate update for beta: beta_j = min_i (C_{ij} - alpha_i)
        beta = np.min(cost_matrix - alpha[:, None], axis=0)
        # Coordinate update for alpha: alpha_i = min_j (C_{ij} - beta_j)
        alpha = np.min(cost_matrix - beta[None, :], axis=1)

    # Recover primal transport coupling via soft relaxation of complementary slackness
    slack = cost_matrix - alpha[:, None] - beta[None, :]
    # Points with zero slack carry positive mass
    transport_plan = np.exp(-slack * 20.0)
    # Project to marginals
    transport_plan = transport_plan * (p[:, None] / np.sum(transport_plan, axis=1, keepdims=True))
    transport_plan = transport_plan * (q[None, :] / np.sum(transport_plan, axis=0, keepdims=True))

    total_cost = float(np.sum(transport_plan * cost_matrix))
    return total_cost, transport_plan


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Test 1D Wasserstein translation property: W_1(X, X + shift) == shift
    shift = 3.5
    X_samples = np.random.randn(200)
    Y_samples = X_samples + shift

    w1_dist = wasserstein_1d(X_samples, Y_samples)
    print(f"1D Wasserstein distance under shift {shift:.2f}: {w1_dist:.4f}")
    assert np.isclose(w1_dist, shift, atol=1e-3), f"Expected {shift}, got {w1_dist}"

    # 2. Test Multi-dimensional Discrete Transport
    # 2 source points moving to 2 target points
    # Source at (0, 0) and (0, 1), target at (2, 0) and (2, 1)
    sources = np.array([[0.0, 0.0], [0.0, 1.0]])
    targets = np.array([[2.0, 0.0], [2.0, 1.0]])
    
    p_weights = np.array([0.5, 0.5])
    q_weights = np.array([0.5, 0.5])

    # Euclidean ground cost
    C = np.linalg.norm(sources[:, None, :] - targets[None, :, :], axis=-1)
    print("Cost Matrix C:\n", C.round(3))

    ot_cost, plan = discrete_optimal_transport_dual(p_weights, q_weights, C, max_iter=300)
    print(f"Optimal Transport Total Work: {ot_cost:.4f}")
    print("Coupling Transport Plan:\n", plan.round(3))

    # Moving both points along x-axis by 2.0 requires total work of 2.0
    assert np.isclose(ot_cost, 2.0, atol=0.1), f"Expected transport cost ~ 2.0, got {ot_cost:.4f}"
    # Mass conservation assertions
    assert np.allclose(np.sum(plan, axis=1), p_weights, atol=1e-2), "Row marginals must match p"
    assert np.allclose(np.sum(plan, axis=0), q_weights, atol=1e-2), "Col marginals must match q"

    print("O01 Sanity Check: PASSED cleanly.")

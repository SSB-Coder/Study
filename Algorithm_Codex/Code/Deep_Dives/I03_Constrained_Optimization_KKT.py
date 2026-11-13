"""
I03. Constrained Optimization & KKT Conditions
From-scratch implementation in pure NumPy.
"""

import numpy as np

class QuadraticProgramKKT:
    """Solves equality and inequality constrained QP: min 0.5 * x^T Q x + c^T x s.t. A_eq x = b_eq, A_ineq x <= b_ineq."""
    @staticmethod
    def solve_equality_kkt(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """Solves min 0.5 x^T Q x + c^T x s.t. A x = b via exact KKT block system <- eq. 3.1"""
        n = len(c)
        m = len(b) if A.ndim > 1 else 1

        # Form KKT Block Matrix: [[Q, A^T], [A, 0]]
        KKT_matrix = np.block([
            [Q, A.T],
            [A, np.zeros((m, m))]
        ])
        rhs = np.concatenate([-c, b])

        # Solve system
        sol = np.linalg.solve(KKT_matrix, rhs)
        x_opt = sol[:n]
        nu_opt = sol[n:]
        return x_opt, nu_opt

    @staticmethod
    def verify_kkt(Q: np.ndarray, c: np.ndarray, A_ineq: np.ndarray, b_ineq: np.ndarray, 
                    x: np.ndarray, lambdas: np.ndarray):
        """Verifies the 4 KKT conditions numerically for min f(x) s.t. A_ineq x <= b_ineq."""
        # 1. Stationarity: Q x + c + A_ineq^T lambda == 0
        stationarity_res = np.linalg.norm(Q @ x + c + A_ineq.T @ lambdas)
        # 2. Primal Feasibility: A_ineq x - b <= 0
        primal_violation = np.max(A_ineq @ x - b_ineq)
        # 3. Dual Feasibility: lambda >= 0
        dual_violation = np.min(lambdas)
        # 4. Complementary Slackness: lambda_i * (A_i x - b_i) == 0
        slackness = np.max(np.abs(lambdas * (A_ineq @ x - b_ineq)))

        return {
            "stationarity_res": float(stationarity_res),
            "primal_feasible": bool(primal_violation <= 1e-6),
            "dual_feasible": bool(dual_violation >= -1e-6),
            "complementary_slackness": float(slackness)
        }

if __name__ == "__main__":
    print("--- I03: Constrained Optimization & KKT Sanity Check ---")
    np.random.seed(42)
    # Minimize f(x1, x2) = x1^2 + x2^2 s.t. x1 + x2 = 2
    # Unconstrained minimum is (0,0); constrained minimum on line is (1, 1)
    Q = 2.0 * np.eye(2)
    c = np.zeros(2)
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([2.0])

    x_star, nu_star = QuadraticProgramKKT.solve_equality_kkt(Q, c, A_eq, b_eq)
    print(f"Optimal solution x*: {x_star}, Equality Multiplier nu*: {nu_star}")
    assert np.allclose(x_star, [1.0, 1.0])
    
    # Inequality KKT check on constraint that prevents moving into the origin:
    # 2 - (x1 + x2) <= 0  <=>  -x1 - x2 <= -2
    A_ineq = np.array([[-1.0, -1.0]])
    b_ineq = np.array([-2.0])
    # Stationarity: Q x + c + A_ineq^T lambda = [2, 2] + lambda * [-1, -1] = 0  => lambda = 2.0 >= 0
    lambdas = np.array([2.0])
    kkt_report = QuadraticProgramKKT.verify_kkt(Q, c, A_ineq, b_ineq, x_star, lambdas)
    print("KKT Verification Report:", kkt_report)
    assert kkt_report["primal_feasible"]
    assert kkt_report["dual_feasible"]
    assert kkt_report["stationarity_res"] < 1e-5
    assert kkt_report["complementary_slackness"] < 1e-5
    print("I03 Sanity Check: PASSED cleanly.")

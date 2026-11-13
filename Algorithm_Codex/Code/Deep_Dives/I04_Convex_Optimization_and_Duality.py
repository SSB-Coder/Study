"""
I04. Convex Optimization & Projected Gradient Descent (PGD)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class ProjectedGradientDescent:
    """Projected Gradient Descent over L2-ball, non-negative orthant, and simplex."""
    @staticmethod
    def project_l2_ball(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm <= radius:
            return v
        return v * (radius / norm)

    @staticmethod
    def project_simplex(v: np.ndarray) -> np.ndarray:
        """Exact O(d log d) projection onto the probability simplex sum(x) = 1, x >= 0 <- eq. 3.2"""
        d = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, d + 1) > (cssv - 1.0))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    @classmethod
    def optimize(cls, grad_fn, x0: np.ndarray, proj_type: str = "simplex", 
                 radius: float = 1.0, lr: float = 0.1, n_iters: int = 100):
        x = np.asarray(x0, dtype=np.float64).copy()
        
        for _ in range(n_iters):
            g = grad_fn(x)
            # 1. Unconstrained gradient step
            x_step = x - lr * g
            
            # 2. Projection onto convex set <- eq. 3.1
            if proj_type == "simplex":
                x = cls.project_simplex(x_step)
            elif proj_type == "l2_ball":
                x = cls.project_l2_ball(x_step, radius=radius)
            elif proj_type == "non_negative":
                x = np.maximum(0.0, x_step)
        return x

if __name__ == "__main__":
    print("--- I04: Projected Gradient Descent Sanity Check ---")
    np.random.seed(42)
    # Minimize f(x) = 0.5 * ||x - c||^2 where target c = [5, -3, 2]
    # Projected onto probability simplex (sum x_i = 1, x_i >= 0)
    c_target = np.array([5.0, -3.0, 2.0])
    def grad_f(x): return x - c_target

    x_init = np.array([0.33, 0.33, 0.34])
    x_opt = ProjectedGradientDescent.optimize(grad_f, x_init, proj_type="simplex", lr=0.1, n_iters=100)

    print(f"Target point (outside simplex): {c_target}")
    print(f"Projected optimal point x*:     {x_opt}")
    print(f"Sum of coordinates:             {np.sum(x_opt):.6f}")
    assert np.isclose(np.sum(x_opt), 1.0)
    assert np.all(x_opt >= 0.0)
    assert x_opt[1] == 0.0 # Negative target coordinate should be projected to exact zero
    print("I04 Sanity Check: PASSED cleanly.")

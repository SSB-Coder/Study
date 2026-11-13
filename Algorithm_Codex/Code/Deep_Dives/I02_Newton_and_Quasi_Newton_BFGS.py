"""
I02. Newton's Method & Quasi-Newton (BFGS)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class BFGSScratch:
    """Quasi-Newton BFGS optimizer with Armijo line search."""
    def __init__(self, max_iter: int = 100, tol: float = 1e-5, c1: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.c1 = c1
        self.trajectory_ = []

    def minimize(self, fn, grad_fn, x0: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        d = len(x)
        # Initial inverse Hessian approximation H_0 = Identity <- eq. 3.1
        H = np.eye(d)

        for iteration in range(self.max_iter):
            self.trajectory_.append(x.copy())
            g = grad_fn(x)
            
            if np.linalg.norm(g) < self.tol:
                break

            # Search direction p_k = -H_k @ g_k <- eq. 3.1
            p = -H @ g

            # Backtracking line search (Armijo condition) <- eq. 3.2
            alpha = 1.0
            f_current = fn(x)
            slope = self.c1 * np.dot(g, p)
            while fn(x + alpha * p) > f_current + alpha * slope:
                alpha *= 0.5
                if alpha < 1e-12:
                    break

            # Step updates
            s = alpha * p # s_k = x_{k+1} - x_k
            x_next = x + s
            g_next = grad_fn(x_next)
            y = g_next - g # y_k = g_{k+1} - g_k

            # Curvature condition: y^T s > 0
            ys = np.dot(y, s)
            if ys > 1e-10:
                rho = 1.0 / ys
                # BFGS Rank-2 update: H = (I - rho * s @ y^T) @ H @ (I - rho * y @ s^T) + rho * s @ s^T <- eq. 3.1
                I = np.eye(d)
                V1 = I - rho * np.outer(s, y)
                V2 = I - rho * np.outer(y, s)
                H = V1 @ H @ V2 + rho * np.outer(s, s)

            x = x_next

        return x

if __name__ == "__main__":
    print("--- I02: BFGS Quasi-Newton Sanity Check ---")
    np.random.seed(42)
    # Test on the classic non-convex Rosenbrock Banana function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    # Global minimum is at (1, 1) where f(1, 1) = 0
    def rosenbrock(v):
        return (1.0 - v[0]) ** 2 + 100.0 * (v[1] - v[0] ** 2) ** 2

    def rosenbrock_grad(v):
        df_dx = -2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0] ** 2)
        df_dy = 200.0 * (v[1] - v[0] ** 2)
        return np.array([df_dx, df_dy])

    x_init = np.array([-1.2, 1.0])
    bfgs = BFGSScratch(max_iter=100, tol=1e-5)
    x_opt = bfgs.minimize(rosenbrock, rosenbrock_grad, x_init)
    
    val_opt = rosenbrock(x_opt)
    print(f"BFGS converged point: {x_opt}, function value: {val_opt:.8f}")
    assert np.allclose(x_opt, [1.0, 1.0], atol=1e-3)
    assert val_opt < 1e-5
    print("I02 Sanity Check: PASSED cleanly.")

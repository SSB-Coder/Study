"""
V05. Root-Finding Methods (Bisection, Newton-Raphson, & Secant)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class RootFinder:
    """
    Implements 1D root-finding algorithms for non-linear equations f(x) = 0.
    """
    
    @staticmethod
    def bisection(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 100):
        """
        Bisection method: unconditionally convergent bracketed search.
        Requires f(a) and f(b) to have opposite signs.
        Linear convergence: error halved every iteration.
        """
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise ValueError(f"Root not bracketed: f({a})={fa}, f({b})={fb} have same sign")
            
        for it in range(max_iter):
            c = (a + b) / 2.0
            fc = f(c)
            if abs(fc) < tol or (b - a) / 2.0 < tol:
                return c, it + 1
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return (a + b) / 2.0, max_iter

    @staticmethod
    def newton_raphson(f, df=None, x0: float = 1.0, tol: float = 1e-12, max_iter: int = 100):
        """
        Newton-Raphson method: quadratic convergence near simple roots.
        x_{k+1} = x_k - f(x_k) / f'(x_k)
        df: analytic derivative function. If None, uses central finite differences.
        """
        x = float(x0)
        h = 1e-7
        
        for it in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                return x, it + 1
            if df is not None:
                fpx = df(x)
            else:
                fpx = (f(x + h) - f(x - h)) / (2.0 * h)
                
            if abs(fpx) < 1e-15:
                raise ZeroDivisionError(f"Vanishing derivative encountered at x = {x}")
                
            x_next = x - fx / fpx
            if abs(x_next - x) < tol:
                return x_next, it + 1
            x = x_next
        return x, max_iter

    @staticmethod
    def secant(f, x0: float, x1: float, tol: float = 1e-12, max_iter: int = 100):
        """
        Secant method: superlinear convergence (golden ratio ~ 1.618) without explicit derivatives.
        x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
        """
        p0, p1 = float(x0), float(x1)
        q0, q1 = f(p0), f(p1)
        
        for it in range(max_iter):
            if abs(q1 - q0) < 1e-15:
                raise ZeroDivisionError("Horizontal secant line encountered")
            p = p1 - q1 * (p1 - p0) / (q1 - q0)
            if abs(f(p)) < tol or abs(p - p1) < tol:
                return p, it + 1
            p0, q0 = p1, q1
            p1, q1 = p, f(p)
        return p1, max_iter

if __name__ == "__main__":
    print("--- V05: Root-Finding Methods Sanity Check ---")
    # Test equation: f(x) = x^3 - 2x - 5 = 0
    # Analytic real root ~ 2.0945514815423265
    f_cubic = lambda x: x**3 - 2.0 * x - 5.0
    df_cubic = lambda x: 3.0 * x**2 - 2.0
    
    # 1. Bisection
    r_bi, it_bi = RootFinder.bisection(f_cubic, 2.0, 3.0)
    print(f"Bisection: root={r_bi:.10f}, iter={it_bi}, residual={abs(f_cubic(r_bi)):.2e}")
    assert abs(f_cubic(r_bi)) < 1e-10
    
    # 2. Newton-Raphson
    r_nr, it_nr = RootFinder.newton_raphson(f_cubic, df_cubic, x0=2.0)
    print(f"Newton-Raphson: root={r_nr:.10f}, iter={it_nr}, residual={abs(f_cubic(r_nr)):.2e}")
    assert abs(f_cubic(r_nr)) < 1e-12
    assert it_nr < 10 # Quadratic convergence is rapid
    
    # 3. Secant
    r_sec, it_sec = RootFinder.secant(f_cubic, 2.0, 3.0)
    print(f"Secant: root={r_sec:.10f}, iter={it_sec}, residual={abs(f_cubic(r_sec)):.2e}")
    assert abs(f_cubic(r_sec)) < 1e-12
    
    # Test transcendental equation: cos(x) - x = 0 (Dottie number ~ 0.7390851332)
    f_dottie = lambda x: np.cos(x) - x
    r_dot, _ = RootFinder.newton_raphson(f_dottie, x0=0.5)
    assert abs(f_dottie(r_dot)) < 1e-12
    print("V05 Sanity Check: PASSED cleanly.")

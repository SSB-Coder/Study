"""
V04. Numerical Integration (Trapezoidal, Simpson's 1/3, & Monte Carlo)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class NumericalIntegrator:
    """
    Implements 1D numerical quadrature and multi-dimensional Monte Carlo integration.
    """
    
    @staticmethod
    def trapezoidal(f, a: float, b: float, n: int = 1000) -> float:
        """
        Composite Trapezoidal Rule: O(h^2) convergence.
        Integral approx = (h/2) * [f(a) + 2 * sum(f(x_i)) + f(b)]
        """
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x], dtype=float)
        h = (b - a) / n
        return float((h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1]))

    @staticmethod
    def simpson(f, a: float, b: float, n: int = 1000) -> float:
        """
        Composite Simpson's 1/3 Rule: O(h^4) convergence.
        Requires n to be even.
        Integral approx = (h/3) * [f(x_0) + 4 sum(f(odd)) + 2 sum(f(even)) + f(x_n)]
        """
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x], dtype=float)
        h = (b - a) / n
        return float((h / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1]))

    @staticmethod
    def monte_carlo(f, bounds: list, n_samples: int = 100000, seed: int = 42):
        """
        Monte Carlo Integration over a hyper-rectangle: O(1 / sqrt(N)) convergence.
        bounds: list of (low, high) tuples for each dimension.
        Returns estimated integral and standard error.
        """
        rng = np.random.RandomState(seed)
        dim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        volume = float(np.prod(highs - lows))
        
        # Sample uniform points
        pts = rng.uniform(lows, highs, size=(n_samples, dim))
        if dim == 1:
            vals = np.array([f(pt[0]) for pt in pts])
        else:
            vals = np.array([f(pt) for pt in pts])
            
        mean_val = float(np.mean(vals))
        std_err = float(volume * np.std(vals, ddof=1) / np.sqrt(n_samples))
        estimate = volume * mean_val
        return estimate, std_err

if __name__ == "__main__":
    print("--- V04: Numerical Integration Sanity Check ---")
    # Test function 1: f(x) = x^2 on [0, 1] -> Exact = 1/3
    f1 = lambda x: x**2
    exact1 = 1.0 / 3.0
    trap1 = NumericalIntegrator.trapezoidal(f1, 0.0, 1.0, n=200)
    simp1 = NumericalIntegrator.simpson(f1, 0.0, 1.0, n=200)
    print(f"f(x) = x^2 on [0, 1]: Exact={exact1:.8f}, Trap={trap1:.8f}, Simp={simp1:.8f}")
    assert abs(trap1 - exact1) < 1e-4
    assert abs(simp1 - exact1) < 1e-10 # Simpson is exact for polynomials up to degree 3!
    
    # Test function 2: f(x) = sin(x) on [0, pi] -> Exact = 2.0
    f2 = lambda x: np.sin(x)
    exact2 = 2.0
    trap2 = NumericalIntegrator.trapezoidal(f2, 0.0, np.pi, n=500)
    simp2 = NumericalIntegrator.simpson(f2, 0.0, np.pi, n=500)
    assert abs(simp2 - exact2) < 1e-8
    
    # Test Monte Carlo on unit circle area: f(x, y) = 1 if x^2+y^2 <= 1 else 0 on [-1, 1]^2 -> Exact = pi
    f_circle = lambda pt: 1.0 if (pt[0]**2 + pt[1]**2) <= 1.0 else 0.0
    mc_est, mc_err = NumericalIntegrator.monte_carlo(f_circle, [(-1.0, 1.0), (-1.0, 1.0)], n_samples=200000)
    print(f"Monte Carlo Circle Area (Pi): Estimate={mc_est:.4f} +/- {mc_err:.4f}, True={np.pi:.4f}")
    assert abs(mc_est - np.pi) < 3.0 * mc_err
    print("V04 Sanity Check: PASSED cleanly.")

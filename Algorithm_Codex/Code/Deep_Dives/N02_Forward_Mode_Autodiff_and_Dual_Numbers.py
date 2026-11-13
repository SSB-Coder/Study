"""
N02: Forward-Mode Automatic Differentiation & Dual Numbers
Pure NumPy implementation of forward-mode autodiff via dual algebra (epsilon^2 = 0).
Computes exact Jacobian-Vector Products (JVPs) and machine-precision derivatives in a single forward pass.
"""

import numpy as np
from typing import Union, Callable, List


class Dual:
    """
    Dual number representation: z = real + dual * epsilon, where epsilon^2 = 0.
    f(x + epsilon) = f(x) + f'(x) * epsilon.
    """
    def __init__(self, real: float, dual: float = 0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __repr__(self) -> str:
        return f"Dual(real={self.real:.6f}, dual={self.dual:.6f})"

    def __add__(self, other: Union["Dual", float, int]) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return Dual(self.real + other, self.dual)

    def __radd__(self, other: Union[float, int]) -> "Dual":
        return self.__add__(other)

    def __sub__(self, other: Union["Dual", float, int]) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return Dual(self.real - other, self.dual)

    def __rsub__(self, other: Union[float, int]) -> "Dual":
        return Dual(other - self.real, -self.dual)

    def __mul__(self, other: Union["Dual", float, int]) -> "Dual":
        if isinstance(other, Dual):
            # (u + u' eps)(v + v' eps) = uv + (u v' + u' v) eps
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        return Dual(self.real * other, self.dual * other)

    def __rmul__(self, other: Union[float, int]) -> "Dual":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Dual", float, int]) -> "Dual":
        if isinstance(other, Dual):
            # (u + u' eps) / (v + v' eps) = (u / v) + ((u' v - u v') / v^2) eps
            denom = other.real ** 2
            return Dual(self.real / other.real, (self.dual * other.real - self.real * other.dual) / denom)
        return Dual(self.real / other, self.dual / other)

    def __rtruediv__(self, other: Union[float, int]) -> "Dual":
        denom = self.real ** 2
        return Dual(other / self.real, (-other * self.dual) / denom)

    def __pow__(self, power: Union[float, int]) -> "Dual":
        # d/dx (u^p) = p * u^(p-1) * u'
        return Dual(self.real ** power, power * (self.real ** (power - 1)) * self.dual)

    def __neg__(self) -> "Dual":
        return Dual(-self.real, -self.dual)

    # Transcendental functions
    def sin(self) -> "Dual":
        return Dual(np.sin(self.real), self.dual * np.cos(self.real))

    def cos(self) -> "Dual":
        return Dual(np.cos(self.real), -self.dual * np.sin(self.real))

    def exp(self) -> "Dual":
        e = np.exp(self.real)
        return Dual(e, self.dual * e)

    def log(self) -> "Dual":
        return Dual(np.log(self.real), self.dual / self.real)


def diff(f: Callable[[Dual], Dual], x: float) -> float:
    """Computes exact derivative f'(x) using forward-mode dual arithmetic."""
    d_in = Dual(x, dual=1.0)
    d_out = f(d_in)
    return d_out.dual


def jvp(f: Callable[[List[Dual]], List[Dual]], x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes Jacobian-Vector Product: J_f(x) @ v
    in a single forward pass.
    """
    dual_inputs = [Dual(x[i], dual=v[i]) for i in range(len(x))]
    dual_outputs = f(dual_inputs)
    return np.array([out.dual for out in dual_outputs])


def compute_jacobian(f: Callable[[List[Dual]], List[Dual]], x: np.ndarray) -> np.ndarray:
    """Computes full Jacobian matrix column-by-column using forward mode."""
    n = len(x)
    m = len(f([Dual(xi) for xi in x]))
    J = np.zeros((m, n))

    for col in range(n):
        v = np.zeros(n)
        v[col] = 1.0  # Basis vector
        J[:, col] = jvp(f, x, v)

    return J


if __name__ == "__main__":
    # 1. Scalar Function Differentiation
    # f(x) = x^3 * sin(x) + exp(2*x)
    # f'(x) = 3*x^2*sin(x) + x^3*cos(x) + 2*exp(2*x)
    def test_func(x: Dual) -> Dual:
        return (x ** 3) * x.sin() + (x * 2.0).exp()

    x_val = 1.5
    computed_derivative = diff(test_func, x_val)
    analytical_derivative = (3.0 * (x_val ** 2) * np.sin(x_val) + 
                             (x_val ** 3) * np.cos(x_val) + 
                             2.0 * np.exp(2.0 * x_val))

    print(f"Dual Forward Derivative: {computed_derivative:.10f}")
    print(f"Analytical Derivative:   {analytical_derivative:.10f}")
    assert np.isclose(computed_derivative, analytical_derivative, atol=1e-12), "Dual derivative mismatch"

    # 2. Multi-variable Vector Mapping Jacobian Evaluation
    # f: R^2 -> R^2
    # f1(x1, x2) = x1^2 + x1 * x2
    # f2(x1, x2) = sin(x1) * exp(x2)
    #
    # Jacobian:
    # [ 2*x1 + x2,    x1 ]
    # [ cos(x1)*e^x2, sin(x1)*e^x2 ]
    def vector_field(inputs: List[Dual]) -> List[Dual]:
        x1, x2 = inputs[0], inputs[1]
        y1 = (x1 ** 2) + x1 * x2
        y2 = x1.sin() * x2.exp()
        return [y1, y2]

    x_pt = np.array([0.8, 1.2])
    J_dual = compute_jacobian(vector_field, x_pt)
    
    J_analytical = np.array([
        [2.0 * x_pt[0] + x_pt[1], x_pt[0]],
        [np.cos(x_pt[0]) * np.exp(x_pt[1]), np.sin(x_pt[0]) * np.exp(x_pt[1])]
    ])

    print("Computed Forward Jacobian:\n", J_dual)
    print("Analytical Jacobian:\n", J_analytical)
    assert np.allclose(J_dual, J_analytical, atol=1e-12), "Jacobian mismatch"

    # 3. Test JVP in arbitrary directional vector v
    v_dir = np.array([1.5, -2.0])
    jvp_result = jvp(vector_field, x_pt, v_dir)
    expected_jvp = J_analytical @ v_dir
    print(f"JVP Result:     {jvp_result}")
    print(f"Expected JVP:   {expected_jvp}")
    assert np.allclose(jvp_result, expected_jvp, atol=1e-12), "JVP mismatch"

    print("N02 Sanity Check: PASSED cleanly.")

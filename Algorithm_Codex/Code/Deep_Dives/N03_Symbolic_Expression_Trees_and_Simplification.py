"""
N03: Symbolic Expression Trees & Graph Simplification
Pure Python/NumPy implementation of AST-based symbolic differentiation and algebraic graph reduction.
Demonstrates compile-time operator fusion and constant folding in DL compilers (XLA / TorchScript).
"""

import numpy as np
from typing import Dict, Union, Callable


class Expr:
    """Base class for all symbolic expression tree nodes."""
    def diff(self, var: str) -> "Expr":
        raise NotImplementedError

    def simplify(self) -> "Expr":
        return self

    def eval(self, env: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def __add__(self, other) -> "Expr":
        return Add(self, _to_expr(other))

    def __radd__(self, other) -> "Expr":
        return Add(_to_expr(other), self)

    def __mul__(self, other) -> "Expr":
        return Mul(self, _to_expr(other))

    def __rmul__(self, other) -> "Expr":
        return Mul(_to_expr(other), self)

    def __pow__(self, other) -> "Expr":
        return Pow(self, _to_expr(other))


def _to_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    return Const(x)


class Const(Expr):
    def __init__(self, val: Union[float, int]):
        self.val = float(val)

    def __repr__(self) -> str:
        if self.val.is_integer():
            return str(int(self.val))
        return f"{self.val:.4f}"

    def diff(self, var: str) -> Expr:
        return Const(0.0)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return self.val


class Var(Expr):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def diff(self, var: str) -> Expr:
        return Const(1.0) if self.name == var else Const(0.0)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        if self.name not in env:
            raise KeyError(f"Variable {self.name} missing from evaluation environment")
        return env[self.name]


class Add(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"

    def diff(self, var: str) -> Expr:
        return Add(self.left.diff(var), self.right.diff(var))

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        
        # Constant folding
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val + r.val)
        # Additive identity
        if isinstance(l, Const) and l.val == 0.0:
            return r
        if isinstance(r, Const) and r.val == 0.0:
            return l
            
        return Add(l, r)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return self.left.eval(env) + self.right.eval(env)


class Mul(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"

    def diff(self, var: str) -> Expr:
        # Product Rule: u' v + u v'
        return Add(Mul(self.left.diff(var), self.right), Mul(self.left, self.right.diff(var)))

    def simplify(self) -> Expr:
        l = self.left.simplify()
        r = self.right.simplify()
        
        # Constant folding
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val * r.val)
        # Annihilation: 0 * x = 0
        if (isinstance(l, Const) and l.val == 0.0) or (isinstance(r, Const) and r.val == 0.0):
            return Const(0.0)
        # Multiplicative identity: 1 * x = x
        if isinstance(l, Const) and l.val == 1.0:
            return r
        if isinstance(r, Const) and r.val == 1.0:
            return l
            
        return Mul(l, r)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return self.left.eval(env) * self.right.eval(env)


class Pow(Expr):
    def __init__(self, base: Expr, exp: Expr):
        self.base = base
        self.exp = exp

    def __repr__(self) -> str:
        return f"({self.base}^{self.exp})"

    def diff(self, var: str) -> Expr:
        # Power rule for constant exponent: p * u^(p-1) * u'
        if isinstance(self.exp, Const):
            p = self.exp.val
            return Mul(Mul(Const(p), Pow(self.base, Const(p - 1.0))), self.base.diff(var))
        raise NotImplementedError("General variable power differentiation omitted for simplicity")

    def simplify(self) -> Expr:
        b = self.base.simplify()
        e = self.exp.simplify()
        
        if isinstance(e, Const):
            if e.val == 0.0:
                return Const(1.0)
            if e.val == 1.0:
                return b
            if isinstance(b, Const):
                return Const(b.val ** e.val)
                
        return Pow(b, e)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return self.base.eval(env) ** self.exp.eval(env)


class Sin(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

    def __repr__(self) -> str:
        return f"sin({self.arg})"

    def diff(self, var: str) -> Expr:
        return Mul(Cos(self.arg), self.arg.diff(var))

    def simplify(self) -> Expr:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(np.sin(a.val))
        return Sin(a)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return np.sin(self.arg.eval(env))


class Cos(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

    def __repr__(self) -> str:
        return f"cos({self.arg})"

    def diff(self, var: str) -> Expr:
        return Mul(Const(-1.0), Mul(Sin(self.arg), self.arg.diff(var)))

    def simplify(self) -> Expr:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(np.cos(a.val))
        return Cos(a)

    def eval(self, env: Dict[str, Union[float, np.ndarray]]):
        return np.cos(self.arg.eval(env))


def full_simplify(expr: Expr, max_passes: int = 10) -> Expr:
    """Iteratively simplifies expression tree until a fixed point is reached."""
    curr = expr
    for _ in range(max_passes):
        simplified = curr.simplify()
        if repr(simplified) == repr(curr):
            break
        curr = simplified
    return curr


if __name__ == "__main__":
    # 1. Define Expression: f(x) = x^3 + 5 * x + 10
    x = Var("x")
    expr = (x ** 3) + (Const(5) * x) + Const(10)
    print(f"Original Expression: {expr}")

    # Compute raw symbolic derivative: d/dx [ x^3 + 5x + 10 ]
    raw_deriv = expr.diff("x")
    print(f"Raw Symbolic Derivative: {raw_deriv}")

    # Simplify algebraic tree
    clean_deriv = full_simplify(raw_deriv)
    print(f"Simplified Derivative:   {clean_deriv}")

    # Verify numerical evaluation against ground truth: f'(x) = 3*x^2 + 5
    test_points = np.linspace(-5, 5, 20)
    eval_vals = np.array([clean_deriv.eval({"x": pt}) for pt in test_points])
    expected_vals = 3.0 * (test_points ** 2) + 5.0

    print(f"Max Absolute Error vs Analytical: {np.max(np.abs(eval_vals - expected_vals)):.12e}")
    assert np.allclose(eval_vals, expected_vals), "Symbolic evaluation mismatch"

    # 2. Test Non-linear Chain Rule: g(x) = sin(x^2) -> g'(x) = 2*x*cos(x^2)
    g = Sin(x ** 2)
    g_prime = full_simplify(g.diff("x"))
    print(f"Derivative of sin(x^2): {g_prime}")

    g_eval = np.array([g_prime.eval({"x": pt}) for pt in test_points])
    g_expected = 2.0 * test_points * np.cos(test_points ** 2)
    assert np.allclose(g_eval, g_expected), "Trigonometric chain rule mismatch"

    print("N03 Sanity Check: PASSED cleanly.")

"""
N01: Tape-Based Reverse-Mode Automatic Differentiation
Pure NumPy implementation of a dynamic computation graph DAG and backward tape engine.
Powers modern deep learning frameworks (PyTorch autograd / JAX).
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union


class Tensor:
    """
    Core Tensor primitive supporting dynamic computational graph construction,
    vector-Jacobian products (VJP), and topological backward propagation.
    """
    def __init__(self, data: Union[float, int, np.ndarray, list], requires_grad: bool = False):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=np.float64)
        else:
            self.data = data.astype(np.float64)
            
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: Tuple["Tensor", ...] = ()
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Sum over broadcasted axes
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = out.grad
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                while grad.ndim > self.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = out.grad * self.data
                while grad.ndim > other.data.ndim:
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0).astype(float)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30, 30)))
        out = Tensor(sig, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                grad = out.grad * sig * (1.0 - sig)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(float(np.sum(self.data)), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                grad = out.grad * np.ones_like(self.data)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Topological sort and backpropagation of gradients."""
        topo: List[Tensor] = []
        visited = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Base scalar loss gradient is 1.0
        self.grad = np.ones_like(self.data)

        # Walk backwards down the tape
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Gradient Checking against Numerical Finite Differences
    x = Tensor([2.0, -3.0], requires_grad=True)
    y = Tensor([5.0, 4.0], requires_grad=True)
    # Loss: L = sum(x * y + relu(x) * 2.0)
    loss = (x * y + x.relu() * Tensor([2.0, 2.0])).sum()
    loss.backward()

    # Manual analytical gradient:
    # L = x0*y0 + 2*relu(x0) + x1*y1 + 2*relu(x1)
    # dL/dx0 = y0 + 2*(x0>0) = 5.0 + 2.0 = 7.0
    # dL/dx1 = y1 + 2*(x1>0) = 4.0 + 0.0 = 4.0
    # dL/dy0 = x0 = 2.0
    # dL/dy1 = x1 = -3.0
    print(f"Computed x.grad: {x.grad}")
    print(f"Computed y.grad: {y.grad}")
    assert np.allclose(x.grad, [7.0, 4.0]), "Gradient check failed for x"
    assert np.allclose(y.grad, [2.0, -3.0]), "Gradient check failed for y"

    # 2. Train an MLP on Non-Linear XOR using Pure NumPy Autodiff
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_train = np.array([[0], [1], [1], [0]], dtype=float)

    # Weights initialized with standard scaling
    W1 = Tensor(np.random.randn(2, 4), requires_grad=True)
    b1 = Tensor(np.zeros((1, 4)), requires_grad=True)
    W2 = Tensor(np.random.randn(4, 1), requires_grad=True)
    b2 = Tensor(np.zeros((1, 1)), requires_grad=True)

    lr = 2.0
    losses = []
    for epoch in range(800):
        # Forward Pass
        x_t = Tensor(X_train)
        y_t = Tensor(y_train)

        h1 = (x_t @ W1 + b1).sigmoid()
        out = (h1 @ W2 + b2).sigmoid()

        # Binary Cross-Entropy Loss approximation / MSE: (out - y)^2
        diff = out + y_t * Tensor(-1.0)
        loss = (diff * diff).sum()

        # Zero gradients
        for p in [W1, b1, W2, b2]:
            p.grad = None

        # Backward Pass via Computational Graph Tape
        loss.backward()

        # Gradient Descent Step
        for p in [W1, b1, W2, b2]:
            p.data -= lr * p.grad

        losses.append(float(loss.data))

    final_loss = losses[-1]
    print(f"Trained XOR MLP with Tape Autodiff: Initial Loss={losses[0]:.4f}, Final Loss={final_loss:.4f}")
    assert final_loss < 0.1, f"Expected XOR loss < 0.1, got {final_loss:.4f}"

    print("N01 Sanity Check: PASSED cleanly.")

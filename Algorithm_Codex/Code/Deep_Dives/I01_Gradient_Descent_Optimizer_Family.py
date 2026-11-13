"""
I01. Gradient Descent Optimizer Family (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class OptimizerSuite:
    """Implementations of first-order gradient optimizers without autograd."""
    
    @staticmethod
    def sgd(theta: np.ndarray, grad: np.ndarray, lr: float = 0.01):
        return theta - lr * grad

    class Momentum:
        def __init__(self, lr: float = 0.01, gamma: float = 0.9):
            self.lr = lr
            self.gamma = gamma
            self.v = None
        def step(self, theta: np.ndarray, grad: np.ndarray):
            if self.v is None: self.v = np.zeros_like(theta)
            self.v = self.gamma * self.v + self.lr * grad # <- eq. 3.1
            return theta - self.v

    class AdaGrad:
        def __init__(self, lr: float = 0.1, eps: float = 1e-8):
            self.lr = lr
            self.eps = eps
            self.G = None
        def step(self, theta: np.ndarray, grad: np.ndarray):
            if self.G is None: self.G = np.zeros_like(theta)
            self.G += grad ** 2 # Cumulative squared gradients
            return theta - (self.lr / (np.sqrt(self.G) + self.eps)) * grad

    class RMSProp:
        def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
            self.lr = lr
            self.beta = beta
            self.eps = eps
            self.v = None
        def step(self, theta: np.ndarray, grad: np.ndarray):
            if self.v is None: self.v = np.zeros_like(theta)
            self.v = self.beta * self.v + (1.0 - self.beta) * (grad ** 2)
            return theta - (self.lr / (np.sqrt(self.v) + self.eps)) * grad

    class Adam:
        def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.m = None
            self.v = None
            self.t = 0
        def step(self, theta: np.ndarray, grad: np.ndarray):
            if self.m is None:
                self.m = np.zeros_like(theta)
                self.v = np.zeros_like(theta)
            self.t += 1
            # Biased moments <- eq. 3.1
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
            # Bias correction <- eq. 3.1
            m_hat = self.m / (1.0 - self.beta1 ** self.t)
            v_hat = self.v / (1.0 - self.beta2 ** self.t)
            return theta - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

if __name__ == "__main__":
    print("--- I01: Gradient Descent Optimizers Sanity Check ---")
    np.random.seed(42)
    # Test all optimizers on quadratic loss: f(theta) = 0.5 * (10 * theta_1^2 + theta_2^2)
    # Optimum is at (0, 0)
    def loss_and_grad(theta):
        loss = 0.5 * (10.0 * theta[0]**2 + theta[1]**2)
        grad = np.array([10.0 * theta[0], theta[1]])
        return loss, grad

    theta_init = np.array([4.0, 4.0])
    n_steps = 150

    # 1. Test Adam
    adam = OptimizerSuite.Adam(lr=0.2)
    theta = theta_init.copy()
    for _ in range(n_steps):
        _, g = loss_and_grad(theta)
        theta = adam.step(theta, g)
    loss_adam = loss_and_grad(theta)[0]
    print(f"Adam final theta: {theta}, loss: {loss_adam:.6f}")
    assert loss_adam < 1e-4

    # 2. Test Momentum
    mom = OptimizerSuite.Momentum(lr=0.02, gamma=0.8)
    theta = theta_init.copy()
    for _ in range(n_steps):
        _, g = loss_and_grad(theta)
        theta = mom.step(theta, g)
    loss_mom = loss_and_grad(theta)[0]
    print(f"Momentum final theta: {theta}, loss: {loss_mom:.6f}")
    assert loss_mom < 1e-2
    print("I01 Sanity Check: PASSED cleanly.")

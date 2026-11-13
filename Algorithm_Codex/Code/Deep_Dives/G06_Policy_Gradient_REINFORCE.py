"""
G06: Policy Gradient Methods (REINFORCE with Baseline)
From-scratch NumPy implementation of REINFORCE on a goal-reaching task.
"""

import numpy as np

class SoftmaxPolicy:
    """Parametric linear softmax policy for discrete actions: pi(a | s) = softmax(theta @ s)."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.W = np.zeros((action_dim, state_dim), dtype=np.float64)

    def probabilities(self, s: np.ndarray) -> np.ndarray:
        logits = self.W @ s
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def select_action(self, s: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.probabilities(s)
        return int(rng.choice(self.action_dim, p=probs))

    def compute_grad_log_prob(self, s: np.ndarray, a: int) -> np.ndarray:
        """
        Gradient of ln pi(a | s) w.r.t W:
        d/dW_k ln pi(a | s) = s * (I(a == k) - pi(k | s))
        Returns matrix of shape (action_dim, state_dim).
        """
        probs = self.probabilities(s)
        grad = np.zeros_like(self.W)
        for k in range(self.action_dim):
            indicator = 1.0 if k == a else 0.0
            grad[k] = s * (indicator - probs[k])
        return grad

    def update(self, grads: list[np.ndarray], advantages: list[float]):
        total_grad = np.zeros_like(self.W)
        for grad, adv in zip(grads, advantages):
            total_grad += grad * adv
        self.W += self.lr * total_grad


class BaselineValueNetwork:
    """Linear baseline state value function V(s) = w^T s."""
    def __init__(self, state_dim: int, lr: float = 0.05):
        self.w = np.zeros(state_dim, dtype=np.float64)
        self.lr = lr

    def predict(self, s: np.ndarray) -> float:
        return float(np.dot(self.w, s))

    def update(self, s: np.ndarray, target: float):
        pred = self.predict(s)
        err = target - pred
        self.w += self.lr * err * s


if __name__ == "__main__":
    print("--- G06: REINFORCE Policy Gradient Sanity Check ---")
    rng = np.random.default_rng(42)
    policy = SoftmaxPolicy(state_dim=2, action_dim=2, lr=0.08)
    baseline = BaselineValueNetwork(state_dim=2, lr=0.08)

    # Synthetic environment:
    # State s = [direction, 1.0]. If direction > 0, optimal action is 1 (reward +1).
    # If direction < 0, optimal action is 0 (reward +1). Suboptimal action gets reward -1.
    for ep in range(300):
        direction = rng.choice([-1.0, 1.0])
        s = np.array([direction, 1.0])
        a = policy.select_action(s, rng)
        
        # Reward
        r = 1.0 if (direction > 0 and a == 1) or (direction < 0 and a == 0) else -1.0
        
        # Advantage
        v_pred = baseline.predict(s)
        adv = r - v_pred
        baseline.update(s, r)

        # Policy update
        grad = policy.compute_grad_log_prob(s, a)
        policy.update([grad], [adv])

    # Evaluate learned policy probabilities
    s_pos = np.array([1.0, 1.0])
    s_neg = np.array([-1.0, 1.0])
    p_pos = policy.probabilities(s_pos)
    p_neg = policy.probabilities(s_neg)

    print(f"P(Action 1 | State direction=+1): {p_pos[1]:.2%}")
    print(f"P(Action 0 | State direction=-1): {p_neg[0]:.2%}")

    assert p_pos[1] > 0.85, "Policy should learn to pick Action 1 for positive state"
    assert p_neg[0] > 0.85, "Policy should learn to pick Action 0 for negative state"
    print("G06 Sanity Check: PASSED cleanly.")

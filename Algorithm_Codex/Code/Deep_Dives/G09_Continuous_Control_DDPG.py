"""
G09: Continuous Control with Deep Deterministic Policy Gradient (DDPG)
Pure NumPy implementation of continuous actor-critic update step.
"""

import numpy as np

class ContinuousDDPGAgent:
    """
    Continuous actor-critic with soft target updates.
    Actor: a = clip(theta_actor * s, -1.0, 1.0)
    Critic: Q(s, a) = - 0.5 * (a - w_sa * s)^2 + w_bias
    For exact tracking: grad_a Q = - (a - w_sa * s)
    """
    def __init__(self, state_dim: int = 1, lr_actor: float = 0.1, lr_critic: float = 0.1, gamma: float = 0.9, tau: float = 0.05):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Actor: parameter scalar/vector mapping state to continuous action
        self.theta = np.zeros(state_dim, dtype=np.float64)
        self.theta_target = np.zeros(state_dim, dtype=np.float64)

        # Critic: bilinear weights
        self.w_sa = 0.5
        self.w_bias = 0.0
        self.w_sa_target = 0.5
        self.w_bias_target = 0.0

    def get_action(self, s: np.ndarray, noise: float = 0.0) -> float:
        a = float(np.dot(self.theta, s) + noise)
        return float(np.clip(a, -1.0, 1.0))

    def critic_predict(self, s: np.ndarray, a: float, use_target: bool = False) -> float:
        w_sa = self.w_sa_target if use_target else self.w_sa
        w_b = self.w_bias_target if use_target else self.w_bias
        s_val = float(s[0]) if hasattr(s, '__len__') else float(s)
        return float(- 0.5 * (a - w_sa * s_val) ** 2 + w_b)

    def update_step(self, s: np.ndarray, a: float, r: float, next_s: np.ndarray, done: bool):
        s_val = float(s[0]) if hasattr(s, '__len__') else float(s)
        next_s_val = float(next_s[0]) if hasattr(next_s, '__len__') else float(next_s)

        # 1. Target Q
        next_a = float(np.clip(np.dot(self.theta_target, next_s), -1.0, 1.0))
        target_q = r if done else r + self.gamma * self.critic_predict(next_s, next_a, use_target=True)

        # 2. Update Critic
        pred_q = self.critic_predict(s, a, use_target=False)
        td_err = target_q - pred_q
        # grad w.r.t w_sa = (a - w_sa * s) * s
        grad_w_sa = (a - self.w_sa * s_val) * s_val
        self.w_sa += self.lr_critic * td_err * grad_w_sa
        self.w_bias += self.lr_critic * td_err

        # 3. Update Actor: grad_theta Q(s, mu(s)) = grad_a Q(s, a) * grad_theta mu(s)
        current_a = float(np.clip(np.dot(self.theta, s), -1.0, 1.0))
        # grad_a Q = - (current_a - w_sa * s)
        grad_a_q = - (current_a - self.w_sa * s_val)
        grad_theta_mu = s
        self.theta += self.lr_actor * grad_a_q * grad_theta_mu

        # 4. Soft target updates
        self.theta_target = self.tau * self.theta + (1.0 - self.tau) * self.theta_target
        self.w_sa_target = self.tau * self.w_sa + (1.0 - self.tau) * self.w_sa_target
        self.w_bias_target = self.tau * self.w_bias + (1.0 - self.tau) * self.w_bias_target


if __name__ == "__main__":
    print("--- G09: Continuous Control (DDPG) Sanity Check ---")
    rng = np.random.default_rng(42)
    agent = ContinuousDDPGAgent(state_dim=1, lr_actor=0.15, lr_critic=0.15, gamma=0.0, tau=0.1)

    # 1D continuous target task:
    # State is desired continuous velocity target s in [-1.0, 1.0]
    # Reward is negative squared tracking error: r = - 0.5 * (a - s)^2
    for _ in range(600):
        s = rng.uniform(-1.0, 1.0, size=1)
        noise = float(rng.normal(0, 0.2))
        a = agent.get_action(s, noise=noise)
        r = float(- 0.5 * (a - s[0]) ** 2)
        agent.update_step(s, a, r, next_s=s, done=True)

    # Evaluate actor tracking accuracy
    test_states = [0.8, -0.6, 0.0]
    errors = []
    for ts in test_states:
        s_arr = np.array([ts])
        action = agent.get_action(s_arr, noise=0.0)
        err = abs(action - ts)
        errors.append(err)
        print(f"Target State {ts:+.2f} -> Continuous Action {action:+.2f} (Error: {err:.4f})")

    mean_err = np.mean(errors)
    assert mean_err < 0.25, f"Actor should learn continuous target tracking with low error, got {mean_err}"
    print("G09 Sanity Check: PASSED cleanly.")

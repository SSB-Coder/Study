"""
G07: Advantage Actor-Critic (A2C) & Generalized Advantage Estimation (GAE)
From-scratch NumPy implementation of online A2C with GAE support.
"""

import numpy as np

class ActorCriticAgent:
    """Advantage Actor-Critic (A2C) with linear function approximation."""
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 0.05, lr_critic: float = 0.1, gamma: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # Actor: W_actor (action_dim, state_dim)
        self.W_actor = np.zeros((action_dim, state_dim), dtype=np.float64)
        # Critic: w_critic (state_dim,)
        self.w_critic = np.zeros(state_dim, dtype=np.float64)

    def get_action_probs(self, s: np.ndarray) -> np.ndarray:
        logits = self.W_actor @ s
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def select_action(self, s: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.get_action_probs(s)
        return int(rng.choice(self.action_dim, p=probs))

    def get_value(self, s: np.ndarray) -> float:
        return float(np.dot(self.w_critic, s))

    def update_step(self, s: np.ndarray, a: int, r: float, next_s: np.ndarray, done: bool):
        """1-step online Advantage Actor-Critic update."""
        v_s = self.get_value(s)
        v_next = 0.0 if done else self.get_value(next_s)
        
        # TD error (Advantage)
        td_error = r + self.gamma * v_next - v_s

        # Critic update: grad_w V(s) = s
        self.w_critic += self.lr_critic * td_error * s

        # Actor update: grad_theta ln pi(a | s)
        probs = self.get_action_probs(s)
        grad_actor = np.zeros_like(self.W_actor)
        for k in range(self.action_dim):
            indicator = 1.0 if k == a else 0.0
            grad_actor[k] = s * (indicator - probs[k])

        self.W_actor += self.lr_actor * td_error * grad_actor


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """
    Computes Generalized Advantage Estimation (GAE) vector across an episode trajectory.
    values has length T + 1 (with terminal value at index T).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


if __name__ == "__main__":
    print("--- G07: Actor-Critic (A2C) Sanity Check ---")
    rng = np.random.default_rng(42)
    agent = ActorCriticAgent(state_dim=2, action_dim=2, lr_actor=0.08, lr_critic=0.1, gamma=0.9)

    # Train on 2-state decision task:
    # State s=[1, 0] -> optimal action is 0 (reward +1)
    # State s=[0, 1] -> optimal action is 1 (reward +1)
    for _ in range(400):
        state_idx = rng.integers(0, 2)
        s = np.zeros(2)
        s[state_idx] = 1.0
        
        a = agent.select_action(s, rng)
        r = 1.0 if a == state_idx else -1.0
        next_s = np.zeros(2)  # Terminal transition
        agent.update_step(s, a, r, next_s, done=True)

    probs_0 = agent.get_action_probs(np.array([1.0, 0.0]))
    probs_1 = agent.get_action_probs(np.array([0.0, 1.0]))

    print(f"Action probabilities for State 0: {probs_0}")
    print(f"Action probabilities for State 1: {probs_1}")

    assert probs_0[0] > 0.85, "State 0 should strongly prefer Action 0"
    assert probs_1[1] > 0.85, "State 1 should strongly prefer Action 1"

    # Test GAE computation
    rewards_test = np.array([1.0, 0.0, 2.0])
    values_test = np.array([0.5, 0.8, 1.2, 0.0])
    gae = compute_gae(rewards_test, values_test, gamma=0.9, lam=0.9)
    assert len(gae) == 3
    print(f"Computed GAE Advantages: {gae}")
    print("G07 Sanity Check: PASSED cleanly.")

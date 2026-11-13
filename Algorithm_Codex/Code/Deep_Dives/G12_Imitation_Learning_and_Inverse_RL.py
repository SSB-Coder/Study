"""
G12: Imitation Learning & Maximum Entropy Inverse RL (MaxEnt IRL)
Pure NumPy implementation of feature expectation matching.
"""

import numpy as np

class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning for linear reward R(s) = theta^T phi(s)."""
    def __init__(self, n_states: int, feature_dim: int, lr: float = 0.1):
        self.n_states = n_states
        self.feature_dim = feature_dim
        self.lr = lr
        self.theta = np.zeros(feature_dim, dtype=np.float64)

    def get_rewards(self, features: np.ndarray) -> np.ndarray:
        # features: (n_states, feature_dim) -> returns (n_states,)
        return features @ self.theta

    def compute_state_visitation_freq(self, P_transition: np.ndarray, initial_dist: np.ndarray, horizon: int = 10) -> np.ndarray:
        """Computes occupancy measure D(s) across finite horizon."""
        d = np.copy(initial_dist)
        d_accum = np.copy(d)
        for _ in range(horizon - 1):
            d = d @ P_transition
            d_accum += d
        return d_accum / horizon

    def update(self, expert_feature_expectations: np.ndarray, learner_features: np.ndarray):
        """
        Gradient update: grad = mu_expert - mu_learner
        theta <- theta + lr * grad
        """
        grad = expert_feature_expectations - learner_features
        self.theta += self.lr * grad


if __name__ == "__main__":
    print("--- G12: MaxEnt IRL Sanity Check ---")
    # 3-state chain: 0 -> 1 -> 2 (State 2 is highly desirable in expert eyes)
    # Features: one-hot representations
    features = np.eye(3)
    
    # Expert visits State 2 heavily: feature expectation [0.1, 0.2, 0.7]
    expert_mu = np.array([0.1, 0.2, 0.7])
    
    irl = MaxEntIRL(n_states=3, feature_dim=3, lr=0.2)

    # Train for 50 iterations
    for _ in range(50):
        # Current inferred rewards
        rewards = irl.get_rewards(features)
        # Softmax visitation induced by reward
        learner_mu = np.exp(rewards - np.max(rewards))
        learner_mu /= np.sum(learner_mu)
        
        irl.update(expert_mu, learner_mu)

    final_rewards = irl.get_rewards(features)
    print(f"Recovered State Rewards: State 0={final_rewards[0]:.3f}, State 1={final_rewards[1]:.3f}, State 2={final_rewards[2]:.3f}")

    assert final_rewards[2] > final_rewards[1] > final_rewards[0], "Recovered reward must rank State 2 highest"
    print("G12 Sanity Check: PASSED cleanly.")

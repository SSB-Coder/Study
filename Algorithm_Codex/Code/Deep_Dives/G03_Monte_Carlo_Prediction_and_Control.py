"""
G03: Monte Carlo Prediction and Control
From-scratch NumPy implementation of First-Visit MC on a simulated GridWorld.
"""

import numpy as np

class MonteCarloControl:
    """On-Policy First-Visit Monte Carlo Control with epsilon-greedy exploration."""
    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.9, epsilon: float = 0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
        self.returns_count = np.zeros((n_states, n_actions), dtype=np.int64)

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        best_a = np.argmax(self.Q[state])
        return int(best_a)

    def update_episode(self, episode: list[tuple[int, int, float]]):
        """
        episode is a list of (state, action, reward) tuples from t=0 to T-1.
        Uses backwards iteration to compute discounted returns in O(T).
        """
        visited = set()
        G = 0.0
        # Iterate backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Check if this is the first visit to (state, action)
            sa_pair = (state, action)
            first_visit = True
            for prev_t in range(t):
                if (episode[prev_t][0], episode[prev_t][1]) == sa_pair:
                    first_visit = False
                    break
            
            if first_visit:
                self.returns_sum[state, action] += G
                self.returns_count[state, action] += 1
                self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]


def run_synthetic_grid_episode(agent: MonteCarloControl, rng: np.random.Generator, max_steps: int = 30):
    """
    Simulates a 1D corridor with 4 states: 0, 1, 2, 3 (goal).
    Actions: 0 = Left, 1 = Right.
    """
    state = 0
    episode = []
    for _ in range(max_steps):
        action = agent.select_action(state, rng)
        if action == 1:
            next_state = min(state + 1, 3)
        else:
            next_state = max(state - 1, 0)

        if next_state == 3:
            reward = 10.0
            episode.append((state, action, reward))
            break
        else:
            reward = -1.0
            episode.append((state, action, reward))
            state = next_state
    return episode


if __name__ == "__main__":
    print("--- G03: Monte Carlo Control Sanity Check ---")
    rng = np.random.default_rng(42)
    agent = MonteCarloControl(n_states=4, n_actions=2, gamma=0.95, epsilon=0.15)

    # Train for 500 episodes
    for ep in range(500):
        episode = run_synthetic_grid_episode(agent, rng)
        agent.update_episode(episode)

    print("Learned Q-values for corridor states:")
    for s in range(3):
        print(f"State {s}: Left={agent.Q[s, 0]:.2f}, Right={agent.Q[s, 1]:.2f}")
        assert agent.Q[s, 1] > agent.Q[s, 0], f"State {s} should prefer moving right towards goal"

    print("G03 Sanity Check: PASSED cleanly.")

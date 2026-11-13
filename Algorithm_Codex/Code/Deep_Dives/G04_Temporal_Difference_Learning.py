"""
G04: Temporal-Difference Learning (SARSA and Q-Learning)
Pure NumPy implementation demonstrating the classic Cliff Walking comparison.
"""

import numpy as np

class CliffWalkingEnvironment:
    """
    4x12 Grid:
    (0,0) is top-left, (3,0) is START, (3,11) is GOAL.
    Cliff: (3, 1) through (3, 10). Falling off gives reward -100 and resets to (3,0).
    Normal step reward is -1.
    """
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.n_states = 48
        self.n_actions = 4  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.start = (3, 0)
        self.goal = (3, 11)
        self.reset()

    def reset(self) -> int:
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def step(self, action: int) -> tuple[int, float, bool]:
        r, c = self.agent_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        nr, nc = np.clip(r + dr, 0, self.rows - 1), np.clip(c + dc, 0, self.cols - 1)
        next_pos = (int(nr), int(nc))

        # Check cliff
        if next_pos[0] == 3 and 1 <= next_pos[1] <= 10:
            self.agent_pos = self.start
            return self._pos_to_state(self.agent_pos), -100.0, False
        
        self.agent_pos = next_pos
        if next_pos == self.goal:
            return self._pos_to_state(self.agent_pos), -1.0, True
        return self._pos_to_state(self.agent_pos), -1.0, False


class QLearningAgent:
    """Off-Policy TD Control (Q-Learning)."""
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, next_s: int, done: bool):
        target = r if done else r + self.gamma * np.max(self.Q[next_s])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])


class SarsaAgent:
    """On-Policy TD Control (SARSA)."""
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, next_s: int, next_a: int, done: bool):
        target = r if done else r + self.gamma * self.Q[next_s, next_a]
        self.Q[s, a] += self.lr * (target - self.Q[s, a])


if __name__ == "__main__":
    print("--- G04: TD Learning (SARSA vs Q-Learning) Sanity Check ---")
    rng = np.random.default_rng(42)
    env = CliffWalkingEnvironment()
    q_agent = QLearningAgent(env.n_states, env.n_actions, lr=0.2, epsilon=0.1)

    # Train Q-Learning for 400 episodes
    q_rewards = []
    for _ in range(400):
        s = env.reset()
        total_r = 0.0
        for _ in range(200):
            a = q_agent.select_action(s, rng)
            next_s, r, done = env.step(a)
            q_agent.update(s, a, r, next_s, done)
            total_r += r
            s = next_s
            if done:
                break
        q_rewards.append(total_r)

    # Sanity check: Agent should reach the goal efficiently
    # Greedy test path length: optimal direct edge route is 13 steps (-13 total reward)
    s = env.reset()
    test_reward = 0.0
    for _ in range(30):
        a = int(np.argmax(q_agent.Q[s]))
        next_s, r, done = env.step(a)
        test_reward += r
        s = next_s
        if done:
            break

    print(f"Q-learning greedy test episode reward: {test_reward}")
    assert test_reward >= -20.0, f"Q-learning greedy path should reach goal in under 20 steps, got {test_reward}"
    print("G04 Sanity Check: PASSED cleanly.")

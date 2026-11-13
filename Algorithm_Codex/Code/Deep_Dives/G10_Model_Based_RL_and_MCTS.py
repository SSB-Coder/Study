"""
G10: Model-Based Reinforcement Learning (Dyna-Q)
Pure NumPy implementation showing planning dramatically accelerates learning.
"""

import numpy as np

class DynaQAgent:
    """Dyna-Q agent integrating real experience with internal model planning."""
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.2, gamma: float = 0.95, epsilon: float = 0.2, n_planning: int = 10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning

        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        # Model stores: model[s][a] = (r, next_s)
        self.model = {}
        self.observed_states = set()

    def select_action(self, s: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        max_q = np.max(self.Q[s])
        candidates = np.where(self.Q[s] == max_q)[0]
        return int(rng.choice(candidates))

    def update(self, s: int, a: int, r: float, next_s: int, rng: np.random.Generator):
        # 1. Real Q-learning update
        target = r + self.gamma * np.max(self.Q[next_s])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

        # 2. Update model
        if s not in self.model:
            self.model[s] = {}
        self.model[s][a] = (r, next_s)
        self.observed_states.add(s)

        # 3. Mental planning (hallucinated experiences)
        if self.n_planning > 0 and len(self.observed_states) > 0:
            obs_list = list(self.observed_states)
            for _ in range(self.n_planning):
                sim_s = int(rng.choice(obs_list))
                sim_actions = list(self.model[sim_s].keys())
                sim_a = int(rng.choice(sim_actions))
                sim_r, sim_next_s = self.model[sim_s][sim_a]
                
                sim_target = sim_r + self.gamma * np.max(self.Q[sim_next_s])
                self.Q[sim_s, sim_a] += self.lr * (sim_target - self.Q[sim_s, sim_a])


if __name__ == "__main__":
    print("--- G10: Model-Based RL (Dyna-Q) Sanity Check ---")
    # Linear chain: 0 -> 1 -> 2 -> 3 (Goal, reward +10)
    # Compare plain Q-learning (planning=0) vs Dyna-Q (planning=25) on Episode 2
    q_ep2 = []
    dyna_ep2 = []
    for seed in range(10):
        # Plain Q-learning
        rng = np.random.default_rng(seed)
        q = DynaQAgent(n_states=4, n_actions=2, lr=0.5, gamma=0.9, epsilon=0.05, n_planning=0)
        for ep in range(2):
            s = 0; steps = 0
            while s < 3 and steps < 100:
                a = q.select_action(s, rng); ns = min(s + 1, 3) if a == 1 else max(s - 1, 0)
                q.update(s, a, 10.0 if ns == 3 else 0.0, ns, rng); s = ns; steps += 1
            if ep == 1:
                q_ep2.append(steps)

        # Dyna-Q
        rng = np.random.default_rng(seed)
        dyna = DynaQAgent(n_states=4, n_actions=2, lr=0.5, gamma=0.9, epsilon=0.05, n_planning=25)
        for ep in range(2):
            s = 0; steps = 0
            while s < 3 and steps < 100:
                a = dyna.select_action(s, rng); ns = min(s + 1, 3) if a == 1 else max(s - 1, 0)
                dyna.update(s, a, 10.0 if ns == 3 else 0.0, ns, rng); s = ns; steps += 1
            if ep == 1:
                dyna_ep2.append(steps)

    mean_q = float(np.mean(q_ep2))
    mean_dyna = float(np.mean(dyna_ep2))
    print(f"Episode 2 Mean Steps to Goal -> Standard Q-Learning: {mean_q:.1f}, Dyna-Q (Model Planning): {mean_dyna:.1f}")
    assert mean_dyna < mean_q, "Dyna-Q planning must solve Episode 2 significantly faster than standard Q-learning!"
    assert mean_dyna <= 4.0, f"Dyna-Q should solve 3-step chain near optimally on Ep 2, got {mean_dyna}"
    print("G10 Sanity Check: PASSED cleanly.")

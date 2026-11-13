"""
G13: Offline Reinforcement Learning (Conservative Q-Learning - CQL)
Pure NumPy implementation showing suppression of Out-of-Distribution (OOD) actions.
"""

import numpy as np

class ConservativeQLearning:
    """Tabular CQL agent demonstrating penalization of unobserved actions."""
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, alpha_cql: float = 1.0, gamma: float = 0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.alpha_cql = alpha_cql
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def update_offline(self, dataset: list[tuple[int, int, float, int]]):
        """
        dataset is a collection of static transitions (s, a, r, next_s).
        CQL update:
        Bellman loss + alpha * [ sum_a exp(Q(s, a))/Z * grad_Q - grad_Q(s, a_data) ]
        """
        for s, a, r, next_s in dataset:
            target = r + self.gamma * np.max(self.Q[next_s])
            td_err = target - self.Q[s, a]

            # CQL regularizer: push down all actions by their softmax probability,
            # push up observed dataset action
            exp_q = np.exp(self.Q[s] - np.max(self.Q[s]))
            probs = exp_q / np.sum(exp_q)

            # Standard TD update on (s, a)
            self.Q[s, a] += self.lr * td_err

            # Apply CQL penalty across all actions
            for act in range(self.n_actions):
                if act == a:
                    self.Q[s, act] += self.lr * self.alpha_cql * (1.0 - probs[act])
                else:
                    self.Q[s, act] -= self.lr * self.alpha_cql * probs[act]


if __name__ == "__main__":
    print("--- G13: Offline RL (CQL) Sanity Check ---")
    # Dataset only contains demonstrations of Action 0 in State 0 with reward +2.0
    # Action 1 was never observed in the dataset (OOD action)
    dataset = [(0, 0, 2.0, 0) for _ in range(50)]

    cql = ConservativeQLearning(n_states=1, n_actions=2, lr=0.1, alpha_cql=0.5, gamma=0.5)
    cql.update_offline(dataset)

    print(f"Learned Q-values for State 0: Action 0 (In-Dataset) = {cql.Q[0, 0]:.4f}, Action 1 (OOD) = {cql.Q[0, 1]:.4f}")
    assert cql.Q[0, 0] > cql.Q[0, 1], "CQL must prefer observed action over OOD action"
    assert cql.Q[0, 1] < 0.0, "CQL should actively push unobserved action value into the negative penalty regime"
    print("G13 Sanity Check: PASSED cleanly.")

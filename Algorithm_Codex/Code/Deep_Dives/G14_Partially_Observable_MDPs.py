"""
G14: Partially Observable MDPs (POMDP) & Bayesian Belief State Updates
Pure NumPy implementation of exact Bayesian belief tracking on the classic Tiger Problem.
"""

import numpy as np

class TigerPOMDP:
    """
    Classic Tiger Problem (Kaelbling, Littman, Cassandra 1998):
    Hidden States: 0 = Tiger Left, 1 = Tiger Right.
    Actions: 0 = Listen (cost -1), 1 = Open Left, 2 = Open Right.
    Observations: 0 = Hear tiger on Left, 1 = Hear tiger on Right.
    Listening is noisy: hears correctly with probability 0.85.
    Opening door gives +10 (treasure) or -100 (eaten by tiger), and resets state randomly.
    """
    def __init__(self):
        self.n_states = 2
        self.n_actions = 3
        self.n_obs = 2
        
        # State transitions P(s' | s, a)
        # Action 0 (Listen) leaves state unchanged
        # Actions 1 & 2 reset state uniformly
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.T[:, 0, :] = np.eye(2)
        self.T[:, 1, :] = 0.5
        self.T[:, 2, :] = 0.5

        # Observations O(o | s', a)
        # For action 0 (Listen):
        self.O = np.zeros((self.n_obs, self.n_states, self.n_actions))
        self.O[0, 0, 0] = 0.85  # Hear left | Tiger left
        self.O[1, 0, 0] = 0.15  # Hear right | Tiger left
        self.O[0, 1, 0] = 0.15  # Hear left | Tiger right
        self.O[1, 1, 0] = 0.85  # Hear right | Tiger right
        # Opening doors gives uninformative uniform observation
        self.O[:, :, 1] = 0.5
        self.O[:, :, 2] = 0.5

    def update_belief(self, b: np.ndarray, action: int, observation: int) -> np.ndarray:
        """
        Exact Bayesian update:
        b'(s') proportional to O(o | s', a) * sum_s T(s' | s, a) * b(s)
        """
        b_pred = np.zeros(self.n_states)
        for s_next in range(self.n_states):
            b_pred[s_next] = np.sum(self.T[:, action, s_next] * b)

        numerator = self.O[observation, :, action] * b_pred
        denom = np.sum(numerator)
        if denom == 0:
            return np.ones(self.n_states) / self.n_states
        return numerator / denom


if __name__ == "__main__":
    print("--- G14: POMDP Belief State Sanity Check ---")
    pomdp = TigerPOMDP()

    # Start with uniform ignorance: 50% left, 50% right
    b = np.array([0.5, 0.5])
    print(f"Initial Prior Belief: Tiger Left={b[0]:.2%}, Tiger Right={b[1]:.2%}")

    # Agent listens (action 0) and hears tiger on the LEFT (obs 0)
    b = pomdp.update_belief(b, action=0, observation=0)
    print(f"After 1st 'Hear Left' observation: Tiger Left={b[0]:.2%}, Tiger Right={b[1]:.2%}")
    assert b[0] > 0.80, "Belief should strongly shift towards Left"

    # Agent listens again and hears tiger on the LEFT again
    b = pomdp.update_belief(b, action=0, observation=0)
    print(f"After 2nd 'Hear Left' observation: Tiger Left={b[0]:.2%}, Tiger Right={b[1]:.2%}")
    assert b[0] > 0.95, "Belief should exceed 95% certainty after 2 consecutive corroborating observations"

    # Agent opens RIGHT door (treasure!), state resets uniformly
    b = pomdp.update_belief(b, action=2, observation=0)
    print(f"After opening door (reset): Tiger Left={b[0]:.2%}, Tiger Right={b[1]:.2%}")
    assert abs(b[0] - 0.5) < 1e-4, "Belief must reset to 50/50 uniform ignorance after opening door"

    print("G14 Sanity Check: PASSED cleanly.")

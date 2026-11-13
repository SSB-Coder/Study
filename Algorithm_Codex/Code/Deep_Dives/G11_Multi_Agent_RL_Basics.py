"""
G11: Multi-Agent Reinforcement Learning (Minimax-Q)
Pure NumPy implementation of Minimax-Q for 2-player zero-sum Matrix Game (Matching Pennies).
"""

import numpy as np

class MinimaxQMatchingPennies:
    """
    Minimax-Q for 2-player zero-sum Matching Pennies.
    Actions: 0 = Heads, 1 = Tails.
    Payoff matrix for Player 1:
          Heads (0)   Tails (1)
    Heads:  +1.0       -1.0
    Tails:  -1.0       +1.0
    Nash Equilibrium: Mixed strategy [0.5, 0.5] with game value V* = 0.0.
    """
    def __init__(self, lr: float = 0.05, gamma: float = 0.0):
        # State space is trivial (1 state)
        self.Q = np.zeros((2, 2), dtype=np.float64)  # Q[a1, a2]
        self.pi = np.array([0.5, 0.5])
        self.V = 0.0
        self.lr = lr
        self.gamma = gamma

    def solve_minimax(self):
        """
        Solves 2x2 zero-sum matrix game for optimal mixed strategy pi = [p, 1-p]:
        Expected payoff against a2=0: p * Q[0,0] + (1-p) * Q[1,0]
        Expected payoff against a2=1: p * Q[0,1] + (1-p) * Q[1,1]
        At equilibrium, p* satisfies equality (or boundary in [0, 1]).
        """
        # Linear equations: p*(Q00 - Q10) + Q10 = p*(Q01 - Q11) + Q11
        # p * ((Q00 - Q10) - (Q01 - Q11)) = Q11 - Q10
        denom = (self.Q[0, 0] - self.Q[1, 0]) - (self.Q[0, 1] - self.Q[1, 1])
        if abs(denom) > 1e-6:
            p = (self.Q[1, 1] - self.Q[1, 0]) / denom
            p = np.clip(p, 0.0, 1.0)
        else:
            p = 0.5

        self.pi = np.array([p, 1.0 - p])
        # V = min_a2 (pi @ Q[:, a2])
        val_against_0 = p * self.Q[0, 0] + (1.0 - p) * self.Q[1, 0]
        val_against_1 = p * self.Q[0, 1] + (1.0 - p) * self.Q[1, 1]
        self.V = min(val_against_0, val_against_1)

    def update(self, a1: int, a2: int, r: float):
        target = r + self.gamma * self.V
        self.Q[a1, a2] += self.lr * (target - self.Q[a1, a2])
        self.solve_minimax()


if __name__ == "__main__":
    print("--- G11: Multi-Agent Minimax-Q Sanity Check ---")
    rng = np.random.default_rng(42)
    agent = MinimaxQMatchingPennies(lr=0.02)

    # True payoff table
    payoffs = np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])

    # Simulate 2000 rounds of self-play / opponent play
    for _ in range(2000):
        # Player 1 samples from learned policy with slight exploration
        p_act = agent.pi[0]
        a1 = 0 if rng.random() < p_act else 1
        # Opponent plays random or adaptive
        a2 = int(rng.integers(0, 2))
        r = payoffs[a1, a2]
        agent.update(a1, a2, r)

    print(f"Learned Minimax-Q Matrix:\n{agent.Q}")
    print(f"Learned Nash Equilibrium Strategy: P(Heads) = {agent.pi[0]:.4f}, P(Tails) = {agent.pi[1]:.4f}")
    print(f"Game Value: {agent.V:.4f} (Expected: ~0.0)")

    # Assert convergence towards mixed strategy equilibrium [0.5, 0.5]
    assert 0.35 <= agent.pi[0] <= 0.65, "Minimax-Q should converge near 50/50 Nash mixed strategy"
    print("G11 Sanity Check: PASSED cleanly.")

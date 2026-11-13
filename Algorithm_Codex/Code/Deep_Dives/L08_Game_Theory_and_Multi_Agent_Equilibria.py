"""
L08: Game Theory & Multi-Agent Equilibria
Pure NumPy implementation finding Pure and Mixed Nash Equilibria in normal-form games.
"""

import numpy as np

class NormalFormGame2P:
    """2-Player Normal-Form Game solver."""
    def __init__(self, u1: np.ndarray, u2: np.ndarray):
        # u1, u2 are matrices of shape (rows, cols) representing payoffs
        self.u1 = np.array(u1, dtype=np.float64)
        self.u2 = np.array(u2, dtype=np.float64)
        self.n_r, self.n_c = self.u1.shape

    def find_pure_nash_equilibria(self) -> list[tuple[int, int]]:
        """Identifies cells (r, c) that are mutual best responses."""
        # Player 1 best response to each column of Player 2
        p1_best = np.zeros_like(self.u1, dtype=bool)
        for c in range(self.n_c):
            max_val = np.max(self.u1[:, c])
            p1_best[:, c] = (self.u1[:, c] == max_val)

        # Player 2 best response to each row of Player 1
        p2_best = np.zeros_like(self.u2, dtype=bool)
        for r in range(self.n_r):
            max_val = np.max(self.u2[r, :])
            p2_best[r, :] = (self.u2[r, :] == max_val)

        # Mutual best response
        equilibria = []
        for r in range(self.n_r):
            for c in range(self.n_c):
                if p1_best[r, c] and p2_best[r, c]:
                    equilibria.append((r, c))
        return equilibria

    def solve_2x2_mixed_nash(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes mixed strategy Nash Equilibrium for 2x2 game:
        Player 1 plays p = [p, 1-p], Player 2 plays q = [q, 1-q].
        """
        assert self.n_r == 2 and self.n_c == 2
        # Player 2 makes Player 1 indifferent:
        # q * u1[0,0] + (1-q) * u1[0,1] = q * u1[1,0] + (1-q) * u1[1,1]
        denom_q = (self.u1[0, 0] - self.u1[1, 0]) - (self.u1[0, 1] - self.u1[1, 1])
        q = (self.u1[1, 1] - self.u1[0, 1]) / denom_q if abs(denom_q) > 1e-6 else 0.5
        q = np.clip(q, 0.0, 1.0)

        # Player 1 makes Player 2 indifferent:
        denom_p = (self.u2[0, 0] - self.u2[0, 1]) - (self.u2[1, 0] - self.u2[1, 1])
        p = (self.u2[1, 1] - self.u2[1, 0]) / denom_p if abs(denom_p) > 1e-6 else 0.5
        p = np.clip(p, 0.0, 1.0)

        return np.array([p, 1.0 - p]), np.array([q, 1.0 - q])


if __name__ == "__main__":
    print("--- L08: Game Theory Nash Equilibrium Sanity Check ---")
    # 1. Prisoner's Dilemma:
    # Cooperate (0), Defect (1)
    # Defect is strictly dominant -> Pure Nash Equilibrium is (Defect, Defect) = (1, 1)
    u1_pd = [
        [ 3,  0],
        [ 5,  1]
    ]
    u2_pd = [
        [ 3,  5],
        [ 0,  1]
    ]
    pd_game = NormalFormGame2P(u1_pd, u2_pd)
    pne_pd = pd_game.find_pure_nash_equilibria()
    print(f"Prisoner's Dilemma Pure Nash Equilibria: {pne_pd}")
    assert pne_pd == [(1, 1)], "Prisoner's Dilemma must have unique PNE at (Defect, Defect)"

    # 2. Matching Pennies (Zero-Sum Game with NO pure equilibria, only 50/50 mixed)
    u1_mp = [
        [ 1, -1],
        [-1,  1]
    ]
    u2_mp = [
        [-1,  1],
        [ 1, -1]
    ]
    mp_game = NormalFormGame2P(u1_mp, u2_mp)
    pne_mp = mp_game.find_pure_nash_equilibria()
    print(f"Matching Pennies Pure Equilibria: {pne_mp} (Expected: empty)")
    assert len(pne_mp) == 0

    p_mix, q_mix = mp_game.solve_2x2_mixed_nash()
    print(f"Matching Pennies Mixed Equilibrium: P1={p_mix.round(2)}, P2={q_mix.round(2)}")
    assert np.allclose(p_mix, [0.5, 0.5]) and np.allclose(q_mix, [0.5, 0.5])
    print("L08 Sanity Check: PASSED cleanly.")

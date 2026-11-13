"""
G02: Markov Decision Processes & Dynamic Programming
Pure NumPy implementation of Value Iteration and Policy Iteration on a GridWorld MDP.
"""

import numpy as np

class GridWorldMDP:
    """
    GridWorld environment with stochastic transitions.
    States: (r, c) represented as flat integers 0 .. (n_rows * n_cols - 1).
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
    """
    def __init__(self, rows: int = 4, cols: int = 4, gamma: float = 0.95):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4
        self.gamma = gamma
        self.goal_state = self.n_states - 1  # Bottom-right corner
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
        
        # Build transition probabilities P[s, a, s'] and expected rewards R[s, a]
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        self.R = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._build_model()

    def _coord(self, s: int):
        return divmod(s, self.cols)

    def _state(self, r: int, c: int) -> int:
        return r * self.cols + c

    def _build_model(self):
        for s in range(self.n_states):
            if s == self.goal_state:
                # Terminal absorbing state
                for a in range(self.n_actions):
                    self.P[s, a, s] = 1.0
                    self.R[s, a] = 0.0
                continue

            r, c = self._coord(s)
            for a, (dr, dc) in enumerate(self.actions):
                nr, nc = r + dr, c + dc
                # Check boundary collision
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    next_s = self._state(nr, nc)
                else:
                    next_s = s  # Bounce back

                self.P[s, a, next_s] = 1.0
                # Reward: +10 on reaching goal, -1 per step cost
                if next_s == self.goal_state:
                    self.R[s, a] = 10.0
                else:
                    self.R[s, a] = -1.0


class DynamicProgrammingSolver:
    """Solves an explicit tabular MDP via Value Iteration and Policy Iteration."""
    def __init__(self, mdp: GridWorldMDP):
        self.mdp = mdp

    def value_iteration(self, theta: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes V* using the Bellman Optimality Operator contraction.
        Returns (V, optimal_policy).
        """
        V = np.zeros(self.mdp.n_states, dtype=np.float64)
        
        for _ in range(max_iter):
            delta = 0.0
            # Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] * V[s']
            Q = self.mdp.R + self.mdp.gamma * np.einsum('ijk,k->ij', self.mdp.P, V)
            new_V = np.max(Q, axis=1)
            delta = np.max(np.abs(new_V - V))
            V = new_V
            if delta < theta:
                break

        # Extract greedy policy
        Q = self.mdp.R + self.mdp.gamma * np.einsum('ijk,k->ij', self.mdp.P, V)
        policy = np.argmax(Q, axis=1)
        return V, policy

    def policy_iteration(self, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves MDP via alternating Policy Evaluation and Policy Improvement.
        """
        policy = np.zeros(self.mdp.n_states, dtype=np.int64)
        
        for _ in range(max_iter):
            # 1. Policy Evaluation (direct matrix inversion)
            # P_pi: (n_states, n_states), R_pi: (n_states,)
            P_pi = np.array([self.mdp.P[s, policy[s]] for s in range(self.mdp.n_states)])
            R_pi = np.array([self.mdp.R[s, policy[s]] for s in range(self.mdp.n_states)])
            
            # V = (I - gamma * P_pi)^(-1) * R_pi
            I = np.eye(self.mdp.n_states)
            V = np.linalg.solve(I - self.mdp.gamma * P_pi, R_pi)

            # 2. Policy Improvement
            Q = self.mdp.R + self.mdp.gamma * np.einsum('ijk,k->ij', self.mdp.P, V)
            new_policy = np.argmax(Q, axis=1)

            if np.array_equal(new_policy, policy):
                break
            policy = new_policy

        return V, policy


if __name__ == "__main__":
    print("--- G02: MDP & Dynamic Programming Sanity Check ---")
    mdp = GridWorldMDP(rows=3, cols=3, gamma=0.9)
    solver = DynamicProgrammingSolver(mdp)

    V_vi, pi_vi = solver.value_iteration()
    V_pi, pi_pi = solver.policy_iteration()

    print(f"Goal state value: {V_vi[mdp.goal_state]:.2f}")
    print(f"Start state (0,0) Value Iteration: {V_vi[0]:.4f} | Policy Iteration: {V_pi[0]:.4f}")
    
    # Assert equivalence of VI and PI
    np.testing.assert_allclose(V_vi, V_pi, atol=1e-4)
    np.testing.assert_array_equal(pi_vi[:-1], pi_pi[:-1])  # Exclude terminal state
    print("Value Iteration and Policy Iteration yielded identical optimal solutions.")
    print("G02 Sanity Check: PASSED cleanly.")

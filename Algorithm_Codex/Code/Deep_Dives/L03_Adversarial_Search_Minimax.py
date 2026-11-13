"""
L03: Adversarial Search (Minimax with Alpha-Beta Pruning)
Pure NumPy/Python implementation on Tic-Tac-Toe demonstrating exact pruning.
"""

import numpy as np

class TicTacToeGame:
    """Tic-Tac-Toe game state representation. 1 = X (MAX), -1 = O (MIN), 0 = empty."""
    def __init__(self, board=None):
        self.board = np.zeros(9, dtype=int) if board is None else np.array(board, dtype=int)

    def get_actions(self) -> list[int]:
        return list(np.where(self.board == 0)[0])

    def make_move(self, action: int, player: int) -> 'TicTacToeGame':
        new_board = self.board.copy()
        new_board[action] = player
        return TicTacToeGame(new_board)

    def check_winner(self) -> int:
        """Returns +1 if X won, -1 if O won, 0 for draw or non-terminal."""
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Cols
            (0, 4, 8), (2, 4, 6)              # Diagonals
        ]
        for a, b, c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3: return 1
            if s == -3: return -1
        return 0

    def is_terminal(self) -> bool:
        return self.check_winner() != 0 or len(self.get_actions()) == 0


class AdversarialSearchSolver:
    """Minimax and Alpha-Beta solvers with node expansion tracking."""
    def __init__(self):
        self.nodes_visited = 0

    def alphabeta(self, state: TicTacToeGame, depth: int, alpha: float, beta: float, is_max: bool) -> int:
        self.nodes_visited += 1
        winner = state.check_winner()
        if winner != 0 or state.is_terminal():
            return winner

        if is_max:
            max_eval = -100
            for action in state.get_actions():
                child = state.make_move(action, player=1)
                ev = self.alphabeta(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = 100
            for action in state.get_actions():
                child = state.make_move(action, player=-1)
                ev = self.alphabeta(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval


if __name__ == "__main__":
    print("--- L03: Minimax with Alpha-Beta Pruning Sanity Check ---")
    # Mid-game board state:
    # X . O
    # . X .
    # O . .
    # X has central advantage; optimal play should yield 1 (Win for X)
    mid_board = [
        1,  0, -1,
        0,  1,  0,
       -1,  0,  0
    ]
    game = TicTacToeGame(mid_board)
    solver = AdversarialSearchSolver()

    val = solver.alphabeta(game, depth=6, alpha=-float('inf'), beta=float('inf'), is_max=True)
    nodes = solver.nodes_visited

    print(f"Evaluated Game Value: {val} (1 = X wins under optimal play)")
    print(f"Nodes Visited with Alpha-Beta Pruning: {nodes}")

    assert val == 1, "X should hold a winning advantage from this double diagonal position"
    assert nodes < 150, f"Alpha-Beta pruning should prune heavily (visited {nodes} nodes)"
    print("L03 Sanity Check: PASSED cleanly.")

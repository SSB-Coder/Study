"""
U04. State-Space Control & the Linear-Quadratic Regulator (LQR)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LinearQuadraticRegulator:
    """
    Discrete-Time Linear-Quadratic Regulator (LQR).
    System dynamics: x_{k+1} = A x_k + B u_k
    Cost: J = sum (x_k^T Q x_k + u_k^T R u_k)
    Optimal feedback control: u_k = -K x_k
    where K = (R + B^T P B)^{-1} B^T P A
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.K, self.P = self.solve_dare()

    def solve_dare(self, max_iters: int = 500, tol: float = 1e-9):
        """
        Solves Discrete Algebraic Riccati Equation (DARE) via fixed-point iteration <- eq. 3.1:
        P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A
        """
        P = self.Q.copy()
        
        for it in range(max_iters):
            # Compute term: B^T P B + R
            BPB_R = self.B.T @ P @ self.B + self.R
            # Invert
            BPB_R_inv = np.linalg.inv(BPB_R)
            
            # Riccati update
            P_next = self.Q + self.A.T @ P @ self.A - self.A.T @ P @ self.B @ BPB_R_inv @ self.B.T @ P @ self.A
            
            if np.linalg.norm(P_next - P) < tol:
                P = P_next
                break
            P = P_next
            
        # Compute optimal gain K <- eq. 3.2
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K, P

    def compute_control(self, x: np.ndarray) -> np.ndarray:
        return -self.K @ x

    def simulate(self, x0: np.ndarray, n_steps: int = 50):
        """Simulates closed-loop response from initial state x0."""
        states = [np.array(x0, dtype=float)]
        controls = []
        x = np.array(x0, dtype=float)
        
        for _ in range(n_steps):
            u = self.compute_control(x)
            x_next = self.A @ x + self.B @ u
            controls.append(u)
            states.append(x_next)
            x = x_next
            
        return np.array(states), np.array(controls)

if __name__ == "__main__":
    print("--- U04: State-Space Control & LQR Sanity Check ---")
    # Inverted pendulum / cart-pole linearized discrete dynamics (dt = 0.05)
    # State: [position, velocity, angle, angular_velocity]
    A = np.array([
        [1.0, 0.05, 0.0,   0.0],
        [0.0, 1.0,  0.05,  0.0],
        [0.0, 0.0,  1.05,  0.05],
        [0.0, 0.0,  1.0,   1.0]
    ])
    B = np.array([
        [0.0],
        [0.05],
        [0.0],
        [0.1]
    ])
    
    Q = np.diag([1.0, 0.1, 10.0, 1.0]) # Penalize angle heavily
    R = np.array([[0.1]])              # Cheap actuator effort
    
    lqr = LinearQuadraticRegulator(A, B, Q, R)
    print(f"Optimal LQR Gain K: {lqr.K}")
    
    # Simulate recovery from initial angle tilt: x0 = [0, 0, 0.2 rad (~11 deg), 0]
    x0 = np.array([0.0, 0.0, 0.2, 0.0])
    states, controls = lqr.simulate(x0, n_steps=120)
    
    final_state = states[-1]
    print(f"Initial State: {x0}")
    print(f"Final State (t=120 steps): {final_state}")
    
    # Asserts:
    # 1. Closed-loop system must drive all states to 0 (origin regulation)
    assert np.linalg.norm(final_state) < 0.01, "LQR failed to stabilize system"
    assert np.linalg.norm(final_state) < 0.05 * np.linalg.norm(x0), "LQR failed to attenuate state"
    # 2. Closed-loop eigenvalues must lie strictly inside unit circle (Schur stable)
    closed_loop_A = A - B @ lqr.K
    eigenvalues = np.linalg.eigvals(closed_loop_A)
    max_magnitude = np.max(np.abs(eigenvalues))
    print(f"Max Closed-Loop Eigenvalue Magnitude: {max_magnitude:.4f}")
    assert max_magnitude < 1.0, "Closed-loop system must be strictly stable"
    print("U04 Sanity Check: PASSED cleanly.")

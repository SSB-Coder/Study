"""
X04. Extended Kalman Filter (EKF) for Nonlinear State Estimation
From-scratch implementation in pure NumPy.
"""

import numpy as np

class ExtendedKalmanFilter:
    """
    EKF for a mobile robot with non-linear unicycle kinematics:
    State: x = [px, py, theta]^T
    Control: u = [v, omega]^T (forward speed, yaw rate)
    Measurement: range and bearing to a known landmark: z = [range, bearing]^T
    """
    
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float) # Process noise covariance (3x3)
        self.R = np.array(R, dtype=float) # Measurement noise covariance (2x2)

    def predict(self, u: np.ndarray, dt: float):
        """
        Non-linear state transition f(x, u) and Jacobian F_k <- eq. 3.1.
        px_{k+1} = px_k + v * cos(theta_k) * dt
        py_{k+1} = py_k + v * sin(theta_k) * dt
        theta_{k+1} = theta_k + omega * dt
        """
        v, omega = u[0], u[1]
        theta = self.x[2]
        
        # State extrapolation
        self.x[0] += v * np.cos(theta) * dt
        self.x[1] += v * np.sin(theta) * dt
        self.x[2] += omega * dt
        # Normalize angle to [-pi, pi]
        self.x[2] = (self.x[2] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Jacobian of motion model w.r.t state x <- eq. 3.2
        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] =  v * np.cos(theta) * dt
        
        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, landmark: np.ndarray):
        """
        Non-linear measurement model h(x) and Jacobian H_k <- eq. 3.3.
        z = [r, phi]^T where:
        r = sqrt((lx - px)^2 + (ly - py)^2)
        phi = atan2(ly - py, lx - px) - theta
        """
        lx, ly = landmark[0], landmark[1]
        px, py, theta = self.x[0], self.x[1], self.x[2]
        
        dx = lx - px
        dy = ly - py
        q = dx**2 + dy**2
        r = np.sqrt(q)
        phi = np.arctan2(dy, dx) - theta
        phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
        
        z_pred = np.array([r, phi])
        
        # Measurement Jacobian H = dh / dx
        H = np.zeros((2, 3))
        H[0, 0] = -dx / r
        H[0, 1] = -dy / r
        H[0, 2] = 0.0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0
        
        # Innovation
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ y
        self.x[2] = (self.x[2] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

if __name__ == "__main__":
    print("--- X04: Extended Kalman Filter (EKF) Sanity Check ---")
    np.random.seed(42)
    dt = 0.1
    # True robot initial pose: [0, 0, 0]
    true_x = np.array([0.0, 0.0, 0.0])
    
    # EKF initialized with noisy prior
    ekf = ExtendedKalmanFilter(x0=np.array([0.5, -0.5, 0.1]),
                               P0=np.eye(3) * 1.0,
                               Q=np.diag([0.01, 0.01, 0.005]),
                               R=np.diag([0.05, 0.02]))
    
    landmark = np.array([10.0, 10.0])
    control_u = np.array([1.5, 0.1]) # v=1.5 m/s, omega=0.1 rad/s
    
    # Simulate 50 steps
    for step in range(50):
        # 1. Propagate true state
        true_x[0] += control_u[0] * np.cos(true_x[2]) * dt
        true_x[1] += control_u[0] * np.sin(true_x[2]) * dt
        true_x[2] += control_u[1] * dt
        
        # 2. EKF Predict
        ekf.predict(control_u, dt)
        
        # 3. Simulate noisy range-bearing measurement
        dx = landmark[0] - true_x[0]
        dy = landmark[1] - true_x[1]
        true_r = np.sqrt(dx**2 + dy**2)
        true_phi = np.arctan2(dy, dx) - true_x[2]
        
        noisy_z = np.array([
            true_r + np.random.normal(0, np.sqrt(0.05)),
            true_phi + np.random.normal(0, np.sqrt(0.02))
        ])
        
        # 4. EKF Update
        ekf.update(noisy_z, landmark)
        
    pos_error = np.linalg.norm(ekf.x[:2] - true_x[:2])
    print(f"True Robot Pose: {true_x}")
    print(f"EKF Estimated Pose: {ekf.x}")
    print(f"Final Estimation Error: {pos_error:.3f} m")
    
    assert pos_error < 0.5, "EKF failed to track true robot pose"
    print("X04 Sanity Check: PASSED cleanly.")

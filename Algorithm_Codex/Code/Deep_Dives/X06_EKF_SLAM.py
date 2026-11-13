"""
X06. EKF-SLAM (Simultaneous Localization and Mapping)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class EKFSLAM:
    """
    Online EKF-SLAM for 2D mobile robot in an unknown landmark environment.
    Joint State Vector: [rx, ry, r_theta, m1_x, m1_y, m2_x, m2_y, ...]^T
    Maintains full joint covariance matrix P tracking robot-map correlations.
    """
    
    def __init__(self, init_pose: np.ndarray, Q_robot: np.ndarray, R_sensor: np.ndarray):
        self.mu = np.array(init_pose, dtype=float) # Initial robot pose (3,)
        self.P = np.eye(3) * 1e-4                  # Robot pose covariance
        self.Q = Q_robot                           # Process noise (3x3)
        self.R = R_sensor                          # Measurement noise (2x2)
        self.landmark_map = {}                     # landmark_id -> index in state vector

    def predict(self, u: np.ndarray, dt: float):
        """Propagates robot pose and inflates robot-landmark cross-covariances <- eq. 3.1."""
        v, omega = u[0], u[1]
        theta = self.mu[2]
        
        # State extrapolation
        self.mu[0] += v * np.cos(theta) * dt
        self.mu[1] += v * np.sin(theta) * dt
        self.mu[2] += omega * dt
        self.mu[2] = (self.mu[2] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Jacobian F_x
        N = len(self.mu)
        F_x = np.eye(N)
        F_x[0, 2] = -v * np.sin(theta) * dt
        F_x[1, 2] =  v * np.cos(theta) * dt
        
        # Q_full has Q in top 3x3, 0 elsewhere
        Q_full = np.zeros((N, N))
        Q_full[:3, :3] = self.Q
        
        self.P = F_x @ self.P @ F_x.T + Q_full

    def update(self, lm_id: int, z: np.ndarray):
        """
        Observes landmark with relative [range, bearing].
        If new, initializes landmark into state vector. If existing, performs EKF update.
        """
        r, phi = z[0], z[1]
        
        # 1. Initialize new landmark if never seen before <- eq. 3.2
        if lm_id not in self.landmark_map:
            rx, ry, rth = self.mu[:3]
            # Landmark coordinates in global frame
            lx = rx + r * np.cos(rth + phi)
            ly = ry + r * np.sin(rth + phi)
            
            idx = len(self.mu)
            self.landmark_map[lm_id] = idx
            
            # Expand state vector
            self.mu = np.concatenate([self.mu, [lx, ly]])
            
            # Expand covariance matrix
            old_N = len(self.P)
            new_P = np.zeros((old_N + 2, old_N + 2))
            new_P[:old_N, :old_N] = self.P
            new_P[old_N:, old_N:] = np.eye(2) * 100.0 # High initial uncertainty
            self.P = new_P
            return

        # 2. Update existing landmark <- eq. 3.3
        idx = self.landmark_map[lm_id]
        lx, ly = self.mu[idx], self.mu[idx + 1]
        rx, ry, rth = self.mu[0], self.mu[1], self.mu[2]
        
        dx = lx - rx
        dy = ly - ry
        q = dx**2 + dy**2
        d_sqrt = np.sqrt(q)
        phi_pred = np.arctan2(dy, dx) - rth
        phi_pred = (phi_pred + np.pi) % (2.0 * np.pi) - np.pi
        z_pred = np.array([d_sqrt, phi_pred])
        
        # Measurement Jacobian H of size (2 x N)
        N = len(self.mu)
        H = np.zeros((2, N))
        # Derivatives w.r.t robot pose
        H[0, 0] = -dx / d_sqrt
        H[0, 1] = -dy / d_sqrt
        H[0, 2] = 0.0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0
        # Derivatives w.r.t landmark coordinates
        H[0, idx] = dx / d_sqrt
        H[0, idx + 1] = dy / d_sqrt
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q
        
        # Innovation
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.mu += K @ y
        self.mu[2] = (self.mu[2] + np.pi) % (2.0 * np.pi) - np.pi
        
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

if __name__ == "__main__":
    print("--- X06: EKF-SLAM Sanity Check ---")
    np.random.seed(42)
    dt = 0.1
    slam = EKFSLAM(init_pose=[0.0, 0.0, 0.0],
                   Q_robot=np.diag([0.01, 0.01, 0.002]),
                   R_sensor=np.diag([0.05, 0.02]))
    
    # 3 static world landmarks
    true_landmarks = {
        1: np.array([5.0, 5.0]),
        2: np.array([10.0, 0.0]),
        3: np.array([12.0, 8.0])
    }
    
    true_robot = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.05])
    
    for step in range(40):
        # Move true robot
        true_robot[0] += u[0] * np.cos(true_robot[2]) * dt
        true_robot[1] += u[0] * np.sin(true_robot[2]) * dt
        true_robot[2] += u[1] * dt
        
        slam.predict(u, dt)
        
        # Observe visible landmarks
        for lmid, lpos in true_landmarks.items():
            dx = lpos[0] - true_robot[0]
            dy = lpos[1] - true_robot[1]
            r = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx) - true_robot[2]
            z = np.array([
                r + np.random.normal(0, np.sqrt(0.05)),
                phi + np.random.normal(0, np.sqrt(0.02))
            ])
            slam.update(lmid, z)
            
    print(f"True Robot Pose: {true_robot}")
    print(f"SLAM Estimated Pose: {slam.mu[:3]}")
    pose_err = np.linalg.norm(slam.mu[:2] - true_robot[:2])
    print(f"Robot Position Error: {pose_err:.3f} m")
    
    for lmid in true_landmarks:
        idx = slam.landmark_map[lmid]
        lm_err = np.linalg.norm(slam.mu[idx:idx+2] - true_landmarks[lmid])
        print(f"Landmark {lmid} Error: {lm_err:.3f} m")
        assert lm_err < 1.0, f"Landmark {lmid} mapping failed"
        
    assert pose_err < 0.5, "SLAM localization failed"
    print("X06 Sanity Check: PASSED cleanly.")

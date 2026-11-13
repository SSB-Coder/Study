"""
X05. Unscented Kalman Filter (UKF) via the Scaled Unscented Transform
From-scratch implementation in pure NumPy.
"""

import numpy as np

class UnscentedKalmanFilter:
    """
    UKF using Julier-Uhlmann Scaled Unscented Transform.
    Approximates nonlinear probability distributions without explicit Jacobians.
    """
    
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.n = len(self.x)
        
        # Scaling parameters <- eq. 3.1
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = (alpha**2) * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        
        # Compute weights
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lam)))
        self.wc = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lam)))
        self.wm[0] = self.lam / (self.n + self.lam)
        self.wc[0] = self.lam / (self.n + self.lam) + (1.0 - alpha**2 + beta)

    def generate_sigma_points(self) -> np.ndarray:
        """Generates 2n + 1 deterministic sigma points <- eq. 3.2."""
        sigma_pts = np.zeros((2 * self.n + 1, self.n))
        sigma_pts[0] = self.x
        
        # Matrix square root via Cholesky decomposition
        L = np.linalg.cholesky(self.P)
        for i in range(self.n):
            sigma_pts[i + 1] = self.x + self.gamma * L[:, i]
            sigma_pts[self.n + i + 1] = self.x - self.gamma * L[:, i]
        return sigma_pts

    def predict(self, u: np.ndarray, dt: float):
        """Passes sigma points through nonlinear motion model."""
        sigmas = self.generate_sigma_points()
        v, omega = u[0], u[1]
        
        sigmas_out = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            px, py, th = sigmas[i]
            sigmas_out[i, 0] = px + v * np.cos(th) * dt
            sigmas_out[i, 1] = py + v * np.sin(th) * dt
            sigmas_out[i, 2] = th + omega * dt
            
        # Reconstruct mean
        self.x = np.sum(self.wm[:, np.newaxis] * sigmas_out, axis=0)
        self.x[2] = (self.x[2] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Reconstruct covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigmas_out)):
            diff = sigmas_out[i] - self.x
            diff[2] = (diff[2] + np.pi) % (2.0 * np.pi) - np.pi
            P_pred += self.wc[i] * np.outer(diff, diff)
            
        self.P = P_pred + self.Q

    def update(self, z: np.ndarray, landmark: np.ndarray):
        """Passes sigma points through nonlinear observation model."""
        sigmas = self.generate_sigma_points()
        lx, ly = landmark[0], landmark[1]
        
        Z_sigmas = np.zeros((len(sigmas), 2))
        for i in range(len(sigmas)):
            dx = lx - sigmas[i, 0]
            dy = ly - sigmas[i, 1]
            r = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx) - sigmas[i, 2]
            phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
            Z_sigmas[i] = [r, phi]
            
        # Measurement mean
        z_pred = np.sum(self.wm[:, np.newaxis] * Z_sigmas, axis=0)
        z_pred[1] = (z_pred[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        # Innovation covariance S and Cross-covariance P_xz <- eq. 3.3
        S = np.zeros((2, 2))
        Pxz = np.zeros((self.n, 2))
        
        for i in range(len(sigmas)):
            z_diff = Z_sigmas[i] - z_pred
            z_diff[1] = (z_diff[1] + np.pi) % (2.0 * np.pi) - np.pi
            S += self.wc[i] * np.outer(z_diff, z_diff)
            
            x_diff = sigmas[i] - self.x
            x_diff[2] = (x_diff[2] + np.pi) % (2.0 * np.pi) - np.pi
            Pxz += self.wc[i] * np.outer(x_diff, z_diff)
            
        S += self.R
        K = Pxz @ np.linalg.inv(S)
        
        # State & Covariance update
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        self.x += K @ y
        self.x[2] = (self.x[2] + np.pi) % (2.0 * np.pi) - np.pi
        self.P -= K @ S @ K.T

if __name__ == "__main__":
    print("--- X05: Unscented Kalman Filter (UKF) Sanity Check ---")
    np.random.seed(42)
    dt = 0.1
    true_x = np.array([0.0, 0.0, 0.0])
    ukf = UnscentedKalmanFilter(x0=np.array([0.5, -0.5, 0.1]),
                                P0=np.eye(3) * 1.0,
                                Q=np.diag([0.01, 0.01, 0.005]),
                                R=np.diag([0.05, 0.02]))
    
    landmark = np.array([10.0, 10.0])
    control_u = np.array([1.5, 0.1])
    
    for step in range(50):
        true_x[0] += control_u[0] * np.cos(true_x[2]) * dt
        true_x[1] += control_u[0] * np.sin(true_x[2]) * dt
        true_x[2] += control_u[1] * dt
        
        ukf.predict(control_u, dt)
        
        dx = landmark[0] - true_x[0]
        dy = landmark[1] - true_x[1]
        noisy_z = np.array([
            np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(0.05)),
            np.arctan2(dy, dx) - true_x[2] + np.random.normal(0, np.sqrt(0.02))
        ])
        ukf.update(noisy_z, landmark)
        
    pos_error = np.linalg.norm(ukf.x[:2] - true_x[:2])
    print(f"UKF Final Error: {pos_error:.3f} m")
    assert pos_error < 0.5, "UKF failed to track state"
    print("X05 Sanity Check: PASSED cleanly.")

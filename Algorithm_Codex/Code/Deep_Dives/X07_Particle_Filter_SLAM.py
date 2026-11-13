"""
X07. Particle Filter SLAM (FastSLAM-lite)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class FastSLAMPoseParticle:
    """
    A single FastSLAM particle containing:
    - Robot pose trajectory: [x, y, theta]
    - An independent 2D EKF for each landmark conditioned on this robot path.
    """
    def __init__(self, x0: np.ndarray, weight: float = 1.0):
        self.pose = np.array(x0, dtype=float)
        self.weight = weight
        # lm_id -> {'mu': np.ndarray (2,), 'P': np.ndarray (2, 2)}
        self.landmarks = {}

class FastSLAM:
    """
    Rao-Blackwellized Particle Filter for SLAM (FastSLAM 1.0).
    Decomposes joint SLAM posterior p(x_{1:t}, m | z_{1:t}, u_{1:t}) into:
    p(x_{1:t} | z, u) * prod_i p(m_i | x_{1:t}, z)
    Reduces O(M^2) compute to O(N_particles * M)!
    """
    
    def __init__(self, n_particles: int = 30, x0: np.ndarray = None,
                 Q_motion: np.ndarray = None, R_meas: np.ndarray = None):
        self.n_particles = n_particles
        if x0 is None: x0 = np.array([0.0, 0.0, 0.0])
        self.Q = np.diag([0.02, 0.02, 0.005]) if Q_motion is None else Q_motion
        self.R = np.diag([0.05, 0.02]) if R_meas is None else R_meas
        self.particles = [FastSLAMPoseParticle(x0, weight=1.0 / n_particles) for _ in range(n_particles)]

    def predict(self, u: np.ndarray, dt: float):
        """Samples noisy motion update for each particle <- eq. 3.1."""
        v, omega = u[0], u[1]
        for p in self.particles:
            noisy_v = v + np.random.normal(0, np.sqrt(self.Q[0, 0]))
            noisy_w = omega + np.random.normal(0, np.sqrt(self.Q[2, 2]))
            p.pose[0] += noisy_v * np.cos(p.pose[2]) * dt
            p.pose[1] += noisy_v * np.sin(p.pose[2]) * dt
            p.pose[2] += noisy_w * dt
            p.pose[2] = (p.pose[2] + np.pi) % (2.0 * np.pi) - np.pi

    def update(self, lm_id: int, z: np.ndarray):
        """Updates each particle's landmark EKF and weights particle by measurement likelihood <- eq. 3.2."""
        r, phi = z[0], z[1]
        
        for p in self.particles:
            rx, ry, rth = p.pose
            if lm_id not in p.landmarks:
                # Initialize 2D landmark EKF on this particle
                lx = rx + r * np.cos(rth + phi)
                ly = ry + r * np.sin(rth + phi)
                p.landmarks[lm_id] = {
                    'mu': np.array([lx, ly]),
                    'P': np.eye(2) * 50.0
                }
            else:
                # Update existing 2D landmark EKF
                lm = p.landmarks[lm_id]
                lx, ly = lm['mu']
                dx = lx - rx
                dy = ly - ry
                q = dx**2 + dy**2
                d_sqrt = np.sqrt(q)
                z_hat = np.array([d_sqrt, np.arctan2(dy, dx) - rth])
                z_hat[1] = (z_hat[1] + np.pi) % (2.0 * np.pi) - np.pi
                
                # 2x2 Jacobian w.r.t landmark coordinates only!
                H = np.array([
                    [dx / d_sqrt, dy / d_sqrt],
                    [-dy / q,     dx / q]
                ])
                
                # Innovation covariance
                S = H @ lm['P'] @ H.T + self.R
                K = lm['P'] @ H.T @ np.linalg.inv(S)
                
                # Innovation
                y = z - z_hat
                y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
                
                # Update landmark
                lm['mu'] += K @ y
                lm['P'] = (np.eye(2) - K @ H) @ lm['P']
                
                # Update particle weight based on Gaussian likelihood
                det_S = np.linalg.det(S)
                inv_S = np.linalg.inv(S)
                exponent = -0.5 * (y.T @ inv_S @ y)
                likelihood = np.exp(np.clip(exponent, -50.0, 0.0)) / np.sqrt((2.0 * np.pi)**2 * det_S + 1e-10)
                p.weight *= max(likelihood, 1e-10)

        # Normalize weights
        total_w = sum(p.weight for p in self.particles)
        for p in self.particles:
            p.weight /= total_w

    def resample_if_needed(self):
        """Low-variance systematic resampling when N_eff drops below threshold <- eq. 3.3."""
        weights = np.array([p.weight for p in self.particles])
        n_eff = 1.0 / np.sum(weights**2)
        
        if n_eff < self.n_particles / 2.0:
            new_particles = []
            cumsum = np.cumsum(weights)
            step = 1.0 / self.n_particles
            r = np.random.uniform(0, step)
            c_idx = 0
            
            for i in range(self.n_particles):
                u = r + i * step
                while u > cumsum[c_idx] and c_idx < self.n_particles - 1:
                    c_idx += 1
                # Deep copy particle
                src = self.particles[c_idx]
                clone = FastSLAMPoseParticle(src.pose.copy(), weight=1.0 / self.n_particles)
                for lmid, lm in src.landmarks.items():
                    clone.landmarks[lmid] = {'mu': lm['mu'].copy(), 'P': lm['P'].copy()}
                new_particles.append(clone)
            self.particles = new_particles

    def get_best_estimate(self):
        """Returns best particle's pose and map."""
        best_p = max(self.particles, key=lambda p: p.weight)
        return best_p.pose, best_p.landmarks

if __name__ == "__main__":
    print("--- X07: FastSLAM-lite Sanity Check ---")
    np.random.seed(42)
    dt = 0.1
    slam = FastSLAM(n_particles=25)
    
    true_landmarks = {
        1: np.array([6.0, 4.0]),
        2: np.array([12.0, -2.0])
    }
    
    true_robot = np.array([0.0, 0.0, 0.0])
    u = np.array([1.2, 0.08])
    
    for step in range(35):
        true_robot[0] += u[0] * np.cos(true_robot[2]) * dt
        true_robot[1] += u[0] * np.sin(true_robot[2]) * dt
        true_robot[2] += u[1] * dt
        
        slam.predict(u, dt)
        
        for lmid, lpos in true_landmarks.items():
            dx = lpos[0] - true_robot[0]
            dy = lpos[1] - true_robot[1]
            z = np.array([
                np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(0.05)),
                np.arctan2(dy, dx) - true_robot[2] + np.random.normal(0, np.sqrt(0.02))
            ])
            slam.update(lmid, z)
            
        slam.resample_if_needed()
        
    est_pose, est_lms = slam.get_best_estimate()
    print(f"True Robot Pose: {true_robot}")
    print(f"FastSLAM Best Pose: {est_pose}")
    pose_err = np.linalg.norm(est_pose[:2] - true_robot[:2])
    print(f"Pose Error: {pose_err:.3f} m")
    
    assert pose_err < 0.5, "FastSLAM localization error excessive"
    for lmid in true_landmarks:
        lm_err = np.linalg.norm(est_lms[lmid]['mu'] - true_landmarks[lmid])
        print(f"Landmark {lmid} Error: {lm_err:.3f} m")
        assert lm_err < 0.6, f"Landmark {lmid} mapping failed"
    print("X07 Sanity Check: PASSED cleanly.")

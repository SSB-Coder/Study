"""
E06. Kalman Filters & Particle Filters
From-scratch implementation in pure NumPy.
"""

import numpy as np

class KalmanFilter1D:
    """1D Constant-Velocity Linear Kalman Filter."""
    def __init__(self, dt: float = 1.0, process_noise_std: float = 0.1, measurement_noise_std: float = 0.5):
        self.dt = dt
        # State: [position, velocity]^T
        self.x = np.array([0.0, 0.0])
        # Transition matrix F <- eq. 3.1
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])
        # Measurement matrix H (we only measure position)
        self.H = np.array([[1.0, 0.0]])
        # Covariance matrices
        self.Q = (process_noise_std ** 2) * np.array([[(dt**4)/4, (dt**3)/2],
                                                      [(dt**3)/2, dt**2]])
        self.R = np.array([[measurement_noise_std ** 2]])
        self.P = np.eye(2) * 1.0

    def predict(self):
        # Time update <- eq. 3.1
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def update(self, measurement: float):
        # Measurement update <- eq. 3.1
        z = np.array([measurement])
        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + (K @ y_residual).ravel()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0]

class ParticleFilter1D:
    """Sequential Importance Resampling (SIR) Particle Filter for 1D non-linear tracking."""
    def __init__(self, n_particles: int = 500, process_std: float = 0.5, measurement_std: float = 1.0):
        self.n_particles = n_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        # Initialize particles uniformly
        self.particles = np.random.uniform(-10, 10, size=n_particles)
        self.weights = np.full(n_particles, 1.0 / n_particles)

    def predict(self, velocity: float = 1.0):
        # Propagate particles through non-linear physics + noise <- eq. 3.2
        self.particles += velocity + np.random.normal(0, self.process_std, size=self.n_particles)

    def update(self, measurement: float):
        # Weight particles by measurement likelihood
        diff = measurement - self.particles
        likelihood = np.exp(-0.5 * (diff / self.measurement_std) ** 2) + 1e-15
        self.weights *= likelihood
        self.weights /= np.sum(self.weights) # Normalize

        # Systematic Resampling if effective sample size drops
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2.0:
            indices = np.random.choice(self.n_particles, size=self.n_particles, replace=True, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.n_particles)

        return np.sum(self.particles * self.weights)

if __name__ == "__main__":
    print("--- E06: Kalman & Particle Filters Sanity Check ---")
    np.random.seed(42)
    # Simulate a 1D object moving with constant velocity v = 2.0
    n_steps = 30
    true_positions = [i * 2.0 for i in range(n_steps)]
    noisy_measurements = [pos + np.random.normal(0, 1.5) for pos in true_positions]

    # Test Kalman Filter
    kf = KalmanFilter1D(dt=1.0, process_noise_std=0.1, measurement_noise_std=1.5)
    kf_estimates = []
    for z in noisy_measurements:
        kf.predict()
        est = kf.update(z)
        kf_estimates.append(est)

    # Test Particle Filter
    pf = ParticleFilter1D(n_particles=500, process_std=0.5, measurement_std=1.5)
    pf.particles = np.random.normal(0, 1.0, 500)
    pf_estimates = []
    for z in noisy_measurements:
        pf.predict(velocity=2.0)
        est = pf.update(z)
        pf_estimates.append(est)

    kf_rmse = np.sqrt(np.mean((np.array(kf_estimates) - true_positions) ** 2))
    raw_rmse = np.sqrt(np.mean((np.array(noisy_measurements) - true_positions) ** 2))
    pf_rmse = np.sqrt(np.mean((np.array(pf_estimates) - true_positions) ** 2))

    print(f"Raw Measurement Noise RMSE: {raw_rmse:.4f}")
    print(f"Kalman Filter Tracking RMSE: {kf_rmse:.4f}")
    print(f"Particle Filter RMSE:        {pf_rmse:.4f}")
    assert kf_rmse < raw_rmse # Filter must substantially clean noisy measurements
    assert pf_rmse < raw_rmse
    print("E06 Sanity Check: PASSED cleanly.")

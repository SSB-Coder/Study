"""
W12. Multi-Body Dynamics — Slider-Crank Mechanism
From-scratch implementation in pure NumPy.
"""

import numpy as np

class SliderCrankMechanism:
    """
    Simulates the kinematics and dynamics of an internal combustion engine slider-crank linkage.
    Parameters:
    - r: Crank radius (m) (half stroke)
    - l: Connecting rod length (m)
    - m_piston: Mass of reciprocating piston (kg)
    - omega: Crank angular velocity (rad/s)
    """
    
    def __init__(self, r: float = 0.05, l: float = 0.15, m_piston: float = 0.5, omega: float = 314.159):
        self.r = r
        self.l = l
        self.m_piston = m_piston
        self.omega = omega
        self.lambda_ratio = r / l # Obliquity ratio <= 1/3 for typical engines

    def position(self, theta: np.ndarray) -> np.ndarray:
        """
        Exact kinematic displacement x(theta) from Top Dead Center (TDC) <- eq. 3.1:
        x = r * (1 - cos(theta)) + l * (1 - sqrt(1 - lambda^2 * sin^2(theta)))
        """
        theta = np.array(theta, dtype=float)
        term1 = self.r * (1.0 - np.cos(theta))
        term2 = self.l * (1.0 - np.sqrt(np.maximum(1.0 - (self.lambda_ratio * np.sin(theta))**2, 0.0)))
        return term1 + term2

    def velocity(self, theta: np.ndarray) -> np.ndarray:
        """
        Piston velocity v = dx / dt = omega * dx / dtheta <- eq. 3.2.
        """
        theta = np.array(theta, dtype=float)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        denom = np.sqrt(np.maximum(1.0 - (self.lambda_ratio * sin_t)**2, 1e-8))
        dx_dtheta = self.r * (sin_t + (self.lambda_ratio * sin_t * cos_t) / denom)
        return self.omega * dx_dtheta

    def acceleration(self, theta: np.ndarray) -> np.ndarray:
        """
        Piston acceleration a = dv / dt.
        Using 2nd-order harmonic Fourier series approximation:
        a approx r * omega^2 * (cos(theta) + lambda * cos(2 * theta)) <- eq. 3.3
        """
        theta = np.array(theta, dtype=float)
        return (self.r * self.omega**2) * (np.cos(theta) + self.lambda_ratio * np.cos(2.0 * theta))

    def inertia_force(self, theta: np.ndarray) -> np.ndarray:
        """
        Primary + Secondary reciprocating inertia forces:
        F_inertia = -m_piston * a(theta)
        """
        return -self.m_piston * self.acceleration(theta)

if __name__ == "__main__":
    print("--- W12: Slider-Crank Multi-Body Dynamics Sanity Check ---")
    # 3000 RPM engine: omega = 3000 * 2 * pi / 60 = 314.16 rad/s
    engine = SliderCrankMechanism(r=0.045, l=0.14, m_piston=0.45, omega=314.159)
    
    thetas = np.linspace(0, 2 * np.pi, 361) # 361 points so index 180 is exactly pi (BDC)
    pos = engine.position(thetas)
    vel = engine.velocity(thetas)
    acc = engine.acceleration(thetas)
    f_iner = engine.inertia_force(thetas)
    
    # Kinematic invariants:
    # 1. At Top Dead Center (TDC, theta = 0): displacement is 0, velocity is 0
    assert np.isclose(pos[0], 0.0, atol=1e-8), "Position at TDC must be 0"
    assert np.isclose(vel[0], 0.0, atol=1e-8), "Velocity at TDC must be 0"
    
    # 2. At Bottom Dead Center (BDC, theta = pi): displacement is 2 * r (stroke = 90 mm)
    bdc_idx = 180
    assert np.isclose(pos[bdc_idx], 2 * engine.r, atol=1e-4), "Displacement at BDC must equal 2*r"
    assert np.isclose(vel[bdc_idx], 0.0, atol=1e-8), "Velocity at BDC must be 0"
    
    # 3. Peak acceleration occurs at TDC
    max_acc_tdc = engine.acceleration(0.0)
    print(f"Peak Piston Acceleration at TDC: {max_acc_tdc:.2f} m/s^2 (~{max_acc_tdc / 9.81:.1f} g)")
    print(f"Peak Reciprocating Inertia Force: {abs(engine.inertia_force(0.0)):.2f} N")
    
    assert max_acc_tdc > 0.0
    print("W12 Sanity Check: PASSED cleanly.")

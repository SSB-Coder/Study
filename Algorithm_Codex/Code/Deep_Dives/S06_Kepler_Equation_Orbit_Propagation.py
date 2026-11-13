"""
S06. Orbital Mechanics — Kepler's Equation & Orbit Propagation
From-scratch implementation in pure NumPy.
"""

import numpy as np

class KeplerOrbitPropagator:
    """
    Propagates a two-body Keplerian elliptic orbit forward in time.
    Solves Kepler's transcendental equation M = E - e * sin(E) for eccentric anomaly E
    using Newton-Raphson iteration, then computes position and velocity in Cartesian space.
    """
    
    def __init__(self, semi_major_axis_km: float = 7000.0, eccentricity: float = 0.1,
                 mu: float = 398600.4418):
        """
        semi_major_axis_km: a in km (e.g. 7000 km for Low Earth Orbit)
        eccentricity: e in [0, 1) for elliptic orbits
        mu: Earth's standard gravitational parameter GM in km^3 / s^2
        """
        self.a = float(semi_major_axis_km)
        self.e = float(eccentricity)
        self.mu = float(mu)
        # Mean orbital motion n = sqrt(mu / a^3) <- eq. 3.1
        self.n = np.sqrt(self.mu / (self.a**3))
        # Orbital period T = 2 * pi / n
        self.period = 2.0 * np.pi / self.n

    def solve_kepler_equation(self, M: float, tol: float = 1e-12, max_iter: int = 50) -> float:
        """
        Solves Kepler's equation M = E - e * sin(E) for Eccentric Anomaly E <- eq. 3.2.
        Uses Newton-Raphson: E_{k+1} = E_k - (E_k - e * sin(E_k) - M) / (1 - e * cos(E_k))
        """
        # Initial guess: E0 = M + e * sin(M)
        E = M + self.e * np.sin(M)
        
        for _ in range(max_iter):
            f = E - self.e * np.sin(E) - M
            f_prime = 1.0 - self.e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                return E
        return E

    def propagate(self, dt: float, M0: float = 0.0):
        """
        Propagates orbit by dt seconds from initial Mean Anomaly M0.
        Returns Cartesian position [x, y] and velocity [vx, vy] in orbital plane.
        """
        # 1. Mean anomaly at time t <- eq. 3.1
        M = (M0 + self.n * dt) % (2.0 * np.pi)
        
        # 2. Solve for Eccentric Anomaly E
        E = self.solve_kepler_equation(M)
        
        # 3. True anomaly nu <- eq. 3.3
        sin_nu = (np.sqrt(1.0 - self.e**2) * np.sin(E)) / (1.0 - self.e * np.cos(E))
        cos_nu = (np.cos(E) - self.e) / (1.0 - self.e * np.cos(E))
        nu = np.arctan2(sin_nu, cos_nu)
        
        # 4. Orbital radius r
        r = self.a * (1.0 - self.e * np.cos(E))
        
        # 5. Position in perifocal orbital plane
        pos = np.array([r * np.cos(nu), r * np.sin(nu)])
        
        # 6. Velocity in orbital plane
        h = np.sqrt(self.mu * self.a * (1.0 - self.e**2)) # Specific angular momentum
        vel = np.array([
            -(self.mu / h) * np.sin(nu),
             (self.mu / h) * (self.e + np.cos(nu))
        ])
        
        return pos, vel, r, nu

if __name__ == "__main__":
    print("--- S06: Kepler Orbit Propagation Sanity Check ---")
    # Low Earth Orbit satellite: a = 7000 km, e = 0.05
    orbit = KeplerOrbitPropagator(semi_major_axis_km=7000.0, eccentricity=0.05)
    print(f"Orbital Period: {orbit.period / 60.0:.2f} minutes (~{orbit.period:.1f} s)")
    
    # 1. Propagate to Periapsis (t = 0): r_peri = a * (1 - e) = 7000 * 0.95 = 6650 km
    pos_p, vel_p, r_p, _ = orbit.propagate(dt=0.0)
    print(f"Periapsis Radius: {r_p:.2f} km (Expected: {orbit.a * (1.0 - orbit.e):.2f})")
    assert np.isclose(r_p, orbit.a * (1.0 - orbit.e), atol=1e-4)
    
    # 2. Propagate to Apoapsis (t = Period / 2): r_apo = a * (1 + e) = 7000 * 1.05 = 7350 km
    pos_a, vel_a, r_a, _ = orbit.propagate(dt=orbit.period / 2.0)
    print(f"Apoapsis Radius: {r_a:.2f} km (Expected: {orbit.a * (1.0 + orbit.e):.2f})")
    assert np.isclose(r_a, orbit.a * (1.0 + orbit.e), atol=1e-4)
    
    # 3. Vis-Viva Energy Conservation: v^2 = mu * (2/r - 1/a)
    v_sq_peri = np.sum(vel_p**2)
    v_sq_vis_viva = orbit.mu * (2.0 / r_p - 1.0 / orbit.a)
    print(f"Periapsis Vis-Viva v^2 Check: Computed={v_sq_peri:.4f}, Exact={v_sq_vis_viva:.4f}")
    assert np.isclose(v_sq_peri, v_sq_vis_viva, rtol=1e-6)
    print("S06 Sanity Check: PASSED cleanly.")

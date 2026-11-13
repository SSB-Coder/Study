"""
U03. Proportional-Integral-Derivative (PID) Controller
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PIDController:
    """
    Industrial PID controller with:
    - Proportional, Integral, and Derivative gains (Kp, Ki, Kd)
    - Anti-windup clamping to prevent integrator saturation
    - First-order derivative low-pass filter to reject measurement noise
    """
    
    def __init__(self, Kp: float = 2.0, Ki: float = 1.0, Kd: float = 0.5,
                 out_min: float = -10.0, out_max: float = 10.0, tau_d: float = 0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.out_min = out_min
        self.out_max = out_max
        self.tau_d = tau_d # Derivative filter time constant
        
        # Internal states
        self.integrator = 0.0
        self.prev_error = 0.0
        self.d_filter = 0.0

    def reset(self):
        self.integrator = 0.0
        self.prev_error = 0.0
        self.d_filter = 0.0

    def step(self, setpoint: float, measured: float, dt: float) -> float:
        """
        Computes control output u(t) <- eq. 3.1.
        """
        error = setpoint - measured
        
        # 1. Proportional term
        P = self.Kp * error
        
        # 2. Integral term with forward Euler integration
        self.integrator += self.Ki * error * dt
        # Anti-windup clamp <- eq. 3.2
        self.integrator = float(np.clip(self.integrator, self.out_min, self.out_max))
        I = self.integrator
        
        # 3. Derivative term with low-pass filtering to reject sensor spikes <- eq. 3.3
        # d_raw = (error - prev_error) / dt
        # Filter: tau_d * d_filter_dot + d_filter = d_raw
        alpha = dt / (self.tau_d + dt)
        d_raw = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.d_filter = (1.0 - alpha) * self.d_filter + alpha * d_raw
        D = self.Kd * self.d_filter
        
        self.prev_error = error
        
        # Total saturated control signal
        u_raw = P + I + D
        u_clamped = float(np.clip(u_raw, self.out_min, self.out_max))
        return u_clamped

class SecondOrderPlant:
    """Simulated mass-spring-damper physical plant: m*x_ddot + c*x_dot + k*x = u."""
    def __init__(self, m: float = 1.0, c: float = 0.5, k: float = 2.0):
        self.m = m
        self.c = c
        self.k = k
        self.x = 0.0
        self.v = 0.0

    def step(self, u: float, dt: float):
        # Semi-implicit Euler
        a = (u - self.c * self.v - self.k * self.x) / self.m
        self.v += a * dt
        self.x += self.v * dt
        return self.x

if __name__ == "__main__":
    print("--- U03: PID Controller Sanity Check ---")
    plant = SecondOrderPlant(m=1.0, c=0.5, k=1.0)
    pid = PIDController(Kp=15.0, Ki=8.0, Kd=4.0, out_min=-50.0, out_max=50.0)
    
    dt = 0.01
    sim_time = 5.0
    steps = int(sim_time / dt)
    
    target = 1.0 # Step response to setpoint = 1.0
    positions = []
    
    for _ in range(steps):
        u = pid.step(setpoint=target, measured=plant.x, dt=dt)
        pos = plant.step(u=u, dt=dt)
        positions.append(pos)
        
    final_pos = positions[-1]
    steady_state_error = abs(target - final_pos)
    print(f"Final Position: {final_pos:.4f} (Target: {target:.4f})")
    print(f"Steady-State Error: {steady_state_error:.2e}")
    
    # Asserts:
    # 1. Integrator eliminates steady-state error
    assert steady_state_error < 1e-3, "PID must eliminate steady-state error"
    # 2. System reaches setpoint within settling time
    assert abs(positions[int(steps * 0.8)] - target) < 0.05, "System must settle near target"
    print("U03 Sanity Check: PASSED cleanly.")

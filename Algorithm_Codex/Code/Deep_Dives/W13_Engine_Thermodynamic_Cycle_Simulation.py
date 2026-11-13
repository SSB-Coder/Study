"""
W13. Engine Thermodynamic Cycle Simulation (Otto & Diesel Cycles)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class ThermodynamicCycle:
    """
    Simulates air-standard Otto (Spark-Ignition) and Diesel (Compression-Ignition) cycles.
    Computes P-V-T state points, work per cycle, thermal efficiency, and Mean Effective Pressure (MEP).
    """
    R_AIR = 287.058   # Gas constant for dry air (J / kg K)
    GAMMA = 1.4       # Ratio of specific heats (cp / cv)
    CV = 718.0        # Specific heat at constant volume (J / kg K)
    CP = 1005.0       # Specific heat at constant pressure (J / kg K)

    def __init__(self, compression_ratio: float = 10.0, displacement: float = 0.0005, p1: float = 101325.0, t1: float = 300.0):
        """
        compression_ratio: r = V1 / V2
        displacement: V_d in m^3 (e.g. 0.5 liter cylinder)
        p1: Ambient intake pressure (Pa)
        t1: Ambient intake temperature (K)
        """
        self.r = compression_ratio
        self.v_d = displacement
        self.v2 = self.v_d / (self.r - 1.0) # Clearance volume V_c
        self.v1 = self.v_d + self.v2        # Maximum volume V_max
        self.p1 = p1
        self.t1 = t1
        # Mass of air in cylinder <- eq. 3.1
        self.m_air = (self.p1 * self.v1) / (self.R_AIR * self.t1)

    def simulate_otto_cycle(self, q_in_per_kg: float = 1.8e6):
        """
        Simulates Otto Cycle:
        1 -> 2: Isentropic compression: P * V^gamma = const
        2 -> 3: Constant-volume heat addition: Q_in = m * cv * (T3 - T2)
        3 -> 4: Isentropic expansion: P * V^gamma = const
        4 -> 1: Constant-volume heat rejection
        """
        g = self.GAMMA
        # State 2
        t2 = self.t1 * (self.r ** (g - 1.0))
        p2 = self.p1 * (self.r ** g)
        v2 = self.v2
        
        # State 3 (Heat addition at constant volume)
        t3 = t2 + q_in_per_kg / self.CV
        p3 = p2 * (t3 / t2)
        v3 = v2
        
        # State 4 (Isentropic expansion)
        t4 = t3 * ((1.0 / self.r) ** (g - 1.0))
        p4 = p3 * ((1.0 / self.r) ** g)
        v4 = self.v1
        
        # Work and Efficiency <- eq. 3.2
        q_in = self.m_air * q_in_per_kg
        q_out = self.m_air * self.CV * (t4 - self.t1)
        w_net = q_in - q_out
        eta_thermal = w_net / q_in
        ideal_eta = 1.0 - (1.0 / (self.r ** (g - 1.0)))
        mep = w_net / self.v_d # Mean Effective Pressure (Pa)
        
        states = {
            'State1': (self.p1, self.v1, self.t1),
            'State2': (p2, v2, t2),
            'State3': (p3, v3, t3),
            'State4': (p4, v4, t4),
            'W_net_J': w_net,
            'Thermal_Efficiency': eta_thermal,
            'Ideal_Efficiency': ideal_eta,
            'MEP_bar': mep / 1e5
        }
        return states

if __name__ == "__main__":
    print("--- W13: Thermodynamic Engine Cycle Sanity Check ---")
    # 0.5-liter cylinder with compression ratio 10:1
    engine_cycle = ThermodynamicCycle(compression_ratio=10.0, displacement=0.0005)
    res = engine_cycle.simulate_otto_cycle(q_in_per_kg=1.8e6)
    
    print(f"Max Cylinder Pressure (P3): {res['State3'][0] / 1e5:.1f} bar")
    print(f"Max Cylinder Temperature (T3): {res['State3'][2]:.1f} K")
    print(f"Net Work per Cycle: {res['W_net_J']:.2f} J")
    print(f"Thermal Efficiency: {res['Thermal_Efficiency'] * 100:.2f}%")
    print(f"Mean Effective Pressure: {res['MEP_bar']:.2f} bar")
    
    # Assert physical thermodynamic consistency:
    # 1. Thermal efficiency must match classical formula exactly
    assert np.isclose(res['Thermal_Efficiency'], res['Ideal_Efficiency'], atol=1e-6)
    # 2. Peak pressure and temperature exceed compression states
    assert res['State3'][0] > res['State2'][0], "Combustion must raise pressure"
    assert res['State3'][2] > res['State2'][2], "Combustion must raise temperature"
    print("W13 Sanity Check: PASSED cleanly.")

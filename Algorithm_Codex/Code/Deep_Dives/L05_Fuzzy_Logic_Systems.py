"""
L05: Fuzzy Logic Systems (Mamdani Inference Engine)
Pure NumPy implementation of Zadeh (1965) and Mamdani (1974).
Features triangular/trapezoidal membership functions, min-implication,
max-aggregation, and Center of Gravity (Centroid) defuzzification.
"""

import numpy as np

def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular fuzzy membership function."""
    y = np.zeros_like(x, dtype=float)
    # Left slope
    mask_left = (a < x) & (x <= b)
    if b > a:
        y[mask_left] = (x[mask_left] - a) / (b - a)
    # Right slope
    mask_right = (b < x) & (x < c)
    if c > b:
        y[mask_right] = (c - x[mask_right]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


class MamdaniFuzzyInference:
    """
    Classic Tipping/Control System:
    Inputs: service (0-10), food (0-10) -> Output: tip (0-25%)
    """
    def __init__(self, y_resolution: int = 500):
        self.y_domain = np.linspace(0.0, 25.0, y_resolution)

    def evaluate(self, service: float, food: float) -> float:
        # Fuzzification
        # Service membership: poor (0,0,5), good (0,5,10), excellent (5,10,10)
        s_poor = float(trimf(np.array([service]), -1.0, 0.0, 5.0)[0])
        s_good = float(trimf(np.array([service]), 0.0, 5.0, 10.0)[0])
        s_exc = float(trimf(np.array([service]), 5.0, 10.0, 11.0)[0])

        # Food membership: bad (0,0,5), delicious (5,10,10)
        f_bad = float(trimf(np.array([food]), -1.0, 0.0, 5.0)[0])
        f_del = float(trimf(np.array([food]), 5.0, 10.0, 11.0)[0])

        # Rule evaluation (Antecedent firing strengths via min/max)
        # Rule 1: IF service is poor OR food is bad THEN tip is low (0, 5, 10)
        alpha1 = max(s_poor, f_bad)
        # Rule 2: IF service is good THEN tip is medium (10, 15, 20)
        alpha2 = s_good
        # Rule 3: IF service is excellent OR food is delicious THEN tip is high (15, 20, 25)
        alpha3 = max(s_exc, f_del)

        # Implication (clip consequent fuzzy sets)
        tip_low = np.minimum(alpha1, trimf(self.y_domain, 0.0, 5.0, 10.0))
        tip_med = np.minimum(alpha2, trimf(self.y_domain, 10.0, 15.0, 20.0))
        tip_high = np.minimum(alpha3, trimf(self.y_domain, 15.0, 20.0, 25.0))

        # Aggregation (pointwise maximum)
        tip_aggregated = np.maximum(tip_low, np.maximum(tip_med, tip_high))

        # Defuzzification: Center of Gravity (Centroid)
        sum_area = np.sum(tip_aggregated)
        if sum_area == 0:
            return 12.5 # Default midpoint
        tip_centroid = np.sum(self.y_domain * tip_aggregated) / sum_area
        return float(tip_centroid)


if __name__ == "__main__":
    print("--- L05: Fuzzy Logic Systems (Mamdani Inference) Sanity Check ---")

    fuzzy_system = MamdaniFuzzyInference()

    # Case 1: Awful service and bad food -> Low tip (< 10%)
    tip_awful = fuzzy_system.evaluate(service=1.0, food=1.0)
    # Case 2: Average service and food -> Medium tip (~15%)
    tip_medium = fuzzy_system.evaluate(service=5.0, food=5.0)
    # Case 3: Outstanding service and food -> High tip (> 18%)
    tip_great = fuzzy_system.evaluate(service=9.5, food=9.5)

    print(f"Tip for [Service=1, Food=1] (Poor):      {tip_awful:.2f}%")
    print(f"Tip for [Service=5, Food=5] (Average):   {tip_medium:.2f}%")
    print(f"Tip for [Service=9.5, Food=9.5] (Superb): {tip_great:.2f}%")

    assert tip_awful < tip_medium < tip_great, "Fuzzy inference must exhibit strict monotonic ordering"
    assert tip_awful < 10.0, "Poor service must yield low tip (<10%)"
    assert tip_great > 18.0, "Superb service must yield high tip (>18%)"

    print("L05 Sanity Check: PASSED cleanly.")

"""
M04: Probabilistic Streaming Sketches (Count-Min Sketch, HyperLogLog, Reservoir Sampling)
Pure NumPy implementation of sub-linear streaming sketches:
1. Count-Min Sketch for frequency estimation with provable error bounds
2. HyperLogLog for distinct cardinality estimation via stochastic register averaging
3. Reservoir Sampler (Algorithm R) for unbiased uniform stream subsampling
"""

import numpy as np
from typing import List, Any


class CountMinSketch:
    """
    Count-Min Sketch (Cormode & Muthukrishnan).
    Width w = ceil(e / epsilon), Depth d = ceil(ln(1 / delta))
    """
    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        self.w = int(np.ceil(np.e / epsilon))
        self.d = int(np.ceil(np.log(1.0 / delta)))
        self.table = np.zeros((self.d, self.w), dtype=np.int64)
        
        # Universal hash parameters (a * x + b) % prime % w
        self.prime = 2147483647
        self.a = np.random.randint(1, self.prime - 1, size=self.d, dtype=np.int64)
        self.b = np.random.randint(0, self.prime - 1, size=self.d, dtype=np.int64)

    def _hash(self, row: int, item_id: int) -> int:
        return int(((self.a[row] * item_id + self.b[row]) % self.prime) % self.w)

    def update(self, item_id: int, count: int = 1) -> None:
        """Increments counters across all d rows."""
        for row in range(self.d):
            col = self._hash(row, item_id)
            self.table[row, col] += count

    def estimate(self, item_id: int) -> int:
        """Returns conservative minimum frequency estimate."""
        vals = [self.table[row, self._hash(row, item_id)] for row in range(self.d)]
        return int(min(vals))


class HyperLogLog:
    """
    HyperLogLog cardinality estimator with LinearCounting small-range correction.
    """
    def __init__(self, p: int = 8):
        self.p = p
        self.m = 1 << p  # 2^p registers
        self.registers = np.zeros(self.m, dtype=np.uint8)
        
        # Alpha bias correction factor
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def _hash64(self, item_id: int) -> int:
        """Simulates 64-bit integer hashing."""
        # Murmur-like mixing
        x = (item_id * 0x517cc1b727220a95) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 30)
        x = (x * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27)
        x = (x * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 31)
        return x

    def add(self, item_id: int) -> None:
        """Extracts register index from upper p bits and counts leading zeros in remainder."""
        h = self._hash64(item_id)
        register_idx = (h >> (64 - self.p)) & (self.m - 1)
        w = h & ((1 << (64 - self.p)) - 1)
        
        # Count leading zeros + 1
        if w == 0:
            leading_zeros = 64 - self.p + 1
        else:
            leading_zeros = int(64 - self.p - int(np.floor(np.log2(w))))
            
        self.registers[register_idx] = max(self.registers[register_idx], leading_zeros)

    def count(self) -> float:
        """Harmonic mean estimation with small-range fallback."""
        indicator = np.sum(2.0 ** (-self.registers.astype(float)))
        raw_estimate = self.alpha * (self.m ** 2) / indicator

        # Small range correction (LinearCounting)
        if raw_estimate <= 2.5 * self.m:
            v = np.sum(self.registers == 0)
            if v > 0:
                return float(self.m * np.log(self.m / v))
            
        return float(raw_estimate)


class ReservoirSampler:
    """Reservoir Sampling (Algorithm R) maintaining uniform k-sample over a stream."""
    def __init__(self, k: int):
        self.k = k
        self.reservoir: List[Any] = []
        self.total_seen = 0

    def feed(self, item: Any) -> None:
        self.total_seen += 1
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
        else:
            # Replace random slot with probability k / total_seen
            j = np.random.randint(0, self.total_seen)
            if j < self.k:
                self.reservoir[j] = item

    def get_sample(self) -> List[Any]:
        return list(self.reservoir)


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Test Count-Min Sketch
    cms = CountMinSketch(epsilon=0.02, delta=0.01)
    true_counts = {101: 500, 202: 1200, 303: 45, 404: 10}
    for item_id, count in true_counts.items():
        cms.update(item_id, count)

    # Add stream noise
    for _ in range(5000):
        noise_id = np.random.randint(1000, 9999)
        cms.update(noise_id, 1)

    print("Count-Min Sketch Estimates:")
    for item_id, count in true_counts.items():
        est = cms.estimate(item_id)
        print(f"  Item {item_id}: True={count}, Est={est}")
        assert est >= count, f"CMS must never underestimate true count"
        assert est <= count + 5000 * 0.05, f"CMS estimate exceeded error bound"

    # 2. Test HyperLogLog
    hll = HyperLogLog(p=8)
    n_unique = 15000
    for i in range(n_unique):
        hll.add(i)

    cardinality_est = hll.count()
    error_pct = abs(cardinality_est - n_unique) / n_unique * 100
    print(f"HyperLogLog: True={n_unique}, Estimated={cardinality_est:.1f}, Error={error_pct:.2f}%")
    assert error_pct < 15.0, f"HLL error too high: {error_pct:.2f}%"

    # 3. Test Reservoir Sampler
    sampler = ReservoirSampler(k=100)
    for i in range(10000):
        sampler.feed(i)
    sample = sampler.get_sample()
    assert len(sample) == 100, "Reservoir sample size must equal k"
    print(f"Reservoir Sample: len={len(sample)}, min={min(sample)}, max={max(sample)}")

    print("M04 Sanity Check: PASSED cleanly.")

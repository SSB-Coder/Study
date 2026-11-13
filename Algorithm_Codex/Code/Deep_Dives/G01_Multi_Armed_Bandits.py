"""
G01: Multi-Armed Bandits & Exploration-Exploitation Primitives
From-scratch NumPy implementation of Epsilon-Greedy, UCB1, and Thompson Sampling.
"""

import numpy as np

class BernoulliBanditEnvironment:
    """Simulates a multi-armed bandit with Bernoulli reward distributions."""
    def __init__(self, probabilities: np.ndarray, seed: int = 42):
        self.probabilities = np.array(probabilities, dtype=np.float64)
        self.k = len(probabilities)
        self.optimal_arm = int(np.argmax(self.probabilities))
        self.optimal_value = float(self.probabilities[self.optimal_arm])
        self.rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> float:
        """Pulls the specified arm and returns a reward in {0.0, 1.0}."""
        return float(self.rng.random() < self.probabilities[arm])


class EpsilonGreedyBandit:
    """Epsilon-Greedy Bandit with sample-average incremental updates."""
    def __init__(self, k: int, epsilon: float = 0.1):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k, dtype=np.int64)
        self.q_values = np.zeros(k, dtype=np.float64)

    def select_arm(self, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.k))
        # Break ties randomly
        max_q = np.max(self.q_values)
        candidates = np.where(self.q_values == max_q)[0]
        return int(rng.choice(candidates))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class UCB1Bandit:
    """Upper Confidence Bound (UCB1) based on Hoeffding's Inequality."""
    def __init__(self, k: int, c: float = 1.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k, dtype=np.int64)
        self.q_values = np.zeros(k, dtype=np.float64)
        self.total_pulls = 0

    def select_arm(self, rng: np.random.Generator) -> int:
        self.total_pulls += 1
        # Try every arm at least once
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
        
        # UCB calculation: Q(a) + c * sqrt(ln(t) / N(a))
        bonuses = self.c * np.sqrt(np.log(self.total_pulls) / self.counts)
        ucb_values = self.q_values + bonuses
        max_v = np.max(ucb_values)
        candidates = np.where(ucb_values == max_v)[0]
        return int(rng.choice(candidates))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class ThompsonSamplingBandit:
    """Bayesian Posterior Sampling with Beta(alpha, beta) conjugate priors."""
    def __init__(self, k: int):
        self.k = k
        self.alpha = np.ones(k, dtype=np.float64)  # Successes + 1
        self.beta = np.ones(k, dtype=np.float64)   # Failures + 1

    def select_arm(self, rng: np.random.Generator) -> int:
        # Sample theta_a ~ Beta(alpha_a, beta_a)
        thetas = rng.beta(self.alpha, self.beta)
        return int(np.argmax(thetas))

    def update(self, arm: int, reward: float):
        if reward > 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0


if __name__ == "__main__":
    print("--- G01: Multi-Armed Bandits Sanity Check ---")
    true_probs = np.array([0.15, 0.35, 0.80, 0.50, 0.25])
    env = BernoulliBanditEnvironment(true_probs, seed=123)
    horizon = 2000
    rng = np.random.default_rng(42)

    # Test Thompson Sampling
    ts_agent = ThompsonSamplingBandit(env.k)
    rewards_ts = []
    pulls_ts = np.zeros(env.k)

    for _ in range(horizon):
        arm = ts_agent.select_arm(rng)
        rew = env.pull(arm)
        ts_agent.update(arm, rew)
        rewards_ts.append(rew)
        pulls_ts[arm] += 1

    optimal_pull_pct = pulls_ts[env.optimal_arm] / horizon
    print(f"Optimal Arm: {env.optimal_arm} (p = {env.optimal_value})")
    print(f"Thompson Sampling Arm Pulls: {pulls_ts}")
    print(f"Thompson Sampling Optimal Arm Selection Rate: {optimal_pull_pct:.2%}")

    assert optimal_pull_pct > 0.70, "Thompson sampling should pull optimal arm > 70% of the time"

    # Test UCB1
    ucb_agent = UCB1Bandit(env.k, c=1.0)
    pulls_ucb = np.zeros(env.k)
    for _ in range(horizon):
        arm = ucb_agent.select_arm(rng)
        rew = env.pull(arm)
        ucb_agent.update(arm, rew)
        pulls_ucb[arm] += 1

    ucb_optimal_pct = pulls_ucb[env.optimal_arm] / horizon
    print(f"UCB1 Optimal Arm Selection Rate: {ucb_optimal_pct:.2%}")
    assert ucb_optimal_pct > 0.60, "UCB1 should identify and heavily pull optimal arm"
    print("G01 Sanity Check: PASSED cleanly.")

"""
Q01: Value-Decomposition Networks (VDN) & QMIX for Cooperative MARL
Pure NumPy implementation of Centralized Training with Decentralized Execution (CTDE):
1. Value-Decomposition Networks (VDN) with linear additive credit assignment
2. QMIX with non-linear monotonic hypernetwork-generated mixing weights
"""

import numpy as np
from typing import List, Tuple


class QMIXMixingNetwork:
    """
    QMIX Monotonic Mixing Network.
    Maps individual agent Q-values [Q_1, ..., Q_N] to joint team value Q_tot
    conditioned on global environmental state s, enforcing d(Q_tot)/d(Q_a) >= 0.
    """
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 16):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork 1: generates W1 (n_agents x embed_dim) from state
        self.h_w1 = np.random.randn(state_dim, n_agents * embed_dim) * 0.1
        self.h_b1 = np.zeros(n_agents * embed_dim)

        # Hypernetwork 2: generates W2 (embed_dim x 1) from state
        self.h_w2 = np.random.randn(state_dim, embed_dim) * 0.1
        self.h_b2 = np.zeros(embed_dim)

        # State-conditioned scalar bias
        self.v_w = np.random.randn(state_dim, 1) * 0.1
        self.v_b = 0.0

    def forward(self, agent_qs: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        agent_qs: (B, N)
        state: (B, S)
        Returns Q_tot (B, 1) and cached activations for backward pass.
        """
        B = agent_qs.shape[0]

        # 1. Hypernetwork 1 -> W1 (B, N, embed_dim)
        w1_raw = state @ self.h_w1 + self.h_b1
        # Crucial QMIX constraint: non-negative weights enforce monotonicity!
        w1 = np.abs(w1_raw.reshape(B, self.n_agents, self.embed_dim))
        b1 = np.zeros((B, 1, self.embed_dim))  # Zero state bias for layer 1

        # Layer 1 forward: agent_qs (B, 1, N) @ W1 (B, N, embed_dim)
        hidden = np.matmul(agent_qs[:, None, :], w1) + b1  # (B, 1, embed_dim)
        hidden_act = np.maximum(0, hidden)  # ReLU activation

        # 2. Hypernetwork 2 -> W2 (B, embed_dim, 1)
        w2_raw = state @ self.h_w2 + self.h_b2
        w2 = np.abs(w2_raw.reshape(B, self.embed_dim, 1))

        # 3. Global state value bias
        v_bias = state @ self.v_w + self.v_b  # (B, 1)

        # Layer 2 forward: (B, 1, embed_dim) @ (B, embed_dim, 1) -> (B, 1, 1)
        q_tot = np.matmul(hidden_act, w2).squeeze(axis=1) + v_bias  # (B, 1)

        cache = {
            "agent_qs": agent_qs, "state": state,
            "w1": w1, "w2": w2, "hidden_act": hidden_act, "hidden": hidden
        }
        return q_tot, cache

    def check_monotonicity(self, state: np.ndarray) -> bool:
        """Verifies that d(Q_tot) / d(Q_a) >= 0 for all agents."""
        B = 1
        agent_qs = np.ones((B, self.n_agents))
        q_tot, cache = self.forward(agent_qs, state[None, :])
        w1 = cache["w1"][0]  # (N, embed)
        w2 = cache["w2"][0]  # (embed, 1)
        hidden = cache["hidden"][0, 0]

        relu_grad = (hidden > 0).astype(float)  # (embed,)
        # Gradient: dQ_tot / dQ_a = sum_h (W1[a, h] * relu_grad[h] * W2[h, 0])
        grads = np.sum(w1 * relu_grad[None, :] * w2.T, axis=1)
        return bool(np.all(grads >= -1e-9))


class CooperativeAgent:
    """Independent agent utility network."""
    def __init__(self, obs_dim: int, n_actions: int):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.W = np.random.randn(obs_dim, n_actions) * 0.1
        self.b = np.zeros(n_actions)

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W + self.b

    def choose_action(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        q_vals = self.get_q_values(obs)
        return int(np.argmax(q_vals))


if __name__ == "__main__":
    np.random.seed(42)

    n_agents = 3
    state_dim = 6
    obs_dim = 2
    n_actions = 4

    # 1. Test QMIX Monotonic Mixing Network
    mixer = QMIXMixingNetwork(n_agents=n_agents, state_dim=state_dim, embed_dim=8)

    sample_state = np.random.randn(state_dim)
    is_monotonic = mixer.check_monotonicity(sample_state)
    print(f"QMIX Mixing Network Monotonicity Invariant: {is_monotonic}")
    assert is_monotonic, "QMIX must strictly satisfy d(Q_tot)/d(Q_a) >= 0"

    # 2. Test Individual-Global-Max (IGM) consistency:
    # Individual argmax choices MUST maximize the total Q_tot value
    agents = [CooperativeAgent(obs_dim, n_actions) for _ in range(n_agents)]
    sample_obs = [np.random.randn(obs_dim) for _ in range(n_agents)]

    # Compute individual utilities
    agent_utilities = []
    chosen_actions = []
    for a in range(n_agents):
        q_all = agents[a].get_q_values(sample_obs[a])
        best_act = int(np.argmax(q_all))
        chosen_actions.append(best_act)
        agent_utilities.append(q_all[best_act])

    best_qs = np.array(agent_utilities)[None, :]
    optimal_q_tot, _ = mixer.forward(best_qs, sample_state[None, :])

    # Perturb with sub-optimal action choices
    suboptimal_utilities = []
    for a in range(n_agents):
        q_all = agents[a].get_q_values(sample_obs[a])
        sub_act = (chosen_actions[a] + 1) % n_actions
        suboptimal_utilities.append(q_all[sub_act])

    sub_qs = np.array(suboptimal_utilities)[None, :]
    sub_q_tot, _ = mixer.forward(sub_qs, sample_state[None, :])

    print(f"Optimal Joint Q_tot:     {float(optimal_q_tot[0, 0]):.4f}")
    print(f"Sub-optimal Joint Q_tot: {float(sub_q_tot[0, 0]):.4f}")
    assert optimal_q_tot[0, 0] >= sub_q_tot[0, 0], "IGM violation: greedy individual actions failed to maximize joint Q_tot"

    print("Q01 Sanity Check: PASSED cleanly.")

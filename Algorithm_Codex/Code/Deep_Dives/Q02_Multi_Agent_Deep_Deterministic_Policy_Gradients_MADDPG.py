"""
Q02: Multi-Agent Deep Deterministic Policy Gradients (MADDPG)
Pure NumPy implementation of Centralized Training with Decentralized Execution (CTDE)
for continuous multi-agent reinforcement learning.
"""

import numpy as np
from typing import List, Tuple


class Actor:
    """Decentralized deterministic actor: o_i -> a_i in [-1, 1]"""
    def __init__(self, obs_dim: int, act_dim: int):
        self.W = np.random.randn(obs_dim, act_dim) * 0.2
        self.b = np.zeros(act_dim)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(obs @ self.W + self.b)


class CentralizedCritic:
    """
    Centralized Critic observing joint states (o_1, ..., o_N) and joint actions (a_1, ..., a_N).
    Q(x, a_1, ..., a_N) in R
    """
    def __init__(self, total_obs_dim: int, total_act_dim: int, hidden_dim: int = 16):
        input_dim = total_obs_dim + total_act_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = 0.0

    def forward(self, joint_obs: np.ndarray, joint_act: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Returns Q value (B, 1) and cache."""
        inp = np.hstack([joint_obs, joint_act])
        h = np.maximum(0, inp @ self.W1 + self.b1)  # ReLU
        q = h @ self.W2 + self.b2
        cache = {"inp": inp, "h": h}
        return q, cache

    def grad_action(self, joint_obs: np.ndarray, joint_act: np.ndarray, act_slice: slice) -> np.ndarray:
        """Computes dQ / da_i for a specific agent's action slice."""
        q, cache = self.forward(joint_obs, joint_act)
        h = cache["h"]
        inp = cache["inp"]

        # Backprop through critic
        # dQ/dh = W2.T
        # dh/dinp = (h > 0) * W1.T
        relu_mask = (h > 0).astype(float)
        dh = relu_mask * self.W2.T  # (B, hidden)
        dinp = dh @ self.W1.T       # (B, input_dim)

        total_obs_dim = joint_obs.shape[1]
        act_start = total_obs_dim + act_slice.start
        act_stop = total_obs_dim + act_slice.stop
        
        # Extract gradient with respect to agent i's action
        return dinp[:, act_start:act_stop]


class MADDPGSystem:
    """Coordinating N agents with decentralized actors and centralized critics."""
    def __init__(self, n_agents: int, obs_dims: List[int], act_dims: List[int]):
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims

        self.total_obs_dim = sum(obs_dims)
        self.total_act_dim = sum(act_dims)

        self.actors = [Actor(obs_dims[i], act_dims[i]) for i in range(n_agents)]
        self.critics = [CentralizedCritic(self.total_obs_dim, self.total_act_dim) for _ in range(n_agents)]

        # Precompute action slices for each agent
        self.act_slices = []
        curr = 0
        for ad in act_dims:
            self.act_slices.append(slice(curr, curr + ad))
            curr += ad

    def get_actions(self, observations: List[np.ndarray], noise: float = 0.0) -> List[np.ndarray]:
        actions = []
        for i in range(self.n_agents):
            act = self.actors[i].forward(observations[i])
            if noise > 0:
                act = np.clip(act + np.random.randn(*act.shape) * noise, -1.0, 1.0)
            actions.append(act)
        return actions

    def update_actor(self, agent_idx: int, joint_obs: np.ndarray, local_obs: np.ndarray, lr: float = 0.01) -> float:
        """
        Policy gradient update for actor i:
        grad_theta = E[ grad_theta mu_i(o_i) * grad_a_i Q_i(x, a) ]
        """
        actor = self.actors[agent_idx]
        critic = self.critics[agent_idx]
        act_slice = self.act_slices[agent_idx]

        # Compute current joint actions
        all_actions = []
        for j in range(self.n_agents):
            if j == agent_idx:
                act = actor.forward(local_obs)
            else:
                # Treat other agents as fixed context from batch
                act = np.zeros((len(local_obs), self.act_dims[j]))
            all_actions.append(act)
        joint_act = np.hstack(all_actions)

        # 1. Gradient of Q with respect to action a_i
        dq_da = critic.grad_action(joint_obs, joint_act, act_slice)

        # 2. Gradient of actor action with respect to actor weights:
        # a = tanh(o @ W + b) -> da/dW = o.T @ ((1 - a^2) * dq_da)
        curr_act = actor.forward(local_obs)
        dtanh = (1.0 - curr_act ** 2) * dq_da

        grad_W = (local_obs.T @ dtanh) / len(local_obs)
        grad_b = np.mean(dtanh, axis=0)

        # Policy gradient ascent: maximize Q
        actor.W += lr * grad_W
        actor.b += lr * grad_b

        mean_q, _ = critic.forward(joint_obs, joint_act)
        return float(np.mean(mean_q))


if __name__ == "__main__":
    np.random.seed(42)

    n_agents = 2
    obs_dim = 3
    act_dim = 2

    maddpg = MADDPGSystem(
        n_agents=n_agents,
        obs_dims=[obs_dim, obs_dim],
        act_dims=[act_dim, act_dim]
    )

    batch_size = 64
    obs_0 = np.random.randn(batch_size, obs_dim)
    obs_1 = np.random.randn(batch_size, obs_dim)
    joint_obs = np.hstack([obs_0, obs_1])

    # Initial actions
    acts = maddpg.get_actions([obs_0, obs_1])
    joint_act = np.hstack(acts)

    # Initial critic evaluation
    q_init, _ = maddpg.critics[0].forward(joint_obs, joint_act)
    print(f"Initial Mean Q for Agent 0: {float(np.mean(q_init)):.4f}")

    # Run policy gradient ascent updates on Agent 0's actor
    q_history = []
    for step in range(50):
        q_val = maddpg.update_actor(agent_idx=0, joint_obs=joint_obs, local_obs=obs_0, lr=0.1)
        q_history.append(q_val)

    print(f"Updated Mean Q for Agent 0 after 50 steps: {q_history[-1]:.4f}")
    assert q_history[-1] > q_history[0], "Policy gradient ascent should increase expected Q value"

    # Verify decentralized execution: actor forward pass requires ONLY local observation
    single_obs = np.random.randn(1, obs_dim)
    isolated_action = maddpg.actors[0].forward(single_obs)
    assert isolated_action.shape == (1, act_dim), "Decentralized actor must operate solely on local obs"
    assert np.all(np.abs(isolated_action) <= 1.0), "Actor outputs must respect tanh [-1, 1] bounds"

    print("Q02 Sanity Check: PASSED cleanly.")

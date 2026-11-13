"""
G05: Deep Q-Networks (DQN)
From-scratch NumPy implementation of a 2-layer Neural Network DQN with Replay Buffer.
"""

import numpy as np

class ReplayBuffer:
    """Fixed-capacity ring buffer for experience replay."""
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.dones = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.size = 0

    def store(self, s: np.ndarray, a: int, r: float, next_s: np.ndarray, done: bool):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = next_s
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=batch_size)
        return (self.states[indices], self.actions[indices], 
                self.rewards[indices], self.next_states[indices], self.dones[indices])


class MLPActionValue:
    """2-layer MLP (Input -> Hidden ReLU -> Output Linear) for Q-values in NumPy."""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, lr: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        # He initialization
        rng = np.random.default_rng(123)
        self.W1 = rng.normal(0, np.sqrt(2.0 / state_dim), (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, action_dim))
        self.b2 = np.zeros(action_dim)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X: (B, state_dim)
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Q = A1 @ self.W2 + self.b2
        return Q, A1

    def backward_step(self, X: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """
        Updates weights via MSE loss on selected actions:
        Loss = 1/B sum (target - Q(s, a))^2
        """
        B = X.shape[0]
        Q, A1 = self.forward(X)
        
        # Gradient of loss w.r.t Q output
        dQ = np.zeros_like(Q)
        for i in range(B):
            dQ[i, actions[i]] = -2.0 * (targets[i] - Q[i, actions[i]]) / B

        # Backpropagation
        dW2 = A1.T @ dQ
        db2 = np.sum(dQ, axis=0)

        dA1 = dQ @ self.W2.T
        dZ1 = dA1 * (A1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        # Gradient clipping and parameter update
        np.clip(dW1, -1.0, 1.0, out=dW1)
        np.clip(dW2, -1.0, 1.0, out=dW2)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def copy_weights_from(self, source: 'MLPActionValue'):
        self.W1 = np.copy(source.W1)
        self.b1 = np.copy(source.b1)
        self.W2 = np.copy(source.W2)
        self.b2 = np.copy(source.b2)


class DQNAgent:
    """Deep Q-Network with experience replay and target network."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.01, gamma: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.online_net = MLPActionValue(state_dim, hidden_dim=16, action_dim=action_dim, lr=lr)
        self.target_net = MLPActionValue(state_dim, hidden_dim=16, action_dim=action_dim, lr=lr)
        self.target_net.copy_weights_from(self.online_net)
        self.buffer = ReplayBuffer(capacity=2000, state_dim=state_dim)

    def select_action(self, s: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.action_dim))
        Q, _ = self.online_net.forward(s.reshape(1, -1))
        return int(np.argmax(Q[0]))

    def train_step(self, batch_size: int, rng: np.random.Generator):
        if self.buffer.size < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size, rng)
        
        # Compute targets using target network
        next_Q, _ = self.target_net.forward(next_states)
        max_next_q = np.max(next_Q, axis=1)
        targets = rewards + (1.0 - dones.astype(np.float64)) * self.gamma * max_next_q

        # Optimize online network
        self.online_net.backward_step(states, actions, targets)


if __name__ == "__main__":
    print("--- G05: Deep Q-Network (DQN) Sanity Check ---")
    rng = np.random.default_rng(42)
    agent = DQNAgent(state_dim=2, action_dim=2, lr=0.02, gamma=0.9)

    # Synthetic 2D target reaching task:
    # State: [x, velocity]. If x > 0, optimal action is 1; if x < 0, optimal action is 0.
    for step in range(600):
        s = rng.uniform(-1.0, 1.0, size=2)
        a = rng.integers(0, 2)
        # Reward +1 if action moves toward positive sign of state[0]
        r = 1.0 if (s[0] > 0 and a == 1) or (s[0] <= 0 and a == 0) else -1.0
        next_s = s + 0.1 * (2 * a - 1)
        done = bool(rng.random() < 0.1)
        agent.buffer.store(s, a, r, next_s, done)

        if step > 50:
            agent.train_step(batch_size=16, rng=rng)
        if step % 50 == 0:
            agent.target_net.copy_weights_from(agent.online_net)

    # Sanity evaluation
    test_pos = np.array([0.8, 0.0])
    test_neg = np.array([-0.8, 0.0])
    act_pos = agent.select_action(test_pos, epsilon=0.0, rng=rng)
    act_neg = agent.select_action(test_neg, epsilon=0.0, rng=rng)

    print(f"State [0.8, 0] -> Chosen Action: {act_pos} (Target: 1)")
    print(f"State [-0.8, 0] -> Chosen Action: {act_neg} (Target: 0)")
    assert act_pos == 1 and act_neg == 0, "DQN failed to learn directional task!"
    print("G05 Sanity Check: PASSED cleanly.")

"""
21. Active Inference & Variational Free Energy
Deep Dive Implementation

This file explicitly demonstrates an Active Inference agent.
Unlike RL which uses a Reward function, this agent calculates "Surprise" 
(Variational Free Energy) and takes actions specifically to minimize that surprise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GenerativeWorldModel(nn.Module):
    """
    The Agent's internal brain. 
    It learns to predict the sensory observations (o) given the hidden state of the world (s).
    """
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        # P(o | s): The Likelihood Model
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim)
        )
        
    def forward(self, expected_state):
        return self.net(expected_state)

class ActiveInferenceAgent:
    def __init__(self, state_dim=4, obs_dim=10):
        self.world_model = GenerativeWorldModel(state_dim, obs_dim)
        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.01)
        
        # The agent's internal Belief about the world state Q(s)
        # It updates this belief constantly to match reality
        self.internal_belief = torch.randn(1, state_dim, requires_grad=True)
        # We need a separate optimizer just to update the belief!
        self.belief_optimizer = torch.optim.Adam([self.internal_belief], lr=0.1)

    def calculate_free_energy(self, sensory_observation):
        """
        Calculates F = Complexity - Accuracy
        (Simplified for educational demonstration)
        """
        # 1. What does the agent *predict* it will see based on its internal belief?
        predicted_observation = self.world_model(self.internal_belief)
        
        # 2. Accuracy (Reconstruction Loss): How wrong was the prediction?
        # Minimized when prediction perfectly matches the real sensory input.
        accuracy_loss = F.mse_loss(predicted_observation, sensory_observation)
        
        # 3. Complexity (Prior Loss): Is the internal belief getting too wild?
        # Minimized when the internal belief stays close to a standard normal prior N(0,1)
        prior = torch.zeros_like(self.internal_belief)
        complexity_loss = F.mse_loss(self.internal_belief, prior)
        
        # Total Variational Free Energy
        free_energy = accuracy_loss + (0.1 * complexity_loss)
        return free_energy

    def perception_step(self, sensory_observation):
        """
        Minimize Free Energy by updating the internal BELIEF (Q) to match reality.
        This is "changing your mind."
        """
        self.belief_optimizer.zero_grad()
        free_energy = self.calculate_free_energy(sensory_observation)
        
        # Backpropagate into the belief tensor, NOT the model weights!
        free_energy.backward()
        self.belief_optimizer.step()
        
        return free_energy.item()

    def action_step(self, sensory_observation, possible_actions):
        """
        Minimize Free Energy by taking an ACTION that forces reality to match the belief.
        We evaluate possible actions and pick the one that reduces predicted surprise.
        """
        best_action = None
        lowest_expected_free_energy = float('inf')
        
        # In a real setup, actions modify the environment state.
        # Here we mock it by applying an action matrix to the sensory observation.
        for action in possible_actions:
            # Simulate the future observation if we take this action
            hypothetical_obs = sensory_observation + action
            
            # Calculate what our surprise (Free Energy) WOULD be
            expected_free_energy = self.calculate_free_energy(hypothetical_obs).item()
            
            if expected_free_energy < lowest_expected_free_energy:
                lowest_expected_free_energy = expected_free_energy
                best_action = action
                
        return best_action

if __name__ == "__main__":
    print("--- 21. Active Inference Agent ---")
    
    state_dim = 4
    obs_dim = 10
    agent = ActiveInferenceAgent(state_dim, obs_dim)
    
    # 1. The Real World
    # An unexpected event happens (e.g. A predator appears)
    true_sensory_input = torch.randn(1, obs_dim) * 5.0 
    
    print("Agent encounters highly unexpected sensory input.")
    
    # 2. Perception (Updating Belief)
    # The agent loops to update its belief to understand the input
    print("\n--- Perception Phase (Minimizing Surprise internally) ---")
    for step in range(5):
        fe = agent.perception_step(true_sensory_input)
        print(f"  Thought cycle {step+1}: Free Energy (Surprise) = {fe:.4f}")
        
    print("\nBelief updated. Agent now 'understands' the situation.")
    
    # 3. Action (Changing the World)
    # The agent has 3 possible physical actions it can take.
    # It evaluates them not for 'Points', but to minimize future surprise.
    possible_actions = [
        torch.randn(1, obs_dim) * 2.0, # Action A (e.g., Run)
        torch.randn(1, obs_dim) * 0.1, # Action B (e.g., Hide)
        torch.randn(1, obs_dim) * -5.0 # Action C (e.g., Attack)
    ]
    
    print("\n--- Action Phase (Minimizing Surprise externally) ---")
    best_action = agent.action_step(true_sensory_input, possible_actions)
    print(f"Action selected that results in the lowest Expected Free Energy.")
    print("Success: Active Inference perception-action loop complete.")

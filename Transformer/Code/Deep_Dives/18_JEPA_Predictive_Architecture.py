"""
18. Joint Embedding Predictive Architecture (JEPA)
Deep Dive Implementation

This file demonstrates the core components of Yann LeCun's JEPA.
It highlights the asymmetric architecture: 
- The x-encoder and predictor are trained via backpropagation.
- The y-encoder (the target) is NOT trained. Its weights are updated via 
  an Exponential Moving Average (EMA) of the x-encoder to prevent collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Encoder(nn.Module):
    """A standard encoder (could be a ViT or CNN in reality)."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    """Predicts the future abstract state based on current state + action variable (z)"""
    def __init__(self, embed_dim, z_dim):
        super().__init__()
        # Takes the current state embedding AND the action variable z
        self.net = nn.Sequential(
            nn.Linear(embed_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, sx, z):
        combined = torch.cat([sx, z], dim=-1)
        return self.net(combined)

class JEPA(nn.Module):
    def __init__(self, input_dim=784, embed_dim=64, z_dim=16, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau
        
        # 1. The Context Encoder (Trains via Backprop)
        self.x_encoder = Encoder(input_dim, embed_dim)
        
        # 2. The Target Encoder (Does NOT train via Backprop)
        self.y_encoder = copy.deepcopy(self.x_encoder)
        for param in self.y_encoder.parameters():
            param.requires_grad = False
            
        # 3. The Predictor (Trains via Backprop)
        self.predictor = Predictor(embed_dim, z_dim)

    def update_target_encoder(self):
        """
        Momentum update for the y-encoder:
        theta_y = tau * theta_y + (1 - tau) * theta_x
        This prevents Information Collapse.
        """
        with torch.no_grad():
            for param_x, param_y in zip(self.x_encoder.parameters(), self.y_encoder.parameters()):
                param_y.data.mul_(self.ema_tau).add_(param_x.data, alpha=(1.0 - self.ema_tau))

    def forward(self, x_context, y_target, z_action):
        """
        x_context: e.g., Frames 1-5 of a video
        y_target: e.g., Frame 6 of a video
        z_action: e.g., The physical action taken between frames
        """
        # 1. Encode context
        s_x = self.x_encoder(x_context)
        
        # 2. Predict future state
        s_y_pred = self.predictor(s_x, z_action)
        
        # 3. Encode actual target state (NO GRADIENTS)
        with torch.no_grad():
            s_y_actual = self.y_encoder(y_target)
            
        return s_y_pred, s_y_actual

if __name__ == "__main__":
    print("--- 18. Joint Embedding Predictive Architecture (JEPA) ---")
    
    batch_size = 4
    input_dim = 128
    embed_dim = 32
    z_dim = 8
    
    jepa = JEPA(input_dim, embed_dim, z_dim)
    
    # Dummy data
    x_context = torch.randn(batch_size, input_dim) # "The dog is running"
    z_action = torch.randn(batch_size, z_dim)      # Action: "Jumps"
    y_target = torch.randn(batch_size, input_dim)  # Actual future: "Dog in mid-air"
    
    # Forward Pass
    print("Encoding context and predicting abstract future state...")
    s_y_pred, s_y_actual = jepa(x_context, y_target, z_action)
    
    # Calculate Energy Loss (L2 Distance in embedding space)
    # Notice we are NOT predicting pixels (input_dim), we are predicting embeddings (embed_dim)
    loss = F.mse_loss(s_y_pred, s_y_actual)
    
    print(f"Predicted State Shape: {s_y_pred.shape}")
    print(f"Target State Shape: {s_y_actual.shape}")
    print(f"Energy Loss: {loss.item():.4f}")
    
    print("\nUpdating y-encoder weights via Exponential Moving Average (EMA)...")
    jepa.update_target_encoder()
    print("Success: JEPA forward pass and EMA collapse prevention complete.")

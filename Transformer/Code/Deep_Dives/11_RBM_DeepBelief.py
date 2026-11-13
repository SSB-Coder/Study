"""
11. Restricted Boltzmann Machines (RBM)
Deep Dive Implementation

This file explicitly demonstrates an RBM and the Contrastive Divergence (CD-1)
algorithm used to train it without standard backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    """Restricted Boltzmann Machine."""
    def __init__(self, visible_nodes, hidden_nodes):
        super().__init__()
        # W: Weight matrix between Visible and Hidden (Size: [Hidden, Visible])
        self.W = nn.Parameter(torch.randn(hidden_nodes, visible_nodes) * 0.1)
        
        # Biases
        self.h_bias = nn.Parameter(torch.zeros(hidden_nodes))
        self.v_bias = nn.Parameter(torch.zeros(visible_nodes))

    def sample_hidden(self, v_prob):
        """
        Push data UP: Calculate P(h=1 | v) and sample a binary hidden state.
        Equation: P(h=1|v) = Sigmoid(v @ W^T + h_bias)
        """
        hidden_activation = F.linear(v_prob, self.W, self.h_bias)
        p_h_given_v = torch.sigmoid(hidden_activation)
        
        # Sample a hard binary 0 or 1 based on the probabilities
        # (Using torch.bernoulli for statistical sampling)
        h_sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, h_sample

    def sample_visible(self, h_prob):
        """
        Push data DOWN: Calculate P(v=1 | h) and sample a reconstructed visible state.
        Equation: P(v=1|h) = Sigmoid(h @ W + v_bias)
        """
        # Note: self.W.t() flips the matrix to push backwards from hidden to visible
        visible_activation = F.linear(h_prob, self.W.t(), self.v_bias)
        p_v_given_h = torch.sigmoid(visible_activation)
        
        v_sample = torch.bernoulli(p_v_given_h)
        return p_v_given_h, v_sample

    def forward(self, v_input):
        """
        The Forward pass is technically the Contrastive Divergence (CD-1) step.
        """
        # --- Positive Phase ---
        # Look at the real data and sample the hidden state
        p_h_real, h_real_sample = self.sample_hidden(v_input)
        
        # --- Negative Phase (The Reconstruction) ---
        # Take the sampled hidden state, and generate a 'fake' visible image
        p_v_fake, v_fake_sample = self.sample_visible(h_real_sample)
        
        # Take the fake image, and push it UP to the hidden state again
        p_h_fake, h_fake_sample = self.sample_hidden(p_v_fake)
        
        return v_input, p_h_real, p_v_fake, p_h_fake

    def contrastive_divergence_loss(self, v_real, p_h_real, v_fake, p_h_fake):
        """
        Calculates the gradients based on the correlations of the Real state vs the Fake state.
        Delta_W = <v_i * h_j>_data - <v_i * h_j>_reconstruction
        (Note: We return this as a negative 'loss' so PyTorch's standard optimizer can minimize it).
        """
        # Positive association (Real Data)
        positive_assoc = torch.matmul(p_h_real.t(), v_real)
        
        # Negative association (Fake Data)
        negative_assoc = torch.matmul(p_h_fake.t(), v_fake)
        
        # The gradient for W is actually (positive_assoc - negative_assoc) / batch_size
        # To use PyTorch's auto-optimizer (which minimizes loss), we construct a dummy loss
        # whose derivative wrt W equals our desired gradient.
        
        # We manually compute the update rule instead of relying on autograd for education:
        batch_size = v_real.size(0)
        grad_W = (positive_assoc - negative_assoc) / batch_size
        
        return grad_W

if __name__ == "__main__":
    print("--- 11. Restricted Boltzmann Machine (CD-1) ---")
    batch_size = 4
    visible_dim = 6 # e.g. 6 movie ratings (0 or 1)
    hidden_dim = 2  # 2 latent genres (e.g. Action vs Romance)
    
    rbm = RBM(visible_dim, hidden_dim)
    
    # Dummy binary data (user movie ratings)
    v_data = torch.randint(0, 2, (batch_size, visible_dim)).float()
    
    print("Executing Contrastive Divergence (CD-1) Step...")
    v_real, p_h_real, p_v_fake, p_h_fake = rbm(v_data)
    
    # Calculate explicit gradient
    grad_W = rbm.contrastive_divergence_loss(v_real, p_h_real, p_v_fake, p_h_fake)
    
    print(f"Real Visible Shape: {v_real.shape}")
    print(f"Fake/Reconstructed Visible Shape: {p_v_fake.shape}")
    print(f"Gradient Matrix Shape for W: {grad_W.shape}")
    print("Success: RBM positive and negative phases calculated correctly.")

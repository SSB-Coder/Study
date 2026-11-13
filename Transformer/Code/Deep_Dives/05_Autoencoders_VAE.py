"""
05. Variational Autoencoders (VAE)
Deep Dive Implementation

This file demonstrates the explicit architecture of a VAE.
We show the Encoder splitting into Mu and Sigma, the Reparameterization Trick
to allow backpropagation through randomness, and the Decoder reconstructing the image.
We also include the mathematical ELBO Loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """The Variational Autoencoder."""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # --- The Encoder (The Master Forger's Eyes) ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Instead of outputting one latent vector, the Encoder outputs TWO vectors:
        # 1. The Means (Mu)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # 2. The Log-Variances (LogVar)
        # We use LogVar instead of standard deviation because a neural network output 
        # can be negative, and standard deviation must be positive. We exponentiate it later.
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- The Decoder (The Master Forger's Hands) ---
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """The Reparameterization Trick: z = mu + sigma * epsilon"""
        # Convert log variance to standard deviation: sigma = exp(0.5 * log_var)
        std = torch.exp(0.5 * logvar)
        
        # Sample pure random noise (epsilon) from a Standard Normal Distribution N(0, 1)
        # The noise must be exactly the same shape as the standard deviation tensor.
        eps = torch.randn_like(std)
        
        # Calculate z deterministically to allow backpropagation
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Use Sigmoid to push output pixels between 0 and 1
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # 1. Encode
        mu, logvar = self.encode(x.view(-1, 784))
        # 2. Reparameterize (Sample z)
        z = self.reparameterize(mu, logvar)
        # 3. Decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """
    The ELBO (Evidence Lower Bound) Loss = Reconstruction Loss + Regularization (KL Divergence)
    """
    # 1. Reconstruction Loss (Binary Cross Entropy)
    # How well did the decoder rebuild the original image?
    # reduction='sum' sums the loss across the entire batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL Divergence (Regularization)
    # How close are the mu and logvar distributions to a Standard Normal N(0, 1)?
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total Loss is the sum
    return BCE + KLD


if __name__ == "__main__":
    print("--- 05. Variational Autoencoder (VAE) ---")
    batch_size = 4
    input_dim = 784 # 28x28 image
    latent_dim = 20 # Compress 784 pixels down to 20 numbers
    
    # Create the network
    vae = VAE(input_dim=input_dim, hidden_dim=400, latent_dim=latent_dim)
    
    # Dummy data
    image_batch = torch.rand(batch_size, input_dim) # Random pixel values between 0 and 1
    
    # Forward Pass
    recon_batch, mu, logvar = vae(image_batch)
    
    # Calculate Loss
    loss = vae_loss_function(recon_batch, image_batch, mu, logvar)
    
    print(f"Input Shape: {image_batch.shape}")
    print(f"Latent Space Shape: [Batch, Latent_Dim] -> [{batch_size}, {latent_dim}]")
    print(f"Reconstructed Shape: {recon_batch.shape}")
    print(f"Calculated Total ELBO Loss: {loss.item():.4f}")
    print("Success: VAE forward pass and KL Divergence calculated correctly.")

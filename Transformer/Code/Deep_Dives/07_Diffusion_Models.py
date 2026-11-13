"""
07. Diffusion Models (Stable Diffusion / DDPM)
Deep Dive Implementation

This file demonstrates the core mathematics of a Denoising Diffusion Probabilistic Model (DDPM).
It explicitly shows:
1. The Forward Process (Adding Gaussian Noise deterministically via the variance schedule).
2. The Reverse Process (A mock U-Net predicting the noise that was added).
3. The Simple Mean Squared Error (MSE) Loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The Variance Schedule Math ---
# We define how much noise is added at each step T
T = 1000 # Total timesteps
beta_start = 1e-4
beta_end = 0.02
# Linear variance schedule from beta_start to beta_end
betas = torch.linspace(beta_start, beta_end, T)

# Calculate alphas
alphas = 1.0 - betas
# Calculate alpha_bar (cumulative product of alphas)
# This is the crucial math that lets us jump to any timestep t instantly!
alphas_cumprod = torch.cumprod(alphas, dim=0)

# We need the square roots for the reparameterization trick: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


def forward_diffusion_process(x_0, t):
    """
    Takes an original image x_0 and instantly jumps to timestep t by adding 
    the exact mathematical amount of Gaussian noise.
    """
    # Sample pure random Gaussian noise (epsilon)
    noise = torch.randn_like(x_0)
    
    # Extract the pre-calculated coefficients for the specific timestep t
    # Shape matching for broadcasting [Batch, Channels, Height, Width]
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    # The Core Diffusion Equation:
    x_t = (sqrt_alpha_bar_t * x_0) + (sqrt_one_minus_alpha_bar_t * noise)
    
    return x_t, noise


class MockUNet(nn.Module):
    """
    A simplified representation of a U-Net.
    In reality, a U-Net has complex downsampling and upsampling convolutions with skip connections.
    For this mathematical demonstration, we use a flattened MLP that acts as the noise predictor.
    """
    def __init__(self, image_size=784, time_dim=32):
        super().__init__()
        # Time embedding layer (tells the network what step of the denoising process we are in)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main network: takes image AND time embedding
        self.net = nn.Sequential(
            nn.Linear(image_size + time_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, image_size) # Outputs the exact predicted noise tensor
        )

    def forward(self, x_t, t):
        # Flatten image
        x_flat = x_t.view(x_t.size(0), -1)
        
        # Embed time
        t_embedded = self.time_mlp(t.float().view(-1, 1))
        
        # Combine image and time
        combined = torch.cat([x_flat, t_embedded], dim=1)
        
        # Predict the noise
        predicted_noise_flat = self.net(combined)
        
        # Reshape back to image format
        predicted_noise = predicted_noise_flat.view(x_t.size())
        return predicted_noise


if __name__ == "__main__":
    print("--- 07. Diffusion Model (Forward and Reverse Math) ---")
    
    batch_size = 4
    channels = 1
    height = 28
    width = 28
    image_size = channels * height * width
    
    # 1. Grab a clean image from the dataset (x_0)
    x_0 = torch.rand(batch_size, channels, height, width) # Dummy real image
    
    # 2. Pick a random timestep t for each image in the batch
    # e.g., t = [500, 12, 850, 999]
    t = torch.randint(0, T, (batch_size,))
    
    # 3. Forward Process: Corrupt the image with noise up to timestep t
    print("Executing Forward Diffusion (Adding mathematical noise)...")
    x_t, actual_noise_added = forward_diffusion_process(x_0, t)
    
    # 4. Reverse Process: The Neural Network predicts the noise
    print("Executing Reverse U-Net (Predicting the noise)...")
    unet = MockUNet(image_size=image_size)
    predicted_noise = unet(x_t, t)
    
    # 5. The Loss Function (L_simple)
    # The loss is literally just the Mean Squared Error between the noise we ACTUALLY added,
    # and the noise the network PREDICTED we added.
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_noise, actual_noise_added)
    
    print(f"Original Image (x_0) Shape: {x_0.shape}")
    print(f"Noisy Image (x_t) Shape: {x_t.shape}")
    print(f"Predicted Noise Shape: {predicted_noise.shape}")
    print(f"MSE Loss between Actual and Predicted Noise: {loss.item():.4f}")
    print("Success: Diffusion mathematical framework complete.")

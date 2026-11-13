"""
06. Generative Adversarial Networks (GANs)
Deep Dive Implementation

This file explicitly defines the two competing networks of a GAN:
The Generator (Art Forger) and the Discriminator (Detective).
Crucially, it also demonstrates the Minimax Game objective in a mock training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """The Art Forger: Takes random noise and scales it up to a fake image."""
    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2), # Standard for GANs to prevent dead neurons
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh() # Tanh pushes pixel values between -1 and 1
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """The Detective: Takes an image and predicts Real (1) or Fake (0)."""
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            # Dropout helps prevent the Discriminator from memorizing the data
            nn.Dropout(0.3), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output a single probability between 0 and 1
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("--- 06. Generative Adversarial Network (GAN) ---")
    
    batch_size = 8
    noise_dim = 100
    image_dim = 784 # 28x28 flattened
    
    # Initialize the competing networks
    G = Generator(noise_dim, image_dim)
    D = Discriminator(image_dim)
    
    # Binary Cross Entropy Loss measures how close the probability is to the true label (1 or 0)
    criterion = nn.BCELoss()
    
    # --- MOCK TRAINING STEP ---
    print("\nStarting Training Step...")
    
    # 1. Grab a batch of "REAL" images from the dataset
    real_images = torch.randn(batch_size, image_dim) # Dummy real data
    # Labels for real images are all 1s
    real_labels = torch.ones(batch_size, 1)
    
    # 2. Generate a batch of "FAKE" images
    noise = torch.randn(batch_size, noise_dim)
    fake_images = G(noise)
    # Labels for fake images are all 0s
    fake_labels = torch.zeros(batch_size, 1)
    
    print("--- Training the Discriminator (The Detective) ---")
    # The Detective looks at real images
    predictions_on_real = D(real_images)
    d_loss_real = criterion(predictions_on_real, real_labels) # Minimize -log(D(x))
    
    # The Detective looks at fake images (detach prevents backprop into Generator)
    predictions_on_fake = D(fake_images.detach()) 
    d_loss_fake = criterion(predictions_on_fake, fake_labels) # Minimize -log(1 - D(G(z)))
    
    d_total_loss = d_loss_real + d_loss_fake
    print(f"Discriminator Total Loss: {d_total_loss.item():.4f}")
    
    
    print("\n--- Training the Generator (The Art Forger) ---")
    # The Forger's goal is to fool the Detective into guessing "1" (Real) for its fake images.
    # Notice we pass the fake_images back into D, but this time we compare it to REAL labels!
    predictions_on_fake_again = D(fake_images)
    
    # Maximize log(D(G(z))) which is equivalent to minimizing BCELoss with real_labels
    g_loss = criterion(predictions_on_fake_again, real_labels) 
    
    print(f"Generator Loss: {g_loss.item():.4f}")
    
    print("\nSuccess: GAN Minimax objective and forward passes calculated correctly.")

"""
46. Fourier Neural Operators (FNO)
Deep Dive Implementation: 2D Spectral Convolution for Navier-Stokes & Fluid Dynamics

This file explicitly demonstrates 2D Fourier Neural Operators for continuous PDE surrogate modeling.
We implement the Spectral Convolution Layer:
    (K v)(x) = IFFT2( R_theta * FFT2(v) )
with complex-valued frequency weights and demonstrate zero-shot mesh invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Convolution Layer.
    Computes global spatial convolution in O(N log N) by multiplying
    truncated low-frequency Fourier modes by complex weight tensors.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to retain along spatial dimension 1
        self.modes2 = modes2 # Number of Fourier modes to retain along spatial dimension 2

        # Scale factor for parameter initialization
        scale = 1.0 / (in_channels * out_channels)
        
        # Learnable complex weights for corner Fourier modes
        # We only keep the low-frequency modes; high frequencies are truncated to zero.
        self.weights1 = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        )
        self.weights2 = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        )

    def compl_mul2d(self, input_modes, weights):
        """
        Complex tensor multiplication along channels and frequency modes.
        input_modes: [Batch, in_channels, modes1, modes2]
        weights:     [in_channels, out_channels, modes1, modes2]
        returns:     [Batch, out_channels, modes1, modes2]
        """
        return torch.einsum("bixy,ioxy->boxy", input_modes, weights)

    def forward(self, x):
        """
        x: [Batch, In_Channels, Grid_H, Grid_W]
        """
        batch_size = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # 1. Real 2D Fast Fourier Transform (Spatial -> Frequency Domain)
        # Output shape: [Batch, In_Channels, H, W // 2 + 1] (complex tensor)
        x_ft = torch.fft.rfft2(x)

        # 2. Allocate output tensor in frequency domain
        out_ft = torch.zeros(
            batch_size, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 3. Multiply low-frequency Fourier modes by learnable complex weights
        # Top-left corner modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Bottom-left corner modes
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 4. Inverse 2D Fast Fourier Transform (Frequency -> Spatial Domain)
        # Output shape: [Batch, Out_Channels, H, W]
        x_out = torch.fft.irfft2(out_ft, s=(H, W))
        return x_out


class FNO2d(nn.Module):
    """
    Complete 2D Fourier Neural Operator Architecture.
    Consists of:
      1. Lifting network P (projects coordinate grid + initial velocity -> latent channels)
      2. Stacked Spectral Conv Blocks + local residual 1x1 convolutions
      3. Projection network Q (projects latent channels -> target fluid solution)
    """
    def __init__(self, modes1=12, modes2=12, width=32, in_dim=3, out_dim=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 1. Lifting Layer P
        self.p = nn.Linear(in_dim, self.width)

        # 2. Four Fourier Spectral Blocks
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Local point-wise bypass connections (1x1 Conv equivalent)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # 3. Projection Layer Q
        self.q = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        """
        x: [Batch, H, W, In_Dim] (e.g., initial condition + (x, y) spatial coordinates)
        returns: [Batch, H, W, Out_Dim] (e.g., fluid velocity field at next time step)
        """
        # 1. Lift to higher channel dimension
        x = self.p(x) # [Batch, H, W, Width]
        x = x.permute(0, 3, 1, 2) # [Batch, Width, H, W] for Conv2D / FFT

        # 2. Fourier Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        # Fourier Block 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Fourier Block 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # Fourier Block 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 3. Project down to target output field
        x = x.permute(0, 2, 3, 1) # [Batch, H, W, Width]
        x = self.q(x) # [Batch, H, W, Out_Dim]
        return x


if __name__ == "__main__":
    print("--- 46. 2D Fourier Neural Operator (FNO) ---")
    print("Demonstrating Operator Learning & Zero-Shot Mesh Invariance across resolutions...")

    fno = FNO2d(modes1=8, modes2=8, width=16, in_dim=3, out_dim=1)
    
    # 1. Forward pass at standard training resolution (e.g., 32 x 32 grid)
    batch_size = 2
    h_train, w_train = 32, 32
    # Input channels: [Initial Flow Vorticity, X-Coordinate Grid, Y-Coordinate Grid]
    dummy_input_lowres = torch.randn(batch_size, h_train, w_train, 3)

    out_lowres = fno(dummy_input_lowres)
    print(f"Low-Res Input Grid:  {dummy_input_lowres.shape} [Batch, H=32, W=32, In_Dim=3]")
    print(f"Low-Res Output Field: {out_lowres.shape} [Batch, H=32, W=32, Out_Dim=1]")

    # 2. ZERO-SHOT SUPER-RESOLUTION TEST (e.g., 64 x 64 high-resolution grid)
    # The EXACT SAME trained weights run on a grid with 4x as many points without any interpolation!
    h_eval, w_eval = 64, 64
    dummy_input_highres = torch.randn(batch_size, h_eval, w_eval, 3)

    out_highres = fno(dummy_input_highres)
    print(f"\nZero-Shot High-Res Input:  {dummy_input_highres.shape} [Batch, H=64, W=64, In_Dim=3]")
    print(f"Zero-Shot High-Res Output: {out_highres.shape} [Batch, H=64, W=64, Out_Dim=1]")
    print("Success: FNO evaluated seamlessly across varying spatial resolutions via spectral convolutions.")

"""
15. State Space Models (SSM) & Mamba
Deep Dive Implementation

This file explicitly demonstrates the math of discretizing continuous-time ODEs
using the Zero-Order Hold (ZOH) rule, which is the foundational math of Mamba.
Instead of the complex FFT parallelized convolution (which requires CUDA kernels),
we show the sequential 'Notebook' Recurrent execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitSSMLayer(nn.Module):
    """The Discretized State Space Model Core."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Continuous-Time Parameters
        # A: Evolution Matrix (How memory decays)
        # We initialize it so it's stable (eigenvalues < 0)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        
        # B: Input Matrix (How much input affects memory)
        self.B = nn.Linear(d_model, d_state, bias=False)
        
        # C: Output Matrix (How memory affects prediction)
        self.C = nn.Linear(d_state, d_model, bias=False)
        
        # Delta: The step size for discretization
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.1)

    def discretize(self, x_t):
        """
        Implements Zero-Order Hold (ZOH) Discretization.
        A_bar = exp(Delta * A)
        B_bar = (Delta * A)^-1 * (exp(Delta * A) - I) * Delta * B
        """
        # Exponentiate parameters to keep them positive where required
        A = -torch.exp(self.A_log) # Ensure A is negative for stable decay
        dt = torch.exp(self.log_dt) # Ensure step size is positive
        
        # Because we simplified A to be diagonal (just a vector), the matrix math 
        # simplifies to element-wise operations.
        # A_bar shape: [d_model, d_state]
        A_bar = torch.exp(torch.outer(dt, A))
        
        # For data-dependent B (Mamba's core innovation), B changes per token x_t
        B_t = self.B(x_t) # [Batch, d_state]
        
        # B_bar requires numerical stability if A is close to 0. 
        # For simplicity, we use the approximation: B_bar = Delta * B
        dt_expanded = dt.unsqueeze(0).unsqueeze(2) # [1, d_model, 1]
        B_bar = dt_expanded * B_t.unsqueeze(1)     # [Batch, d_model, d_state]
        
        return A_bar, B_bar

    def forward(self, x_sequence):
        """
        x_sequence: [Batch, Seq_Len, d_model]
        """
        batch_size = x_sequence.size(0)
        seq_len = x_sequence.size(1)
        
        # Initialize the 'Notebook' (Hidden State)
        h_t = torch.zeros(batch_size, self.d_model, self.d_state, device=x_sequence.device)
        
        outputs = []
        
        # Sequential O(N) Processing (The 'Infinite Ledger' method)
        # Note: In real Mamba, this loop is replaced by an ultra-fast hardware-aware Parallel Scan
        for t in range(seq_len):
            x_t = x_sequence[:, t, :]
            
            # Discretize matrices dynamically based on the current input token
            A_bar, B_bar = self.discretize(x_t)
            
            # Recurrent State Update
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            # We add an extra dimension to x_t for broadcasting
            h_t = A_bar.unsqueeze(0) * h_t + B_bar * x_t.unsqueeze(2)
            
            # Output generation
            # y_t = C * h_t
            C_t = self.C.weight.unsqueeze(0).expand(batch_size, -1, -1) # [Batch, d_model, d_state]
            y_t = torch.sum(C_t * h_t, dim=-1)
            
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    print("--- 15. State Space Model (SSM / Mamba math) ---")
    
    batch_size = 2
    seq_length = 100 # Can handle massive sequences without O(N^2) memory blowout!
    d_model = 16
    d_state = 32 # The fixed size of the 'Notebook' memory
    
    # Create the SSM
    ssm = ExplicitSSMLayer(d_model=d_model, d_state=d_state)
    
    dummy_sequence = torch.randn(batch_size, seq_length, d_model)
    
    print(f"Processing sequence of length {seq_length} using O(N) recurrent ledger...")
    output = ssm(dummy_sequence)
    
    print(f"Input Shape: {dummy_sequence.shape} [Batch, SeqLen, EmbedDim]")
    print(f"Internal Memory (h_t) is fixed at size: [Batch, {d_model}, {d_state}]")
    print(f"Output Shape: {output.shape} [Batch, SeqLen, EmbedDim]")
    print("Success: SSM continuous time discretization and recurrent update complete.")

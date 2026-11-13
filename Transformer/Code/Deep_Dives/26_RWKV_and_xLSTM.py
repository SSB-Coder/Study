"""
26. xLSTM (Matrix Memory)
Deep Dive Implementation

This file demonstrates the core concept of the mLSTM block from the xLSTM architecture.
Instead of a standard 1D scalar cell state, it uses a full DxD Matrix for memory,
vastly increasing its storage capacity and retrieval abilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixLSTMCell(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.d = embed_dim
        
        # Linear projections for Input (i), Forget (f), Output (o), Key (k), Value (v)
        self.W_i = nn.Linear(self.d, 1)
        self.W_f = nn.Linear(self.d, 1)
        self.W_o = nn.Linear(self.d, self.d)
        
        self.W_k = nn.Linear(self.d, self.d)
        self.W_v = nn.Linear(self.d, self.d)

    def forward(self, x_t, C_prev, n_prev):
        """
        x_t: Input at time t [Batch, Embed_Dim]
        C_prev: Matrix Memory state at time t-1 [Batch, Embed_Dim, Embed_Dim]
        n_prev: Normalizer state at time t-1 [Batch, Embed_Dim]
        """
        # 1. Calculate Gates
        # The xLSTM paper uses Exponential Gating instead of Sigmoid
        i_t = torch.exp(self.W_i(x_t)) # Input gate
        f_t = torch.exp(self.W_f(x_t)) # Forget gate
        o_t = torch.sigmoid(self.W_o(x_t)) # Output gate (keeps sigmoid for boundedness)
        
        # 2. Calculate Key and Value
        k_t = self.W_k(x_t)
        v_t = self.W_v(x_t)
        
        # 3. Update the MATRIX Memory (The core innovation)
        # We multiply Key and Value (outer product) to create a matrix, then scale by input gate
        # Shape of (v_t.unsqueeze(2) @ k_t.unsqueeze(1)) is [Batch, d, d]
        new_memory_info = i_t.unsqueeze(-1) * torch.matmul(v_t.unsqueeze(2), k_t.unsqueeze(1))
        
        # Scale old memory by forget gate and add new info
        C_t = f_t.unsqueeze(-1) * C_prev + new_memory_info
        
        # 4. Update Normalizer (tracks magnitude of memory additions)
        n_t = f_t * n_prev + i_t * k_t
        
        # 5. Read from Memory (Retrieval)
        # We query the matrix memory using our Key
        h_tilde = torch.matmul(C_t, k_t.unsqueeze(2)).squeeze(2)
        h_tilde_norm = h_tilde / (torch.matmul(n_t.unsqueeze(1), k_t.unsqueeze(2)).squeeze() + 1e-6).unsqueeze(1)
        
        # 6. Apply output gate
        h_t = o_t * h_tilde_norm
        
        return h_t, C_t, n_t

if __name__ == "__main__":
    print("--- 26. xLSTM (Matrix LSTM Cell) ---")
    batch = 2
    embed_dim = 64
    
    # Initialize Memory
    x_t = torch.randn(batch, embed_dim)
    C_prev = torch.zeros(batch, embed_dim, embed_dim) # O(d^2) memory!
    n_prev = torch.zeros(batch, embed_dim)
    
    cell = MatrixLSTMCell(embed_dim)
    
    print("Processing time step 't' with Matrix Memory...")
    h_t, C_t, n_t = cell(x_t, C_prev, n_prev)
    
    print(f"Hidden State Output Shape: {h_t.shape}")
    print(f"Matrix Memory Shape: {C_t.shape} <- Note this is [Batch, {embed_dim}, {embed_dim}]")
    print("Success: Updated the O(d^2) matrix memory cell.")

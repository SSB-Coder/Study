"""
25. Linear Attention and GLA (Simplified)
Deep Dive Implementation

This script demonstrates the fundamental associative trick of Linear Attention:
By removing the non-linear softmax, we can compute (K^T @ V) first, 
reducing complexity from O(N^2) to O(N).
"""

import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, embed_dim, feature_map=nn.ELU()):
        super().__init__()
        self.feature_map = feature_map # Phi(x), replacing softmax
        
    def forward(self, Q, K, V):
        """
        Q, K, V shapes: [Batch, Seq_Len, Embed_Dim]
        """
        # Apply feature map to make values positive and non-linear
        # We add 1.0 to ELU to ensure positivity
        Q_prime = self.feature_map(Q) + 1.0
        K_prime = self.feature_map(K) + 1.0
        
        # --- THE O(N^2) STANDARD WAY (For Comparison) ---
        # scores = Q_prime @ K_prime.transpose(-2, -1)  -> Shape: [Batch, Seq_Len, Seq_Len]
        # out = scores @ V
        
        # --- THE O(N) LINEAR WAY ---
        # 1. Multiply K^T @ V first! 
        # K_prime^T shape: [Batch, Embed_Dim, Seq_Len]
        # V shape:         [Batch, Seq_Len, Embed_Dim]
        # KV_state shape:  [Batch, Embed_Dim, Embed_Dim] (Completely independent of Seq_Len N!)
        KV_state = torch.matmul(K_prime.transpose(-2, -1), V)
        
        # 2. Multiply Q @ KV_state
        # Q_prime shape: [Batch, Seq_Len, Embed_Dim]
        # Result shape:  [Batch, Seq_Len, Embed_Dim]
        out = torch.matmul(Q_prime, KV_state)
        
        # Normalize (simplified)
        Z = 1.0 / (torch.matmul(Q_prime, K_prime.sum(dim=1, keepdim=True).transpose(-2, -1)) + 1e-6)
        return out * Z

if __name__ == "__main__":
    print("--- 25. Linear Attention (O(N)) ---")
    batch = 2
    seq_len = 10000 # Notice how large this is!
    embed_dim = 64
    
    Q = torch.randn(batch, seq_len, embed_dim)
    K = torch.randn(batch, seq_len, embed_dim)
    V = torch.randn(batch, seq_len, embed_dim)
    
    linear_attn = LinearAttention(embed_dim)
    
    print(f"Input Sequence Length: {seq_len}")
    print("Calculating Linear Attention...")
    output = linear_attn(Q, K, V)
    
    print(f"Output Shape: {output.shape}")
    print("Success: Calculated attention without ever building the massive N x N score matrix!")

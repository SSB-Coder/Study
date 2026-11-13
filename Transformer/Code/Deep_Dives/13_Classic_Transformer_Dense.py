"""
13. Classic Dense Transformers (Attention is All You Need)
Deep Dive Implementation

This file explicitly demonstrates Scaled Dot-Product Attention.
Instead of using `nn.MultiheadAttention`, we manually compute Q, K, V
and the exact dot-product attention matrix: Softmax((Q * K^T) / sqrt(d)) * V.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExplicitSelfAttention(nn.Module):
    """The cocktail party listener: looking at every word simultaneously."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # W_Q, W_K, W_V matrices
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x: Input Sequence [Batch, Sequence_Length, Embed_Dim]
        mask: Optional mask to prevent looking at future words
        """
        # 1. Project input into Queries, Keys, and Values
        Q = self.W_q(x) # [Batch, Seq_Len, Embed_Dim]
        K = self.W_k(x) # [Batch, Seq_Len, Embed_Dim]
        V = self.W_v(x) # [Batch, Seq_Len, Embed_Dim]
        
        # 2. Scaled Dot-Product
        # Q @ K^T. We transpose only the last two dimensions of K.
        # This computes the exact relevance score between every pair of words.
        # Shape: [Batch, Seq_Len, Seq_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale by sqrt(d_k) to prevent exploding gradients in the Softmax
        scores = scores / math.sqrt(self.embed_dim)
        
        # Apply mask (e.g., for autoregressive generation)
        if mask is not None:
            # Set masked positions to -Infinity so Softmax forces them to 0
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax to get probabilities (attention weights)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply probabilities by the Values
        # Shape: [Batch, Seq_Len, Seq_Len] @ [Batch, Seq_Len, Embed_Dim] 
        # -> [Batch, Seq_Len, Embed_Dim]
        context_vector = torch.matmul(attention_weights, V)
        
        return context_vector, attention_weights


class TransformerBlock(nn.Module):
    """A complete Transformer layer with Residuals and LayerNorm."""
    def __init__(self, embed_dim, mlp_hidden_dim):
        super().__init__()
        self.attention = ExplicitSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # The Feed-Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. Attention with Residual Connection (x + Attention(x))
        # Note: Modern architectures do LayerNorm BEFORE attention (Pre-Norm)
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x)
        x = x + attn_out # Residual connection
        
        # 2. MLP with Residual Connection
        norm_x2 = self.norm2(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out # Residual connection
        
        return x


if __name__ == "__main__":
    print("--- 13. Classic Dense Transformer ---")
    
    batch_size = 2
    seq_length = 5
    embed_dim = 16
    
    # Create the block
    block = TransformerBlock(embed_dim=embed_dim, mlp_hidden_dim=64)
    
    # Dummy Sequence Data
    dummy_sequence = torch.randn(batch_size, seq_length, embed_dim)
    
    # Forward Pass
    print("Executing Self-Attention and MLP passes...")
    output = block(dummy_sequence)
    
    print(f"Input Shape: {dummy_sequence.shape} [Batch, SeqLen, EmbedDim]")
    print(f"Output Shape: {output.shape} [Batch, SeqLen, EmbedDim]")
    print("Success: Dense Transformer block executed with accurate tensor dimensions.")

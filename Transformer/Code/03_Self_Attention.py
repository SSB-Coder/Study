import torch
import torch.nn as nn
import math

# ==========================================
# CORE ATTENTION MECHANISM
# The engine behind all modern LLMs.
# ==========================================

class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Self Attention from scratch.
    Equation: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        
        # The dimension of each head
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Explicitly defining the weight matrices for Query, Key, and Value
        # Instead of doing nn.Linear for each head, we do one massive nn.Linear
        # and reshape it later for efficiency.
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        
        # Final linear layer to mix the heads back together
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0] # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 1. Project the inputs through the weight matrices
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 2. Reshape into Multiple Heads
        # Shape: (Batch_size, sequence_length, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # 3. Calculate Dot Product of Queries and Keys
        # We use einsum for clarity in tensor multiplication.
        # "nqhd,nkhd->nhqk" means: 
        #   Multiply queries (q) and keys (k) along the 'd' (head_dim) dimension.
        #   Resulting shape: (Batch_size, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 4. Scale and Mask
        if mask is not None:
            # If masking (e.g., in a decoder so it can't look at future words),
            # set the masked positions to -infinity so softmax makes them 0.
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Scale by sqrt(d_k) to prevent gradients from vanishing/exploding
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)

        # 5. Multiply by Values
        # "nhqk,nvhd->nqhd"
        # Resulting shape: (Batch_size, query_len, heads, head_dim)
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])

        # 6. Concatenate heads back together
        # Shape becomes: (Batch_size, query_len, embed_size)
        out = out.reshape(N, query_len, self.embed_size)

        # 7. Final linear projection
        out = self.fc_out(out)
        
        return out


if __name__ == "__main__":
    print("Testing Multi-Head Self-Attention Component...")
    embed_size = 512
    heads = 8
    seq_length = 100
    batch_size = 4
    
    # Dummy sequence (e.g., a sentence of 100 words, each word represented by 512 numbers)
    x = torch.randn(batch_size, seq_length, embed_size)
    
    attention_layer = SelfAttention(embed_size, heads)
    
    # In Self-Attention, Q, K, and V all come from the same input sequence 'x'
    out = attention_layer(values=x, keys=x, queries=x, mask=None)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print("Attention mechanism successfully processed the sequence while maintaining dimensions.")

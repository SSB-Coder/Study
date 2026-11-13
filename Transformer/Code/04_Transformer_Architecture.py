import torch
import torch.nn as nn
from Code.03_Self_Attention import SelfAttention

# ==========================================
# FULL TRANSFORMER ARCHITECTURE
# Built block-by-block using the explicit Attention mechanism.
# ==========================================

class TransformerBlock(nn.Module):
    """
    A single block/layer of the Transformer.
    Modern LLMs stack tens or hundreds of these.
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        
        # Layer Normalization normalizes the outputs of the neurons, keeping variance stable
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # The Feed-Forward Network (MLP) part of the block
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # 1. Attention Step
        attention = self.attention(value, key, query, mask)
        
        # 2. Add & Norm (Residual connection + LayerNorm)
        # We add the original query (input) to the attention output
        x = self.dropout(self.norm1(attention + query))
        
        # 3. Feed Forward Step
        forward = self.feed_forward(x)
        
        # 4. Add & Norm (Residual connection + LayerNorm)
        out = self.dropout(self.norm2(forward + x))
        
        return out


class TransformerEncoder(nn.Module):
    """
    The full Encoder model (like BERT).
    """
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        # 1. Word Embedding: Converts token IDs to dense vectors
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        
        # 2. Positional Embedding: Transformers have no sense of order. 
        # We MUST inject position information mathematically.
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack multiple Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x is a batch of token IDs. Shape: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # Create position indices (0, 1, 2, ..., seq_length-1)
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        
        # Combine word and positional embeddings
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        # Pass through all Transformer blocks
        for layer in self.layers:
            # In the encoder, query, key, and value are all the same 'out'
            out = layer(out, out, out, mask)
            
        return out


# Note: A full LLM like GPT is a "Decoder-Only" architecture, which is similar 
# but uses a masked attention mechanism to prevent looking into the future.
# The structure below demonstrates a standard text-generation Head attached to our layers.

class GenerativeTransformerModel(nn.Module):
    """
    A full model combining the architecture into something that can predict the next token.
    """
    def __init__(
        self,
        vocab_size,
        embed_size=256,
        num_layers=6,
        heads=8,
        device="cpu",
        forward_expansion=4,
        dropout=0.1,
        max_length=1000,
    ):
        super(GenerativeTransformerModel, self).__init__()
        
        # The Core Processing Engine
        self.encoder = TransformerEncoder(
            vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )
        
        # The Output Head (maps the final embeddings back to vocab probabilities)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.device = device

    def make_src_mask(self, src):
        # Creates a mask to ignore padding tokens (usually token 0)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Shape: (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, x):
        src_mask = self.make_src_mask(x)
        # Get deep contextual embeddings
        enc_out = self.encoder(x, src_mask)
        # Predict next word probabilities
        out = self.fc_out(enc_out)
        return out


if __name__ == "__main__":
    print("Testing Full Transformer Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy data: Batch of 2 sentences, 10 tokens each
    # Numbers represent word IDs in a vocabulary
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0], 
                      [1, 8, 7, 3, 4, 5, 6, 7, 2, 0]]).to(device)
    
    vocab_size = 10  # 0 to 9
    
    model = GenerativeTransformerModel(
        vocab_size=vocab_size,
        embed_size=128,
        num_layers=4,
        heads=4,
        device=device,
        max_length=100
    ).to(device)
    
    out = model(x)
    print(f"Input Shape (Tokens): {x.shape}")
    print(f"Output Shape (Logits): {out.shape} - (Batch, Seq_Len, Vocab_Size)")
    print("Model successfully built and passed forward pass!")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ADVANCED ARCHITECTURES: MIXTURE OF EXPERTS (MoE)
# This is how models like GPT-4 and DeepSeek scale
# beyond 1 Trillion parameters efficiently.
# ==========================================

class Expert(nn.Module):
    """
    A single 'Expert' is just a standard Feed-Forward Neural Network.
    In a real MoE, there are many of these (e.g., 8, 16, or 64).
    """
    def __init__(self, embed_size, hidden_size):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(), # or SwiGLU in SOTA models
            nn.Linear(hidden_size, embed_size)
        )

    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    """
    The Router decides which tokens go to which experts.
    This is a simple linear classifier over the embeddings.
    """
    def __init__(self, embed_size, num_experts):
        super(Router, self).__init__()
        self.route_weights = nn.Linear(embed_size, num_experts)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        # logits shape: (batch_size, seq_len, num_experts)
        logits = self.route_weights(x)
        return logits

class SparseMoEBlock(nn.Module):
    """
    A Sparse Mixture of Experts block.
    This Replaces the standard Feed-Forward network inside a Transformer block.
    """
    def __init__(self, embed_size, hidden_size, num_experts=8, top_k=2):
        super(SparseMoEBlock, self).__init__()
        self.num_experts = num_experts
        
        # We only route to 'top_k' experts per token (usually 2).
        # This creates the "Sparsity" - we don't compute the other 6 experts!
        self.top_k = top_k 
        
        self.router = Router(embed_size, num_experts)
        self.experts = nn.ModuleList([Expert(embed_size, hidden_size) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape
        
        # 1. Get routing probabilities
        router_logits = self.router(x)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # 2. Find the top K experts for each token
        # routing_weights shape: (batch, seq, top_k)
        # selected_experts shape: (batch, seq, top_k)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Re-normalize weights so they sum to 1 across the top-K
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        final_output = torch.zeros_like(x)
        
        # Flatten everything to process token-by-token
        flat_x = x.view(-1, embed_size)
        flat_selected = selected_experts.view(-1, self.top_k)
        flat_weights = routing_weights.view(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)
        
        # 3. Route tokens to experts
        # In a real distributed system, this requires complex all-to-all communication.
        # Here we do it sequentially for demonstration.
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            # shape: (num_tokens, top_k) boolean mask
            expert_mask = (flat_selected == i)
            
            # If any token is assigned to this expert...
            if expert_mask.any():
                # Extract the token index and which of the 'top_k' choices it was
                token_idx, choice_idx = torch.where(expert_mask)
                
                # Get the actual token embeddings
                expert_inputs = flat_x[token_idx]
                
                # Run the expert network (Computation happens here)
                expert_outputs = expert(expert_inputs)
                
                # Multiply by the router's confidence weight
                expert_weights = flat_weights[token_idx, choice_idx].unsqueeze(1)
                expert_outputs = expert_outputs * expert_weights
                
                # Add the result to the output tensor
                # Because a token can go to multiple experts, we use index_add_
                flat_output.index_add_(0, token_idx, expert_outputs)

        # Reshape back to original dimensions
        final_output = flat_output.view(batch_size, seq_len, embed_size)
        return final_output


if __name__ == "__main__":
    print("Testing Sparse Mixture of Experts (MoE) Architecture...")
    
    # Configuration
    embed_dim = 256
    hidden_dim = 1024
    num_experts = 8 # Total experts (e.g., 8 different "brains")
    top_k = 2       # Only use 2 out of 8 brains per word
    
    # Dummy Input: Batch of 2, 10 tokens per batch
    x = torch.randn(2, 10, embed_dim)
    
    moe_block = SparseMoEBlock(
        embed_size=embed_dim, 
        hidden_size=hidden_dim, 
        num_experts=num_experts, 
        top_k=top_k
    )
    
    out = moe_block(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print("MoE layer successfully routed tokens and combined expert outputs!")

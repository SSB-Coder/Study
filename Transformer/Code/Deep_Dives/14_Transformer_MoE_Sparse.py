"""
14. Sparse Mixture of Experts (MoE) Transformers
Deep Dive Implementation

This file demonstrates the MoE architecture used in Mixtral / DeepSeek.
We explicitly define the Router and implement the Top-K masking logic so that
the model ONLY computes the MLPs for the selected experts, simulating sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Each Expert is a standalone MLP, defined below for full self-containment.

class ExpertMLP(nn.Module):
    """A standard Feed-Forward network acting as a single Professor."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class ExplicitMoELayer(nn.Module):
    """Replaces the dense MLP with a Router and N Experts."""
    def __init__(self, embed_dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # The Router (The Receptionist)
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        
        # The Experts (The Professors) - stored in a ModuleList
        self.experts = nn.ModuleList([ExpertMLP(embed_dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Embed_Dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Flatten sequence for routing: [Batch*SeqLen, Embed_Dim]
        x_flat = x.view(-1, embed_dim)
        
        # 1. The Router Output
        # Get raw scores for each expert: [Batch*SeqLen, Num_Experts]
        router_logits = self.router(x_flat)
        
        # 2. Top-K Selection
        # Get the highest K values and their indices
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Calculate routing probabilities ONLY for the top K experts using Softmax
        routing_weights = F.softmax(topk_values, dim=-1)
        
        # 3. Sparse Expert Computation
        # Initialize an empty tensor to hold the final combined output
        final_output_flat = torch.zeros_like(x_flat)
        
        # Loop over the K selected slots
        for i in range(self.top_k):
            # Which expert was selected for this slot?
            expert_indices = topk_indices[:, i]
            # What is the weight for this expert?
            expert_weights = routing_weights[:, i].unsqueeze(1)
            
            # Since PyTorch doesn't natively support dynamic graph branching easily without 
            # severe overhead in standard loops, we simulate the 'Sparsity' by only 
            # calculating the experts that were actually selected by at least one token.
            unique_experts = torch.unique(expert_indices)
            
            for expert_idx in unique_experts:
                # Find which tokens chose this specific expert
                token_mask = (expert_indices == expert_idx)
                
                if token_mask.any():
                    # Extract only the tokens that go to this expert
                    tokens_for_expert = x_flat[token_mask]
                    
                    # RUN THE EXPERT (This is the compute-saving step! Other experts sleep)
                    expert_out = self.experts[expert_idx](tokens_for_expert)
                    
                    # Multiply by the router's confidence weight and add to final output
                    weights_for_tokens = expert_weights[token_mask]
                    final_output_flat[token_mask] += expert_out * weights_for_tokens

        # Unflatten back to sequence shape
        final_output = final_output_flat.view(batch_size, seq_len, embed_dim)
        return final_output


if __name__ == "__main__":
    print("--- 14. Sparse Mixture of Experts (MoE) Layer ---")
    
    batch_size = 2
    seq_length = 4
    embed_dim = 16
    num_experts = 8
    top_k = 2
    
    # Create the MoE Layer
    moe = ExplicitMoELayer(embed_dim, hidden_dim=64, num_experts=num_experts, top_k=top_k)
    
    dummy_sequence = torch.randn(batch_size, seq_length, embed_dim)
    
    print(f"Routing tokens to the Top {top_k} out of {num_experts} Experts...")
    output = moe(dummy_sequence)
    
    print(f"Input Shape: {dummy_sequence.shape} [Batch, SeqLen, EmbedDim]")
    print(f"Output Shape: {output.shape} [Batch, SeqLen, EmbedDim]")
    print("Success: MoE Top-K Routing and Sparse Computation complete.")

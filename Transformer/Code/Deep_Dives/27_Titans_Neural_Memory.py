"""
27. Titans: Neural Long-Term Memory
Deep Dive Implementation

This file demonstrates the core concept of the Titans Long-Term Memory module.
Unlike a standard KV Cache, the memory is stored in the weights of a neural network layer.
When new tokens arrive, the model performs a gradient descent step ON ITSELF 
to memorize the information (Test-Time Learning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemoryModule(nn.Module):
    def __init__(self, embed_dim, memory_lr=0.01):
        super().__init__()
        self.d = embed_dim
        self.memory_lr = memory_lr # The learning rate used during INFERENCE
        
        # The "Memory" is just a set of weights. 
        # In a real setup, these weights are updated dynamically per sequence.
        self.memory_weights = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        
        # Projections to create the associative Key and Value
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
    def memorize(self, x_current):
        """
        The TEST-TIME update step.
        We update the memory weights so that querying with K will return V.
        """
        k = self.W_k(x_current) # [Batch, d]
        v = self.W_v(x_current) # [Batch, d]
        
        # We want: Memory_Weights @ k \approx v
        # We calculate the error of our current memory
        v_pred = torch.matmul(k, self.memory_weights)
        error = v - v_pred
        
        # We calculate the gradient and update the weights manually (Delta Rule)
        # Weight Update = Learning_Rate * Outer_Product(error, k)
        with torch.no_grad():
            update = self.memory_lr * torch.matmul(error.unsqueeze(2), k.unsqueeze(1))
            
            # In a real batched setup, memory must be maintained per-batch-item, 
            # but for simplicity we average the update across the batch here.
            self.memory_weights.add_(update.mean(dim=0))
            
    def retrieve(self, x_query):
        """
        Retrieve information from the neural memory using a Query.
        """
        q = self.W_q(x_query)
        # Read from the weights!
        retrieved_info = torch.matmul(q, self.memory_weights)
        return retrieved_info

if __name__ == "__main__":
    print("--- 27. Titans (Neural Long-Term Memory) ---")
    batch = 2
    embed_dim = 64
    
    titans_memory = NeuralMemoryModule(embed_dim)
    
    # Simulate a stream of text coming in
    print("Reading Document Chunk 1...")
    chunk_1 = torch.randn(batch, embed_dim)
    
    # 1. Update the memory weights dynamically!
    titans_memory.memorize(chunk_1)
    
    # 2. Later, we want to retrieve that information
    print("Querying the Neural Memory for past information...")
    query = torch.randn(batch, embed_dim)
    retrieved_memory = titans_memory.retrieve(query)
    
    print(f"Retrieved Memory Shape: {retrieved_memory.shape}")
    print("Success: Information was stored in and retrieved from the dynamic weights, bypassing the need for a massive O(N^2) KV Cache!")

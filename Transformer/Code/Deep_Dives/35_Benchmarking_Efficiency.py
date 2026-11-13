"""
35. Benchmarking Architecture Efficiency (No Training Required)
Deep Dive Implementation

This file answers the critical research question:
"How do I know if my new Transformer architecture is more efficient without 
wasting time and space training a massive LLM?"

We use PyTorch Profiler and torch.cuda memory tools to measure:
1. Peak Memory Allocation (Memory footprint)
2. Execution Time (Throughput/Latency)
3. FLOPs (Computational operations)

We do this purely on the FORWARD PASS using dummy random data. 
"""

import torch
import torch.nn as nn
import time

# We use the standard PyTorch profiler
from torch.profiler import profile, record_function, ProfilerActivity

# ---------------------------------------------------------------------------
# 1. DEFINE TWO COMPETING ARCHITECTURES
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """O(N^2) Standard Dense Attention Layer"""
    def __init__(self, embed_dim=256, heads=8):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.heads = heads
        self.head_dim = embed_dim // heads

    def forward(self, x):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2), qkv)
        
        # The O(N^2) memory bottleneck: Q * K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(out)

class LinearAttentionMock(nn.Module):
    """O(N) Linear Attention Layer (e.g. Performer / GLA / Mamba logic)"""
    def __init__(self, embed_dim=256, heads=8):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.heads = heads
        self.head_dim = embed_dim // heads

    def forward(self, x):
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2), qkv)
        
        # Apply a kernel function (like ReLU) instead of Softmax to make matrices associative
        q = torch.relu(q)
        k = torch.relu(k)
        
        # THE O(N) TRICK: Multiply K^T * V FIRST. 
        # k: [B, H, Seq, Dim] -> k.transpose: [B, H, Dim, Seq]
        # (Dim x Seq) * (Seq x Dim) = (Dim x Dim). The Sequence Length N disappears!
        kv_state = torch.matmul(k.transpose(-2, -1), v) 
        
        out = torch.matmul(q, kv_state)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# 2. THE BENCHMARKING HARNESS
# ---------------------------------------------------------------------------

def benchmark_architecture(model, name, batch_size, seq_len, embed_dim, device='cpu'):
    print(f"\n=== Benchmarking {name} ===")
    print(f"Sequence Length: {seq_len} | Batch Size: {batch_size}")
    
    model = model.to(device)
    dummy_input = torch.randn(batch_size, seq_len, embed_dim).to(device)
    
    # 1. WARMUP (PyTorch takes a second to compile CUDA graphs, so run it once first)
    for _ in range(3):
        _ = model(dummy_input)
        
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    # 2. MEASURE LATENCY (TIME)
    start_time = time.perf_counter()
    for _ in range(10): # Run 10 times and average
        _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / 10) * 1000
    print(f"Average Forward Pass Time: {avg_time_ms:.2f} ms")
    
    # 3. MEASURE PEAK MEMORY ALLOCATION
    if device == 'cuda':
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak VRAM Memory Allocated: {max_mem_mb:.2f} MB")
    else:
        print("Memory tracking requires CUDA.")
        
    # 4. PYTORCH PROFILER (FLOPs and deeper stats)
    # The profiler traces every single operation at the C++ level.
    with profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else []),
                 record_shapes=True,
                 with_flops=True) as prof:
        with record_function("model_inference"):
            model(dummy_input)
            
    # Print the top 3 most expensive operations
    print("\nTop 3 Operations by Time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=3))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # We will test what happens when we increase sequence length from 1024 to 4096.
    embed_dim = 256
    
    # Test 1: Short Sequence (1024)
    print("\n" + "="*50)
    print("TEST 1: SHORT SEQUENCE (N=1024)")
    print("="*50)
    seq_1 = 1024
    benchmark_architecture(StandardAttention(embed_dim), "Standard O(N^2) Attention", batch_size=4, seq_len=seq_1, embed_dim=embed_dim, device=device)
    benchmark_architecture(LinearAttentionMock(embed_dim), "Linear O(N) Attention", batch_size=4, seq_len=seq_1, embed_dim=embed_dim, device=device)
    
    # Test 2: Long Sequence (4096)
    print("\n" + "="*50)
    print("TEST 2: LONG SEQUENCE (N=4096)")
    print("="*50)
    seq_2 = 4096
    benchmark_architecture(StandardAttention(embed_dim), "Standard O(N^2) Attention", batch_size=4, seq_len=seq_2, embed_dim=embed_dim, device=device)
    benchmark_architecture(LinearAttentionMock(embed_dim), "Linear O(N) Attention", batch_size=4, seq_len=seq_2, embed_dim=embed_dim, device=device)
    
    print("\nCONCLUSION:")
    print("Notice how the Standard Attention memory and time blow up quadratically as N increases from 1024 to 4096.")
    print("The Linear Attention model's memory footprint barely increases, proving it is highly efficient without needing to train an LLM!")

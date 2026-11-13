"""
32. Distributed Training Harness (DeepSpeed ZeRO-3 & NCCL)
Deep Dive Implementation

This file demonstrates how to set up a Distributed Data Parallel (DDP) environment
and configure DeepSpeed ZeRO-3 to shard model weights, gradients, and optimizer 
states across multiple GPUs using the NCCL backend.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

# We use a try-except block since deepspeed is not installed by default on all machines.
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("Warning: DeepSpeed not installed. Run 'pip install deepspeed' to execute on real GPUs.")

# ---------------------------------------------------------------------------
# 1. A MASSIVE MOCK MODEL
# ---------------------------------------------------------------------------

class GiantLLM(nn.Module):
    """A dummy model large enough to cause an Out-Of-Memory (OOM) error on a single GPU."""
    def __init__(self, vocab_size=50000, embed_dim=4096, hidden_dim=16384):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # In reality, there would be 96 of these layers
        self.huge_linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.huge_linear_2 = nn.Linear(hidden_dim, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.huge_linear_1(x))
        x = self.huge_linear_2(x)
        return self.lm_head(x)

# ---------------------------------------------------------------------------
# 2. THE ZERO-3 CONFIGURATION JSON
# ---------------------------------------------------------------------------
# ZeRO-3 shards EVERYTHING (Weights, Gradients, Optimizer States) across all GPUs.
# Each GPU only holds a tiny fraction of the model. When a GPU needs a weight for
# the forward pass, it requests it from the other GPUs via NCCL (all-gather).

ds_config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": True, # Mixed precision training is mandatory for large models
        "loss_scale": 0,
        "loss_scale_window": 1000
    },
    "zero_optimization": {
        "stage": 3, # 1: Optimizer states, 2: +Gradients, 3: +Weights
        "offload_optimizer": {
            "device": "cpu", # Offload optimizer states to System RAM
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu", # Offload weights to System RAM when not in use
            "pin_memory": True
        },
        "overlap_comm": True, # Overlap NCCL network communication with GPU compute
        "contiguous_gradients": True
    },
    "steps_per_print": 10
}

# ---------------------------------------------------------------------------
# 3. DISTRIBUTED TRAINING SCRIPT
# ---------------------------------------------------------------------------

def main():
    if not HAS_DEEPSPEED:
        print("\n[SIMULATION MODE] - Explaining the distributed process...")
        print("1. Process Group Initialization: NCCL establishes peer-to-peer PCIe/Infiniband connections.")
        print("2. Sharding: DeepSpeed ZeRO-3 takes the GiantLLM and slices its tensors.")
        print("   If you have 8 GPUs, each GPU holds only 1/8th of the parameters.")
        print("3. Forward Pass: When GPU 0 reaches 'huge_linear_1', it triggers an NCCL All-Gather.")
        print("   GPUs 1-7 send their pieces of the weight to GPU 0 over the network.")
        print("4. Backward Pass: Gradients are computed, then immediately scattered (Reduce-Scatter) back across the GPUs.")
        print("Result: You can train a 70B parameter model that requires 140GB of RAM across eight 24GB GPUs.")
        return

    # 1. Initialize Distributed Process Group (NCCL Backend)
    # DeepSpeed handles the RANK and WORLD_SIZE environment variables automatically
    deepspeed.init_distributed(dist_backend='nccl')
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {local_rank}] Initializing Model...")
    
    # 2. Instantiate the Model (On CPU first, DeepSpeed will shard it and move to GPU)
    # With ZeRO-3, we wrap this in deepspeed.zero.Init() in real applications so the 
    # full model is never materialized in memory at once.
    model = GiantLLM()
    
    # 3. Initialize DeepSpeed Engine
    # This replaces the standard PyTorch optimizer, DDP wrapper, and mixed-precision scaler.
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    print(f"[Rank {local_rank}] DeepSpeed Engine Initialized. Model is now sharded.")

    # 4. Dummy Training Loop
    dummy_input = torch.randint(0, 50000, (4, 128)).cuda(local_rank)
    dummy_target = torch.randint(0, 50000, (4, 128)).cuda(local_rank)
    criterion = nn.CrossEntropyLoss()

    for step in range(3):
        # Forward pass (Triggers NCCL All-Gather to reconstruct weights on the fly)
        outputs = model_engine(dummy_input)
        
        loss = criterion(outputs.view(-1, 50000), dummy_target.view(-1))
        
        # Backward pass (Triggers NCCL Reduce-Scatter to average and distribute gradients)
        model_engine.backward(loss)
        
        # Optimizer step
        model_engine.step()
        
        if local_rank == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    print("--- 32. Distributed Training (ZeRO-3 & NCCL) ---")
    # Normally run via: deepspeed 32_DeepSpeed_ZeRO3_NCCL.py
    main()

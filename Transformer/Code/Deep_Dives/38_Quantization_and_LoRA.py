"""
38. Quantization and LoRA (Low-Rank Adaptation)
Deep Dive Implementation

This file contains the explicit mathematics behind the two most important 
deployment tools in modern AI:
1. Quantization: Compressing a 32-bit/16-bit weight matrix into 8-bit/4-bit.
2. LoRA: Fine-tuning a massive model by freezing the original weights 
   and only training two tiny, low-rank matrices (A and B).
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. QUANTIZATION (FP32 -> INT8)
# ---------------------------------------------------------------------------
# To run a 70B model on consumer hardware, you must compress it. 
# Quantization maps continuous float numbers into discrete integer bins.

def quantize_tensor_int8(tensor):
    """
    Symmetric Quantization (FP32 -> INT8).
    Formula: x_q = round(x / scale)
    where scale = max(abs(tensor)) / 127
    """
    # 1. Find the maximum absolute value in the tensor
    max_val = torch.max(torch.abs(tensor))
    
    # 2. Calculate the scale. INT8 goes from -127 to +127.
    # We want max_val to map exactly to 127.
    scale = max_val / 127.0
    
    # 3. Divide by scale and round to nearest integer
    quantized_tensor = torch.round(tensor / scale)
    
    # 4. Cast to 8-bit integer format
    quantized_tensor = quantized_tensor.to(torch.int8)
    
    return quantized_tensor, scale

def dequantize_tensor_int8(quantized_tensor, scale):
    """
    Formula: x = x_q * scale
    """
    # Convert back to float for the math
    float_tensor = quantized_tensor.to(torch.float32)
    return float_tensor * scale

# ---------------------------------------------------------------------------
# 2. LoRA (Low-Rank Adaptation)
# ---------------------------------------------------------------------------
# Instead of updating a massive 10,000 x 10,000 weight matrix during fine-tuning 
# (which requires too much RAM for the gradients and Adam optimizer), 
# LoRA freezes the big matrix (W) and adds a bypass: W + (A @ B)
# where A is 10,000 x 4, and B is 4 x 10,000. 

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16.0):
        super().__init__()
        
        # 1. The Original Pre-trained Weights (Frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False # FREEZE THE BIG MATRIX!
        
        # 2. The LoRA Matrices (Trainable)
        # B matrix maps: in_features -> rank
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        
        # A matrix maps: rank -> out_features
        self.lora_A = nn.Parameter(torch.randn(out_features, rank))
        
        # 3. Scaling Factor
        # This determines how strongly the LoRA weights impact the base model
        self.scaling = alpha / rank
        
    def forward(self, x):
        # 1. Standard calculation through the frozen base weights: y = x @ W^T
        base_output = torch.nn.functional.linear(x, self.weight)
        
        # 2. LoRA calculation: x @ B^T @ A^T
        # B projects the massive input down to a tiny dimension (e.g. 4)
        # A projects the tiny dimension back up to the massive output dimension
        lora_output = torch.nn.functional.linear(x, self.lora_B)
        lora_output = torch.nn.functional.linear(lora_output, self.lora_A)
        
        # 3. Add them together and scale
        return base_output + (lora_output * self.scaling)


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Testing Quantization ---")
    original_weights = torch.randn(5, 5) * 10  # Random floats between ~ -30 and 30
    print("Original Weights (FP32):\n", original_weights)
    
    quantized_weights, scale = quantize_tensor_int8(original_weights)
    print("\nQuantized Weights (INT8):\n", quantized_weights)
    print("Scale factor:", scale.item())
    
    recovered_weights = dequantize_tensor_int8(quantized_weights, scale)
    print("\nRecovered Weights (FP32):\n", recovered_weights)
    
    error = torch.mean(torch.abs(original_weights - recovered_weights))
    print(f"Quantization Error (Loss of precision): {error.item():.4f}")
    
    print("\n--- 2. Testing LoRA ---")
    # Massive layer: 10,000 x 10,000 parameters = 100,000,000 floats
    lora_layer = LoRALinearLayer(in_features=10000, out_features=10000, rank=4)
    
    # Calculate trainable vs frozen parameters
    frozen_params = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    
    print(f"Frozen Base Parameters: {frozen_params:,}")
    print(f"Trainable LoRA Parameters: {trainable_params:,}")
    print(f"Percentage of trainable weights: {(trainable_params/frozen_params)*100:.4f}%")
    
    # Forward pass
    dummy_input = torch.randn(1, 10000)
    output = lora_layer(dummy_input)
    print(f"\nForward pass successful. Output shape: {output.shape}")

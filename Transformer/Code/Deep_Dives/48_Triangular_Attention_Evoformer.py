"""
48. The Evoformer (AlphaFold 2/3 Architecture)
Deep Dive Implementation: Triangular Multiplicative Update & Pairwise Geometric Reasoning

This file explicitly demonstrates the core mathematical blocks of the Evoformer:
1. Outer Product Mean: Bridging evolutionary MSA mutations to 2D geometric pair maps.
2. Triangular Multiplicative Update (Outgoing edges): Enforcing 3D spatial triangle inequality:
   Z_ij = sum_k (a_ik * b_jk)
3. Triangular Attention: Reasoning around geometric triangles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OuterProductMean(nn.Module):
    """
    Transforms evolutionary co-variation across species into pairwise 2D distance maps.
    Input:  MSA Representation M [Batch, Num_Sequences, Num_Residues, d_m]
    Output: Pair Update ΔZ        [Batch, Num_Residues, Num_Residues, d_z]
    """
    def __init__(self, d_m=32, d_z=64, intermediate_dim=16):
        super().__init__()
        self.d_m = d_m
        self.intermediate_dim = intermediate_dim
        
        # Project residue embeddings down before outer product to save memory
        self.proj_a = nn.Linear(d_m, intermediate_dim)
        self.proj_b = nn.Linear(d_m, intermediate_dim)
        
        # Final linear projection from flattened outer product into pair dimension d_z
        self.out_proj = nn.Linear(intermediate_dim * intermediate_dim, d_z)

    def forward(self, msa):
        # msa: [B, S, R, d_m]
        a = self.proj_a(msa) # [B, S, R, c]
        b = self.proj_b(msa) # [B, S, R, c]

        # Compute outer product for each sequence s, then average across all species S
        # einsum: b s i c, b s j d -> b i j c d
        outer = torch.einsum("bsic,bsjd->bijcd", a, b) # [B, R, R, c, c]
        B, R, _, c, _ = outer.shape
        outer = outer.reshape(B, R, R, c * c) # [B, R, R, c^2]

        pair_update = self.out_proj(outer) # [B, R, R, d_z]
        return pair_update


class TriangularMultiplicativeUpdateOutgoing(nn.Module):
    """
    Updates edge (i, j) by summing information from all intermediate nodes k
    that form a triangle (i, j, k) in 3D Euclidean space.
    Mathematical formulation:
        Z_ij = LayerNorm(Z_ij) + Gate_ij * Proj( sum_k (a_ik * b_jk) )
    """
    def __init__(self, d_z=64, c=32):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(d_z)
        
        # Left edge projection (i -> k) and right edge projection (j -> k)
        self.proj_left = nn.Linear(d_z, c)
        self.proj_right = nn.Linear(d_z, c)
        
        # Gates
        self.gate_left = nn.Linear(d_z, c)
        self.gate_right = nn.Linear(d_z, c)
        self.gate_out = nn.Linear(d_z, d_z)
        
        self.proj_out = nn.Linear(c, d_z)

    def forward(self, z):
        """
        z: Pair representation [Batch, Num_Residues, Num_Residues, d_z]
        """
        z_norm = self.norm(z)

        # 1. Project left and right edges with sigmoid gating
        left = self.proj_left(z_norm) * torch.sigmoid(self.gate_left(z_norm))   # a_ik: [B, R, R, c]
        right = self.proj_right(z_norm) * torch.sigmoid(self.gate_right(z_norm)) # b_jk: [B, R, R, c]

        # 2. Triangular sum over all bridging nodes k: sum_k (a_ik * b_jk)
        # einsum: b i k c, b j k c -> b i j c
        triangle_sum = torch.einsum("bikc,bjkc->bijc", left, right) # [B, R, R, c]

        # 3. Project back to pair dimension with output gate
        out = self.proj_out(triangle_sum) * torch.sigmoid(self.gate_out(z_norm)) # [B, R, R, d_z]
        return z + out # Residual connection


class TriangularSelfAttention(nn.Module):
    """
    Residue pair (i, j) attends to other residue pairs (i, k) along the row,
    with attention weights explicitly biased by the opposing third edge (j, k).
    """
    def __init__(self, d_z=64, num_heads=4):
        super().__init__()
        self.d_z = d_z
        self.num_heads = num_heads
        self.head_dim = d_z // num_heads

        self.norm = nn.LayerNorm(d_z)
        self.q_proj = nn.Linear(d_z, d_z, bias=False)
        self.k_proj = nn.Linear(d_z, d_z, bias=False)
        self.v_proj = nn.Linear(d_z, d_z, bias=False)
        
        # Bias projection from the opposing edge (j, k)
        self.bias_proj = nn.Linear(d_z, num_heads, bias=False)
        self.out_proj = nn.Linear(d_z, d_z)

    def forward(self, z):
        B, R, _, _ = z.shape
        z_norm = self.norm(z)

        # Project Q, K, V
        q = self.q_proj(z_norm).view(B, R, R, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k_proj(z_norm).view(B, R, R, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(z_norm).view(B, R, R, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # Shapes: [B, Heads, R_i, R_j/k, Head_Dim]

        # Pair bias from third edge (j, k)
        bias = self.bias_proj(z_norm).permute(0, 3, 1, 2) # [B, Heads, R_j, R_k]

        # Scaled dot-product along the k dimension
        # (q * k^T) / sqrt(d) + bias
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, Heads, R, R, R]
        scores = scores + bias.unsqueeze(2) # Add bias across row i

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) # [B, Heads, R, R, Head_Dim]

        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, R, R, self.d_z)
        return z + self.out_proj(out)


class MiniEvoformerBlock(nn.Module):
    """
    A compact Evoformer layer coupling evolutionary MSA channels with 2D geometric pair maps.
    """
    def __init__(self, d_m=32, d_z=64):
        super().__init__()
        self.outer_product = OuterProductMean(d_m=d_m, d_z=d_z)
        self.tri_update = TriangularMultiplicativeUpdateOutgoing(d_z=d_z)
        self.tri_attn = TriangularSelfAttention(d_z=d_z)

    def forward(self, msa, pair):
        # 1. Evolutionary communication: inject MSA co-variation into 2D pair geometry
        pair = pair + self.outer_product(msa)
        
        # 2. Geometric communication: enforce 3D triangle inequality
        pair = self.tri_update(pair)
        pair = self.tri_attn(pair)
        
        return msa, pair


if __name__ == "__main__":
    print("--- 48. The Evoformer (AlphaFold 2/3 Architecture) ---")
    print("Demonstrating Evolutionary Co-Variation Outer Product & Triangular Multiplicative Updates...")

    batch_size = 1
    num_sequences = 8    # 8 homologous species in the MSA
    num_residues = 16    # 16 amino acids in the protein chain
    d_m = 32             # MSA embedding dimension
    d_z = 64             # Geometric pair embedding dimension

    evoformer = MiniEvoformerBlock(d_m=d_m, d_z=d_z)

    dummy_msa = torch.randn(batch_size, num_sequences, num_residues, d_m)
    dummy_pair = torch.randn(batch_size, num_residues, num_residues, d_z)

    print(f"Input MSA Shape:  {dummy_msa.shape} [Batch, Sequences=8, Residues=16, d_m=32]")
    print(f"Input Pair Shape: {dummy_pair.shape} [Batch, Residues=16, Residues=16, d_z=64]")

    out_msa, out_pair = evoformer(dummy_msa, dummy_pair)

    print(f"\nUpdated MSA Shape:  {out_msa.shape}")
    print(f"Updated Pair Shape: {out_pair.shape}")
    print("Success: Evoformer executed triangular edge compositions and outer-product evolutionary injection.")

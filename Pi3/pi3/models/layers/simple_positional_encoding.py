import torch
import torch.nn as nn
import math


class SimpleLearnablePositionalEncoding(nn.Module):
    """
    Simple learnable positional encoding that handles arbitrary spatial dimensions.
    Combines temporal and spatial position information.
    """
    
    def __init__(self, d_model, max_seq_len=10, max_spatial_len=1024):
        super().__init__()
        self.d_model = d_model
        
        # Learnable temporal position embeddings
        self.temporal_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        # Learnable spatial position embeddings (1D indexing for flexibility)
        self.spatial_embed = nn.Parameter(torch.zeros(max_spatial_len, d_model))
        
        # Mixing parameter to combine temporal and spatial
        self.temporal_scale = nn.Parameter(torch.ones(1))
        self.spatial_scale = nn.Parameter(torch.ones(1))
        
        # Initialize
        nn.init.normal_(self.temporal_embed, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)
        
    def forward(self, x, start_idx=0):
        """
        Args:
            x: [B, T, S, D]
        Returns:
            x + positional encoding
        """
        B, T, S, D = x.shape
        device = x.device
        
        # Get temporal embeddings
        temporal_pos = self.temporal_embed[start_idx:start_idx+T]  # [T, D]
        temporal_pos = temporal_pos.unsqueeze(1).expand(-1, S, -1)  # [T, S, D]
        temporal_pos = temporal_pos.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, S, D]
        
        # Get spatial embeddings (handle S > max_spatial_len by wrapping)
        if S > self.spatial_embed.size(0):
            # Use modulo for positions beyond max
            spatial_indices = torch.arange(S, device=device) % self.spatial_embed.size(0)
            spatial_pos = self.spatial_embed[spatial_indices]  # [S, D]
        else:
            spatial_pos = self.spatial_embed[:S]  # [S, D]
        
        spatial_pos = spatial_pos.unsqueeze(0).expand(T, -1, -1)  # [T, S, D]
        spatial_pos = spatial_pos.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, S, D]
        
        # Combine with learnable scales
        pos_encoding = self.temporal_scale * temporal_pos + self.spatial_scale * spatial_pos
        
        return x + pos_encoding


class RoPE2DStyle(nn.Module):
    """
    2D Rotary Position Embedding adapted for non-square spatial dimensions.
    Inspired by RoPE but handles arbitrary spatial layouts.
    """
    
    def __init__(self, d_model, max_seq_len=10):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Temporal RoPE parameters
        self.temporal_dim = d_model // 4  # Reserve 1/4 for temporal
        self.spatial_dim = d_model - self.temporal_dim
        
    def forward(self, x, start_idx=0):
        """Apply RoPE-style encoding to x."""
        B, T, S, D = x.shape
        device = x.device
        
        # Split features for temporal and spatial RoPE
        x_temporal = x[..., :self.temporal_dim]
        x_spatial = x[..., self.temporal_dim:]
        
        # Apply temporal RoPE
        temporal_pos = torch.arange(start_idx, start_idx + T, device=device)
        x_temporal = self.apply_rope_1d(x_temporal, temporal_pos, dim=1)
        
        # Apply spatial RoPE (treat as 1D for flexibility with 782 tokens)
        spatial_pos = torch.arange(S, device=device)
        x_spatial_reshaped = x_spatial.transpose(1, 2)  # [B, S, T, spatial_dim]
        x_spatial_reshaped = self.apply_rope_1d(x_spatial_reshaped, spatial_pos, dim=1)
        x_spatial = x_spatial_reshaped.transpose(1, 2)  # [B, T, S, spatial_dim]
        
        # Combine
        return torch.cat([x_temporal, x_spatial], dim=-1)
    
    def apply_rope_1d(self, x, positions, dim):
        """Apply 1D RoPE to tensor x along specified dimension."""
        seq_len = x.shape[dim]
        dim_size = x.shape[-1]
        
        # Create rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_size, 2, device=x.device).float() / dim_size))
        
        # Compute rotation angles
        sincos = torch.outer(positions.float(), inv_freq)  # [seq_len, dim_size//2]
        sin = sincos.sin()
        cos = sincos.cos()
        
        # Apply rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Expand sin/cos to match x dimensions
        for _ in range(x.ndim - dim - 1):
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        
        # Move sin/cos to correct dimension
        perm = list(range(sin.ndim))
        perm[dim] = 1
        perm[1] = dim
        sin = sin.permute(perm)
        cos = cos.permute(perm)
        
        # Apply rotation
        x_out = x.clone()
        x_out[..., 0::2] = x1 * cos - x2 * sin
        x_out[..., 1::2] = x1 * sin + x2 * cos
        
        return x_out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .simple_positional_encoding import SimpleLearnablePositionalEncoding, RoPE2DStyle


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding for sequence modeling."""
    
    def __init__(self, d_model, max_len=15):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # Learnable positional embeddings for temporal dimension
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_len, 1, d_model))
        nn.init.normal_(self.temporal_pos_embed, std=0.02)
        
    def forward(self, x, start_idx=0):
        """
        Args:
            x: [B, T, S, D] - temporal, spatial, feature dimensions
            start_idx: starting position for positional encoding
        """
        B, T, S, D = x.shape
        
        # Get positional embeddings for the temporal positions
        pos_embed = self.temporal_pos_embed[:, start_idx:start_idx+T, :, :]  # [1, T, 1, D]
        # Properly broadcast to all spatial locations
        pos_embed = pos_embed.expand(B, -1, S, -1)  # [B, T, S, D]
        return x + pos_embed


class TemporalCausalAttention(nn.Module):
    """
    Causal self-attention across temporal dimension.
    Each spatial token attends to previous temporal positions at the same spatial location.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: [B, T, S, D] - batch, temporal, spatial, feature
            kv_cache: optional tuple (cached_k, cached_v) each [B*S, n_heads, T_cached, head_dim]
        Returns:
            output: [B, T, S, D]
            new_kv_cache: tuple (k, v) for caching (None if kv_cache was not used)
        """
        B, T, S, D = x.shape

        # Process each spatial location independently
        # Reshape to [B*S, T, D] to handle temporal attention per spatial location
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)

        # Compute QKV
        qkv = self.qkv(x_reshaped)  # [B*S, T, 3*D]
        qkv = qkv.view(B * S, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*S, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # KV-cache: append new K/V to cached K/V, only compute attention for new queries
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # [B*S, n_heads, T_cached+T, head_dim]
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        T_full = k.shape[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*S, n_heads, T, T_full]

        # Apply causal mask
        if T_full > 1:
            # q has T positions starting at offset (T_full - T)
            # Each query at position i can attend to positions 0..i
            offset = T_full - T
            causal_mask = torch.zeros(T, T_full, device=x.device, dtype=torch.bool)
            for i in range(T):
                causal_mask[i, offset + i + 1:] = True
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B*S, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B * S, T, D)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Reshape back to [B, T, S, D]
        out = out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()

        return out, new_kv_cache


class AutoregressiveTransformerBlock(nn.Module):
    """Single transformer block for autoregressive modeling."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TemporalCausalAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, kv_cache=None):
        # x: [B, T, S, D]
        attn_out, new_kv_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv_cache


class AutoregressiveTokenTransformer(nn.Module):
    """
    Autoregressive transformer that takes scene tokens and generates future tokens.
    Designed to work after Pi3's decode() function.
    """
    
    def __init__(self,
                 d_model=2048,  # 2 * dec_embed_dim from Pi3
                 n_heads=16,
                 n_layers=8,
                 d_ff=2048,
                 dropout=0.1,
                 n_future_frames=3,
                 max_seq_len=15,
                 positional_encoding='spatiotemporal',  # 'temporal', 'spatiotemporal', or 'sinusoidal'
                 max_spatial_size=64*64):
        super().__init__()
        
        self.d_model = d_model
        self.n_future_frames = n_future_frames
        self.n_heads = n_heads
        
        # Choose positional encoding type
        if positional_encoding == 'temporal':
            self.temporal_pos_embed = TemporalPositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'spatiotemporal':
            self.temporal_pos_embed = SimpleLearnablePositionalEncoding(
                d_model, max_seq_len, max_spatial_size
            )
        elif positional_encoding == 'rope':
            self.temporal_pos_embed = RoPE2DStyle(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AutoregressiveTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(d_model)
        
        # No token predictor - use transformer output directly
        # The transformer blocks already transform the representation
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def reshape_to_temporal(self, hidden, N):
        """
        Reshape tokens from decode() to temporal format.
        
        Args:
            hidden: [B*N, S, D] - output from decode()
            N: number of current frames
            
        Returns:
            tokens: [B, N, S, D]
        """
        BN, S, D = hidden.shape
        B = BN // N
        return hidden.view(B, N, S, D)
    
    def reshape_to_spatial(self, tokens):
        """
        Reshape tokens back to spatial format for decoders.
        
        Args:
            tokens: [B, T, S, D]
            
        Returns:
            hidden: [B*T, S, D]
        """
        B, T, S, D = tokens.shape
        return tokens.view(B * T, S, D)
    
    def forward(self, hidden, N, pos=None, n_future_frames_override=None, use_kv_cache=False):
        """
        Forward pass to generate future tokens autoregressively.

        Args:
            hidden: [B*N, S, D] - aggregated tokens from decode()
            N: number of current frames
            pos: [B*N, S, 2] - positional encoding from decode()
            n_future_frames_override: Optional override for n_future_frames.
            use_kv_cache: If True, use KV-caching for faster autoregressive generation.
                Each AR step only runs the new token through the transformer, reusing
                cached keys/values from previous steps.

        Returns:
            all_hidden: [B*(N+M), S, D] - current + future tokens
            all_pos: [B*(N+M), S, 2] - positional encoding for all tokens
        """
        # Use override if provided, otherwise use default
        n_future = n_future_frames_override if n_future_frames_override is not None else self.n_future_frames

        # Validate inputs
        BN, S, D = hidden.shape
        assert BN % N == 0, f"Batch size mismatch: {BN} not divisible by {N}"
        assert D == self.d_model, f"Feature dimension mismatch: got {D}, expected {self.d_model}"

        # Reshape to temporal sequence
        tokens = self.reshape_to_temporal(hidden, N)  # [B, N, S, D]
        B, _, S, D = tokens.shape

        if use_kv_cache:
            return self._forward_with_kv_cache(tokens, n_future, pos, N, B, S, D)

        # Start with current tokens
        all_tokens = tokens

        # Generate future tokens autoregressively
        for i in range(n_future):
            # Add temporal positional encoding (always start from 0 for the full sequence)
            tokens_with_pos = self.temporal_pos_embed(all_tokens, start_idx=0)

            # Pass through transformer blocks (causal attention ensures autoregressive behavior)
            x = tokens_with_pos

            for block in self.blocks:
                x, _ = block(x)
            x = self.norm(x)

            # Use the last frame's transformer output directly as the next token
            last_frame = x[:, -1, :, :]  # [B, S, D]
            next_token = last_frame.unsqueeze(1)  # [B, 1, S, D]

            # Append to sequence
            all_tokens = torch.cat([all_tokens, next_token], dim=1)

        # Reshape back to spatial format
        all_hidden = self.reshape_to_spatial(all_tokens)  # [B*(N+M), S, D]

        # Extend positional encoding for future frames
        all_pos = self._extend_pos(pos, B, N, S, n_future)

        return all_hidden, all_pos

    def _forward_with_kv_cache(self, tokens, n_future, pos, N, B, S, D):
        """Fast autoregressive generation with KV-caching."""
        # Step 1: Prefill — run all current tokens through the transformer, cache KV
        tokens_with_pos = self.temporal_pos_embed(tokens, start_idx=0)  # [B, N, S, D]
        x = tokens_with_pos

        layer_caches = []
        for block in self.blocks:
            x, kv_cache = block(x)
            layer_caches.append(kv_cache)
        x = self.norm(x)

        # Collect output for current frames and seed for generation
        all_tokens = [tokens]  # raw tokens (without pos encoding)
        last_frame = x[:, -1:, :, :]  # [B, 1, S, D] — next token prediction

        # Step 2: Decode — generate future tokens one at a time using cached KV
        for i in range(n_future):
            next_token = last_frame[:, 0, :, :].unsqueeze(1)  # [B, 1, S, D]
            all_tokens.append(next_token)

            # Add positional encoding for just this new token
            t_idx = N + i  # temporal position of this new token
            token_with_pos = self.temporal_pos_embed(next_token, start_idx=t_idx)  # [B, 1, S, D]

            x = token_with_pos
            new_caches = []
            for layer_idx, block in enumerate(self.blocks):
                x, kv_cache = block(x, kv_cache=layer_caches[layer_idx])
                new_caches.append(kv_cache)
            layer_caches = new_caches
            x = self.norm(x)

            last_frame = x  # [B, 1, S, D]

        all_tokens = torch.cat(all_tokens, dim=1)  # [B, N+M, S, D]
        all_hidden = self.reshape_to_spatial(all_tokens)  # [B*(N+M), S, D]
        all_pos = self._extend_pos(pos, B, N, S, n_future)

        return all_hidden, all_pos

    def _extend_pos(self, pos, B, N, S, n_future):
        """Extend positional encoding for future frames."""
        if pos is not None:
            pos_per_frame = pos.view(B, N, S, -1)
            spatial_pattern = pos_per_frame[:, 0:1, :, :]
            future_pos = spatial_pattern.repeat(1, n_future, 1, 1)
            all_pos_frames = torch.cat([pos_per_frame, future_pos], dim=1)
            return all_pos_frames.view(-1, S, pos_per_frame.shape[-1])
        return None

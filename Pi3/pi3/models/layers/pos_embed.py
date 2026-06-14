from __future__ import annotations

import torch


class RoPE2D(torch.nn.Module):
    """Minimal 2D rotary position embedding used by the LFG decoder."""

    def __init__(self, freq: float = 100.0):
        super().__init__()
        self.base = freq
        self._cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, seq_len, device, dtype)
        if key not in self._cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self._cache[key] = (freqs.cos(), freqs.sin())
        return self._cache[key]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        left, right = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-right, left), dim=-1)

    def _apply_1d(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = torch.nn.functional.embedding(positions, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(positions, sin)[:, None, :, :]
        return (tokens * cos) + (self._rotate_half(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if tokens.size(-1) % 2 != 0:
            raise ValueError("RoPE2D expects an even feature dimension.")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [batch, tokens, 2].")

        dim = tokens.size(-1) // 2
        max_position = int(positions.max().item()) + 1
        cos, sin = self._cos_sin(dim, max_position, tokens.device, tokens.dtype)

        y_tokens, x_tokens = tokens.chunk(2, dim=-1)
        y_tokens = self._apply_1d(y_tokens, positions[:, :, 0], cos, sin)
        x_tokens = self._apply_1d(x_tokens, positions[:, :, 1], cos, sin)
        return torch.cat((y_tokens, x_tokens), dim=-1)


class PositionGetter:
    """Create cached [y, x] patch-grid positions."""

    def __init__(self):
        self._cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self._cache[key] = torch.cartesian_prod(y, x)
        return self._cache[key].view(1, height * width, 2).expand(batch_size, -1, 2).clone()

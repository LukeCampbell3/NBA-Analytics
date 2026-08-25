"""Expert FFN bank + dense FFN (spec sections 18-20)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseFFN(nn.Module):
    """Baseline 1 (spec section 18): the same block used everywhere else,
    just without any router/experts."""

    def __init__(self, hidden_dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mult, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.net(x), {}


class ExpertBank(nn.Module):
    """n_experts independent 2-layer FFNs, batched via einsum so all
    experts can be evaluated for all tokens (see router.py docstring on
    why this implementation masks rather than dispatches)."""

    def __init__(self, hidden_dim: int, n_experts: int, mult: int = 4):
        super().__init__()
        self.n_experts = n_experts
        inner = hidden_dim * mult
        self.w1 = nn.Parameter(torch.randn(n_experts, hidden_dim, inner) * (hidden_dim**-0.5))
        self.b1 = nn.Parameter(torch.zeros(n_experts, inner))
        self.w2 = nn.Parameter(torch.randn(n_experts, inner, hidden_dim) * (inner**-0.5))
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden_dim))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H) -> (B, T, E, H) -- every expert's output for every token."""
        h = torch.einsum("bth,ehi->btei", x, self.w1) + self.b1  # (B,T,E,inner)
        h = self.act(h)
        out = torch.einsum("btei,eih->bteh", h, self.w2) + self.b2  # (B,T,E,H)
        return out

    def active_params_per_token(self, top_k: int) -> int:
        per_expert = self.w1[0].numel() + self.b1[0].numel() + self.w2[0].numel() + self.b2[0].numel()
        return per_expert * top_k

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

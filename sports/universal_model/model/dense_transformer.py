"""Dense shared Transformer stem, with pluggable FFN sublayer per block so
the same code path builds the dense baseline, the Switch Top-1 baseline,
and the Top-2 MoE target model (spec sections 18-20) -- matched attention
layers, matched hidden size, matched training data; only the FFN type
differs, which is what isolates the value of sparse specialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sports.universal_model.model.experts import DenseFFN
from sports.universal_model.model.moe import Top2MoEFFN
from sports.universal_model.model.switch import SwitchFFN


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.dropout(attn_out)
        ffn_out, diagnostics = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        return x, diagnostics


def build_ffn(block_type: str, hidden_dim: int, n_experts: int, mult: int) -> nn.Module:
    if block_type == "dense":
        return DenseFFN(hidden_dim, mult)
    if block_type == "switch":
        return SwitchFFN(hidden_dim, n_experts, mult)
    if block_type == "top2_moe":
        return Top2MoEFFN(hidden_dim, n_experts, mult)
    raise ValueError(f"unknown block_type={block_type!r}")


class UniversalTransformerStem(nn.Module):
    """block_types: list like ["dense","dense","dense","dense"] (Baseline 1),
    ["dense","dense","switch","switch"] (Baseline 2 -- dense blocks stay
    dense, only the designated MoE-layer positions become sparse, per spec
    section 20's "dense blocks: 4-8, MoE blocks: 2-4"), or with "top2_moe"."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        block_types: list[str],
        n_experts: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, build_ffn(bt, hidden_dim, n_experts, ffn_mult), dropout)
                for bt in block_types
            ]
        )
        self.block_types = block_types
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        x = tokens
        all_diagnostics = []
        for block in self.blocks:
            x, diag = block(x)
            all_diagnostics.append(diag)
        x = self.final_norm(x)
        pooled = x.mean(dim=1)  # (B, H) -- mean pool over the 8 typed tokens
        return pooled, all_diagnostics

    def moe_layers(self) -> list[nn.Module]:
        return [b.ffn for b, bt in zip(self.blocks, self.block_types) if bt in ("switch", "top2_moe")]

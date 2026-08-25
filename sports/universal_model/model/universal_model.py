"""Full universal model: FeatureTokenizer -> UniversalTransformerStem ->
MultiTaskHeads. One class, three configs (dense/switch/top2_moe) via
``block_types`` -- this is the "one shared checkpoint, any sport" object
(spec section 39): the same instance runs mlb and nfl (and would run
nba/golf/f1 the moment an adapter reports sufficient data) without any
sport-specific submodule.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sports.universal_model.model.dense_transformer import UniversalTransformerStem
from sports.universal_model.model.feature_tokenizer import FeatureTokenizer
from sports.universal_model.model.heads import MultiTaskHeads


class UniversalModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        block_types: list[str] | None = None,
        n_experts: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        block_types = block_types or ["dense", "dense", "dense", "dense"]
        self.tokenizer = FeatureTokenizer(hidden_dim=hidden_dim)
        self.stem = UniversalTransformerStem(hidden_dim, n_heads, block_types, n_experts, ffn_mult, dropout)
        self.heads = MultiTaskHeads(hidden_dim)
        self.config = {
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "block_types": block_types,
            "n_experts": n_experts,
            "ffn_mult": ffn_mult,
            "dropout": dropout,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        tokens = self.tokenizer(batch)
        pooled, block_diagnostics = self.stem(tokens)
        outputs = self.heads(pooled)
        outputs["latent_embedding"] = pooled
        outputs["block_diagnostics"] = block_diagnostics
        return outputs

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameters_per_token(self) -> int:
        """Dense params (always active) + only the top_k experts' params
        per MoE/Switch layer (spec section 42, active-parameter efficiency)."""
        dense_total = 0
        active_moe = 0
        for block, bt in zip(self.stem.blocks, self.stem.block_types):
            if bt == "dense":
                dense_total += sum(p.numel() for p in block.ffn.parameters())
            else:
                top_k = block.ffn.router.top_k
                active_moe += block.ffn.experts.active_params_per_token(top_k)
                active_moe += sum(p.numel() for p in block.ffn.router.parameters())
        attn_params = sum(
            sum(p.numel() for p in block.attn.parameters()) + sum(p.numel() for p in block.ln1.parameters())
            + sum(p.numel() for p in block.ln2.parameters())
            for block in self.stem.blocks
        )
        other = sum(p.numel() for p in self.tokenizer.parameters()) + sum(p.numel() for p in self.heads.parameters())
        return dense_total + active_moe + attn_params + other

"""Switch-style Top-1 sparse FFN (Baseline 2, spec section 19)."""
from __future__ import annotations

import torch
import torch.nn as nn

from sports.universal_model.model.experts import ExpertBank
from sports.universal_model.model.router import Router


class SwitchFFN(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int, mult: int = 4):
        super().__init__()
        self.router = Router(hidden_dim, n_experts, top_k=1)
        self.experts = ExpertBank(hidden_dim, n_experts, mult)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        route = self.router(x)
        expert_out = self.experts(x)  # (B,T,E,H)
        weighted = (expert_out * route["weights"].unsqueeze(-1)).sum(dim=2)  # (B,T,H)
        diagnostics = {
            "load_balance_loss": route["load_balance_loss"],
            "z_loss": route["z_loss"],
            "routing_entropy": route["routing_entropy"],
            "tokens_per_expert": route["tokens_per_expert"],
            "top_idx": route["top_idx"],
        }
        return weighted, diagnostics

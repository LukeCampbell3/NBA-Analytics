"""Top-2 sparse MoE FFN -- the target architecture (spec section 20)."""
from __future__ import annotations

import torch
import torch.nn as nn

from sports.universal_model.model.experts import ExpertBank
from sports.universal_model.model.router import Router


class Top2MoEFFN(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int, mult: int = 4):
        super().__init__()
        self.router = Router(hidden_dim, n_experts, top_k=2)
        self.experts = ExpertBank(hidden_dim, n_experts, mult)
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        route = self.router(x)
        expert_out = self.experts(x)
        weighted = (expert_out * route["weights"].unsqueeze(-1)).sum(dim=2)
        diagnostics = {
            "load_balance_loss": route["load_balance_loss"],
            "z_loss": route["z_loss"],
            "routing_entropy": route["routing_entropy"],
            "tokens_per_expert": route["tokens_per_expert"],
            "top_idx": route["top_idx"],
        }
        return weighted, diagnostics

    def add_expert(self) -> None:
        """Function-preserving expert birth (spec section 27): append one
        new expert row initialized so its immediate contribution is zero.
        The router's gate is extended with a zero-weight column so the new
        expert starts with ~1/(n+1) routing probability, not literally
        excluded -- true alpha=0 gating is enforced by the DRM controller
        wrapping this call with an explicit alpha coefficient (see
        drm_controller/mutations.py), this method only handles the
        parameter-shape mutation safely."""
        hidden_dim = self.experts.w1.shape[1]
        inner = self.experts.w1.shape[2]
        device = self.experts.w1.device
        new_w1 = torch.randn(1, hidden_dim, inner, device=device) * (hidden_dim**-0.5)
        new_b1 = torch.zeros(1, inner, device=device)
        new_w2 = torch.zeros(1, inner, hidden_dim, device=device)  # zero output => function-preserving at birth
        new_b2 = torch.zeros(1, hidden_dim, device=device)
        self.experts.w1 = nn.Parameter(torch.cat([self.experts.w1.data, new_w1], dim=0))
        self.experts.b1 = nn.Parameter(torch.cat([self.experts.b1.data, new_b1], dim=0))
        self.experts.w2 = nn.Parameter(torch.cat([self.experts.w2.data, new_w2], dim=0))
        self.experts.b2 = nn.Parameter(torch.cat([self.experts.b2.data, new_b2], dim=0))
        self.experts.n_experts += 1
        self.n_experts += 1
        old_gate = self.router.gate
        new_gate = nn.Linear(old_gate.in_features, old_gate.out_features + 1, device=device)
        with torch.no_grad():
            new_gate.weight[:-1] = old_gate.weight
            new_gate.bias[:-1] = old_gate.bias
            new_gate.weight[-1].zero_()
            new_gate.bias[-1] = old_gate.bias.min() - 5.0  # start near-unreachable, not exactly excluded
        self.router.gate = new_gate
        self.router.n_experts += 1

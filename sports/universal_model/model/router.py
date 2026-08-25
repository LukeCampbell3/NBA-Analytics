"""Router shared by Switch (Top-1) and Top-2 MoE layers (spec sections
19-22). Kept separate from the FFN/expert modules so both baselines reuse
identical gating math and identical diagnostics -- any quality difference
between Switch and Top-2 in the final report is attributable to routing
width, not to incidental implementation differences.

Implementation note (disclosed, not hidden): at this model's scale (8
tokens/example, hidden<=192) there is no custom sparse dispatch kernel.
All experts are evaluated for all tokens and the unselected ones are
masked to zero before summing -- this is correct for every quality/
routing/collapse diagnostic below, and active-parameter/FLOPs accounting
in train/trainer.py is computed analytically (not from wall-clock), but
the *measured* wall-clock throughput of this implementation does NOT
reflect the theoretical sparse compute advantage. See FINAL_REPORT.md
compute-efficiency section.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, n_experts)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T, H) -> gate weights (B, T, n_experts) with only top_k
        nonzero per token, plus diagnostics."""
        logits = self.gate(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)  # (B, T, k)
        mask = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)
        renorm = top_vals.sum(-1, keepdim=True).clamp_min(1e-9)
        weights = torch.zeros_like(probs)
        weights.scatter_(-1, top_idx, top_vals / renorm)

        # Load-balancing auxiliary loss (Switch-style): E * sum_e f_e * P_e
        tokens_per_expert = mask.sum(dim=(0, 1))  # (E,)
        f = tokens_per_expert / tokens_per_expert.sum().clamp_min(1e-9)
        P = probs.mean(dim=(0, 1))  # (E,)
        load_balance_loss = self.n_experts * (f * P).sum()

        # Router z-loss: penalize large logit magnitudes for stability.
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1).mean()

        return {
            "weights": weights,  # (B, T, E)
            "mask": mask,
            "load_balance_loss": load_balance_loss,
            "z_loss": z_loss,
            "routing_entropy": entropy.detach(),
            "tokens_per_expert": tokens_per_expert.detach(),
            "top_idx": top_idx.detach(),
        }

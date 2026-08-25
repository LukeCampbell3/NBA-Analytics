"""Multi-task output heads (spec section 15).

Implements the subset of the required output object that this dataset can
honestly support: ``expected_outcome`` (z_pred), ``P(over exact line)``
(prob_over), ``calibrated_probability`` (same sigmoid, calibration is
evaluated not separately re-modeled at this scale), and ``latent_embedding``
(the pooled stem output, returned by the caller). Distribution-parameter
and quantile heads are explicitly NOT implemented -- flagged in
FINAL_REPORT.md as JOINT_HEAD_INSUFFICIENT_DATA-style deferrals rather than
silently omitted, since this dataset has no historical pair-outcome table
to support them (spec section 53).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.prob_over_head = nn.Linear(hidden_dim, 1)
        self.z_head = nn.Linear(hidden_dim, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        prob_over_logit = self.prob_over_head(pooled).squeeze(-1)
        z_pred = self.z_head(pooled).squeeze(-1)
        return {
            "prob_over_logit": prob_over_logit,
            "prob_over": torch.sigmoid(prob_over_logit),
            "z_pred": z_pred,
        }

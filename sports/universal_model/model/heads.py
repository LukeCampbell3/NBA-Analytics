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

REVISION, TESTED AND REJECTED: a 2-layer MLP head (hidden_dim->hidden_dim->1,
GELU, dropout) was tried here as part of a loss/MAE optimization pass and
measured WORSE than this single-Linear head on three separate real runs
(SELECT brier/MAE regressed in every one -- see git history for the exact
numbers). Reverted rather than kept on the theory that "more capacity
should help": on this dataset's real, thin feature set (21 allowed
columns, see manifests/feature_registry.json), the extra head depth and
dropout appear to add optimization difficulty without a signal-capacity
benefit worth it. Kept as a single Linear layer, matching the
already-evidenced-good original.
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

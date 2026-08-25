"""Transparent multi-task objective (spec section 32) -- masked so
unavailable targets never create fake labels, and every component reported
separately rather than folded into one opaque number."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_losses(outputs: dict, batch: dict, config) -> dict[str, torch.Tensor]:
    y_over = batch["y_over"]
    y_mask = batch["y_over_mask"]
    prob_loss_raw = F.binary_cross_entropy_with_logits(outputs["prob_over_logit"], y_over.clamp(0, 1), reduction="none")
    prob_loss = (prob_loss_raw * y_mask).sum() / y_mask.sum().clamp_min(1.0)

    z_valid = batch["z_valid"]
    reg_loss_raw = F.mse_loss(outputs["z_pred"], batch["z_actual"], reduction="none")
    reg_loss = (reg_loss_raw * z_valid).sum() / z_valid.sum().clamp_min(1.0)

    load_balance = torch.zeros((), device=y_over.device)
    z_loss = torch.zeros((), device=y_over.device)
    n_moe_blocks = 0
    for diag in outputs["block_diagnostics"]:
        if "load_balance_loss" in diag:
            load_balance = load_balance + diag["load_balance_loss"]
            z_loss = z_loss + diag["z_loss"]
            n_moe_blocks += 1
    if n_moe_blocks:
        load_balance = load_balance / n_moe_blocks
        z_loss = z_loss / n_moe_blocks

    total = (
        config.lambda_prob * prob_loss
        + config.lambda_reg * reg_loss
        + config.lambda_load_balance * load_balance
        + config.lambda_z_loss * z_loss
    )
    return {
        "total": total,
        "prob_loss": prob_loss.detach(),
        "reg_loss": reg_loss.detach(),
        "load_balance_loss": load_balance.detach(),
        "router_z_loss": z_loss.detach(),
    }

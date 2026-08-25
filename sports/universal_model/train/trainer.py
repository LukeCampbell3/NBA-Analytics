"""Training loop shared by all three baselines (dense / Switch / Top-2 MoE)
-- spec section 33 stages 2-4. Deferred consolidation (section 30): DRM
OBSERVE/DERIVE/COMMIT cycles happen strictly outside this hot loop, at
checkpoint boundaries only (see drm_controller/controller.py).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.model.universal_model import UniversalModel
from sports.universal_model.train.config import TrainConfig
from sports.universal_model.train.losses import compute_losses
from sports.universal_model.train.sampler import build_temperature_sampler
from sports.universal_model.validation.metrics import classification_metrics, regression_metrics


@torch.no_grad()
def evaluate(model: UniversalModel, dataset: UniversalDataset, batch_size: int = 256) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_y, all_z_pred, all_z_actual = [], [], [], []
    per_sport_probs: dict[str, list] = {}
    per_sport_y: dict[str, list] = {}
    sport_col = dataset.frame["sport"].values
    idx = 0
    for batch in loader:
        out = model(batch)
        bs = batch["y_over"].shape[0]
        batch_sports = sport_col[idx: idx + bs]
        idx += bs
        mask = batch["y_over_mask"].numpy() > 0
        probs = out["prob_over"].numpy()
        y = batch["y_over"].numpy()
        all_probs.append(probs[mask])
        all_y.append(y[mask])
        for s in set(batch_sports):
            smask = (batch_sports == s) & mask
            per_sport_probs.setdefault(s, []).append(probs[smask])
            per_sport_y.setdefault(s, []).append(y[smask])
        zmask = batch["z_valid"].numpy() > 0
        all_z_pred.append(out["z_pred"].numpy()[zmask])
        all_z_actual.append(batch["z_actual"].numpy()[zmask])
    model.train()

    probs_cat = np.concatenate(all_probs) if all_probs else np.array([])
    y_cat = np.concatenate(all_y) if all_y else np.array([])
    macro = {}
    for s in per_sport_probs:
        p = np.concatenate(per_sport_probs[s])
        yy = np.concatenate(per_sport_y[s])
        macro[s] = classification_metrics(p, yy)

    return {
        "micro_classification": classification_metrics(probs_cat, y_cat),
        "macro_by_sport": macro,
        "worst_sport_brier": max((m["brier"] for m in macro.values() if m["brier"] is not None), default=None),
        "regression": regression_metrics(
            np.concatenate(all_z_pred) if all_z_pred else np.array([]),
            np.concatenate(all_z_actual) if all_z_actual else np.array([]),
        ),
    }


def train_model(config: TrainConfig, sports: list[str] | None = None, split_kind: str = "per_sport") -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    derive = UniversalDataset(split="DERIVE", sports=sports, split_kind=split_kind)
    select = UniversalDataset(split="SELECT", sports=sports, split_kind=split_kind)
    return train_on(config, derive, select)


def train_on(config: TrainConfig, derive: UniversalDataset, select: UniversalDataset) -> dict:
    """Lower-level entry point taking already-built datasets, so transfer/
    small-data/negative-transfer studies (validation/transfer.py) can pass
    custom sport subsets or fraction-truncated DERIVE sets without
    duplicating the training loop."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    sampler = build_temperature_sampler(derive, alpha=config.alpha)
    loader = DataLoader(derive, batch_size=config.batch_size, sampler=sampler)
    loader_iter = itertools.cycle(loader)

    model = UniversalModel(
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        block_types=config.block_types,
        n_experts=config.n_experts,
        ffn_mult=config.ffn_mult,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    history = []
    t0 = time.time()
    examples_seen = 0
    for step in range(1, config.steps + 1):
        batch = next(loader_iter)
        out = model(batch)
        losses = compute_losses(out, batch, config)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        examples_seen += batch["y_over"].shape[0]

        if step % config.eval_every == 0 or step == config.steps:
            eval_metrics = evaluate(model, select)
            elapsed = time.time() - t0
            history.append(
                {
                    "step": step,
                    "train_loss": float(losses["total"].item()),
                    "prob_loss": float(losses["prob_loss"].item()),
                    "reg_loss": float(losses["reg_loss"].item()),
                    "load_balance_loss": float(losses["load_balance_loss"].item()),
                    "router_z_loss": float(losses["router_z_loss"].item()),
                    "select_metrics": eval_metrics,
                    "elapsed_sec": elapsed,
                    "examples_per_sec": examples_seen / elapsed if elapsed > 0 else None,
                }
            )
    total_elapsed = time.time() - t0
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "final_select_metrics": history[-1]["select_metrics"] if history else None,
        "total_params": model.total_parameters(),
        "active_params": model.active_parameters_per_token(),
        "wall_time_sec": total_elapsed,
        "examples_per_sec": examples_seen / total_elapsed if total_elapsed > 0 else None,
        "sampler_effective_contribution": getattr(sampler, "effective_contribution", None),
        "config": config,
    }

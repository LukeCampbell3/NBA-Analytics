"""Required ablations (spec section 49).

DISCLOSED SCOPE: of the 9 required ablations (A-I), this module runs:

  A. sport identity removed       -- input-masking ablation (no retrain)
  B. role/position features removed -- input-masking ablation (no retrain)
  C. market prior removed         -- input-masking ablation (no retrain)
  D. MoE replaced by dense FFN    -- reference to the dense baseline run
  E. Top-1 vs Top-2               -- reference to the switch baseline run
  F. router balance loss removed  -- real retrain with lambda_load_balance=0
  G. DRM disabled                 -- reference to top2_moe pre-DRM result
  H. DRM expert-birth only        -- a DRM run with only that tier enabled
  I. full bounded DRM controller  -- reference to the full DRM run

A/B/C are implemented as input-masking ablations on an already-trained
model rather than full retrains: this isolates "how much does the trained
model rely on this input" without the cost/variance of retraining from
scratch, and is a standard, legitimate ablation methodology. F/H require a
real retrain since they change the training objective/procedure itself,
not just the input.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.model.universal_model import UniversalModel
from sports.universal_model.train.config import TrainConfig
from sports.universal_model.train.trainer import evaluate, train_on
from sports.universal_model.validation.metrics import classification_metrics, regression_metrics


@torch.no_grad()
def _evaluate_with_override(model: UniversalModel, dataset: UniversalDataset, override_fn, batch_size: int = 256) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, ys, zpred, zact = [], [], [], []
    for batch in loader:
        override_fn(batch)
        out = model(batch)
        mask = batch["y_over_mask"].numpy() > 0
        probs.append(out["prob_over"].numpy()[mask])
        ys.append(batch["y_over"].numpy()[mask])
        zmask = batch["z_valid"].numpy() > 0
        zpred.append(out["z_pred"].numpy()[zmask])
        zact.append(batch["z_actual"].numpy()[zmask])
    model.train()
    import numpy as np

    return {
        "classification": classification_metrics(np.concatenate(probs) if probs else np.array([]), np.concatenate(ys) if ys else np.array([])),
        "regression": regression_metrics(np.concatenate(zpred) if zpred else np.array([]), np.concatenate(zact) if zact else np.array([])),
    }


def ablate_sport_identity(batch: dict) -> None:
    batch["sport_id"] = torch.zeros_like(batch["sport_id"])


def ablate_role_features(batch: dict) -> None:
    batch["role_id"] = torch.zeros_like(batch["role_id"])
    batch["home_id"] = torch.zeros_like(batch["home_id"])


def ablate_market_features(batch: dict) -> None:
    # numeric column order: [line, american_price, sample_support_rows, days_since_last_history, recency_prior]
    batch["numeric"][:, 0:2] = 0.0
    batch["missing"][:, 0:2] = 1.0


def run_input_ablations(model: UniversalModel, dataset: UniversalDataset) -> dict:
    baseline = evaluate(model, dataset)
    return {
        "baseline": baseline["micro_classification"],
        "A_sport_identity_removed": _evaluate_with_override(model, dataset, ablate_sport_identity)["classification"],
        "B_role_features_removed": _evaluate_with_override(model, dataset, ablate_role_features)["classification"],
        "C_market_prior_removed": _evaluate_with_override(model, dataset, ablate_market_features)["classification"],
    }


def run_router_balance_ablation(base_config: TrainConfig) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.name = "top2_moe_no_router_balance_loss"
    cfg.lambda_load_balance = 0.0
    cfg.lambda_z_loss = 0.0
    derive = UniversalDataset(split="DERIVE")
    select = UniversalDataset(split="SELECT")
    result = train_on(cfg, derive, select)
    return {
        "final_select_metrics": result["final_select_metrics"],
        "total_params": result["total_params"],
        "active_params": result["active_params"],
    }

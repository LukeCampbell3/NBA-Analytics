"""Transfer / small-data / negative-transfer studies (spec sections
11.C/11.F, 45, 46).

DISCLOSED SCOPE LIMITATION: only two sports (mlb, nfl) have
sufficient_for_training=True data in this repository (see
reports/INVENTORY.md). A true "leave-one-sport-out" test across 5 sports
is not possible here -- what this module actually runs is the 2-sport
version of the same question: for each of {mlb, nfl}, compare a
single-sport-only model against the pooled multi-sport model, evaluated on
that sport's own held-out data. This is a real, honestly-scoped instance
of the transfer hypothesis, not the full mission-scale version -- reported
as such in FINAL_REPORT.md, not inflated.
"""
from __future__ import annotations

import copy

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.train.config import TrainConfig
from sports.universal_model.train.trainer import evaluate, train_on


def leave_one_sport_out(base_config: TrainConfig, all_sports: list[str]) -> dict:
    """For each sport S: train single-sport-only on S's own DERIVE/SELECT,
    and separately reuse the pooled multi-sport model's performance on S
    (passed in by the caller after pooled training) is compared in
    validation/run_full_validation.py. This function only produces the
    single-sport-only baselines."""
    results = {}
    for sport in all_sports:
        cfg = copy.deepcopy(base_config)
        cfg.name = f"{sport}_only"
        derive = UniversalDataset(split="DERIVE", sports=[sport])
        select = UniversalDataset(split="SELECT", sports=[sport])
        result = train_on(cfg, derive, select)
        results[sport] = {
            "final_select_metrics": result["final_select_metrics"],
            "total_params": result["total_params"],
            "active_params": result["active_params"],
            "wall_time_sec": result["wall_time_sec"],
        }
    return results


def negative_transfer_audit(single_sport_results: dict, pooled_model, all_sports: list[str]) -> dict:
    """Compare each sport's single-sport-only SELECT metrics against the
    same sport's slice of the pooled multi-sport model's SELECT metrics.
    Reports the delta directly rather than only a verdict (spec section 46:
    "Report delta. If universal training hurts materially, investigate...
    rather than hiding the result.")."""
    audit = {}
    for sport in all_sports:
        select_sport_only = UniversalDataset(split="SELECT", sports=[sport])
        pooled_on_sport = evaluate(pooled_model, select_sport_only)
        solo = single_sport_results[sport]["final_select_metrics"]
        audit[sport] = {
            "solo_model": solo["micro_classification"],
            "pooled_model_on_this_sport": pooled_on_sport["micro_classification"],
            "solo_regression": solo["regression"],
            "pooled_regression_on_this_sport": pooled_on_sport["regression"],
        }
        if solo["micro_classification"]["brier"] is not None and pooled_on_sport["micro_classification"]["brier"] is not None:
            audit[sport]["brier_delta_pooled_minus_solo"] = (
                pooled_on_sport["micro_classification"]["brier"] - solo["micro_classification"]["brier"]
            )
            audit[sport]["negative_transfer"] = audit[sport]["brier_delta_pooled_minus_solo"] > 0.005
        if solo["regression"]["mae"] is not None and pooled_on_sport["regression"]["mae"] is not None:
            audit[sport]["mae_delta_pooled_minus_solo"] = pooled_on_sport["regression"]["mae"] - solo["regression"]["mae"]
    return audit


def small_data_regime_test(base_config: TrainConfig, sport: str, fractions: list[float] = (1.0, 0.5, 0.25, 0.1)) -> dict:
    """Spec section 11.F / 45: does multi-sport pretraining help a small
    sport more as its OWN data becomes scarcer? Truncates `sport`'s DERIVE
    set to a fixed-seed random fraction (never touches SELECT/TEST) and
    trains sport-only at each fraction."""
    results = {}
    full_derive = UniversalDataset(split="DERIVE", sports=[sport])
    select = UniversalDataset(split="SELECT", sports=[sport])
    for frac in fractions:
        cfg = copy.deepcopy(base_config)
        cfg.name = f"{sport}_only_frac{frac}"
        derive = copy.copy(full_derive)
        derive.frame = full_derive.frame.sample(frac=frac, random_state=42).reset_index(drop=True)
        result = train_on(cfg, derive, select)
        results[f"frac_{frac}"] = {
            "n_rows": len(derive),
            "final_select_metrics": result["final_select_metrics"],
        }
    return results

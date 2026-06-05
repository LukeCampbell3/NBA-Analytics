#!/usr/bin/env python3
"""
Optimize the v9.3 selection policy inside the current architecture.

This does not train a new model. It searches:
- raw/calibrated probability blend alpha
- OVER edge threshold
- UNDER edge threshold
- uncertainty percentile cap

The primary report is walk-forward: each test month uses a policy selected
from the immediately preceding month, with calibration fitted only from rows
before that tuning month.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_prop_engine.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_prop_engine", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


V = _load_validator()


def _policy_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    policies = []
    for alpha in np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step):
        for over_edge in np.arange(args.over_edge_min, args.over_edge_max + 1e-9, args.edge_step):
            for under_edge in np.arange(args.under_edge_min, args.under_edge_max + 1e-9, args.edge_step):
                for max_uncertainty in np.arange(args.uncertainty_min, args.uncertainty_max + 1e-9, args.uncertainty_step):
                    policies.append(
                        {
                            "calibration_blend_alpha": round(float(alpha), 4),
                            "over_min_edge": round(float(over_edge), 4),
                            "under_min_edge": round(float(under_edge), 4),
                            "max_uncertainty_percentile": round(float(max_uncertainty), 4),
                        }
                    )
    return policies


def _prepare_window(all_rows: pd.DataFrame, manifest: dict, manifest_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    calibration_rows = all_rows[all_rows["date"] < start].copy()
    score_rows = all_rows[(all_rows["date"] >= start) & (all_rows["date"] <= end)].copy()
    if score_rows.empty or len(calibration_rows) < 500:
        return pd.DataFrame()
    prepared, _ = V._prepare_rows(score_rows, manifest, manifest_path, calibration_rows)
    return prepared


def _score_prepared_policy(rows: pd.DataFrame, policy: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        return rows.copy(), {"n": 0}
    gated = rows[
        (
            ((rows["side"] == "OVER") & (rows["edge"] >= policy["over_min_edge"]))
            | ((rows["side"] == "UNDER") & (rows["edge"] >= policy["under_min_edge"]))
        )
        & (rows["uncertainty"] <= policy["max_uncertainty_percentile"])
    ].copy()
    metrics = V._metrics(gated, "p_selected", outcome_col="selected_outcome")
    metrics["side_share_max"] = float(gated["side"].value_counts(normalize=True).max()) if len(gated) else 1.0
    metrics["market_share_max"] = float(gated["market"].value_counts(normalize=True).max()) if len(gated) else 1.0
    metrics["over_share"] = float((gated["side"] == "OVER").mean()) if len(gated) else 0.0
    metrics["under_share"] = float((gated["side"] == "UNDER").mean()) if len(gated) else 0.0
    return gated, metrics


def _score_policy(prepared: pd.DataFrame, policy: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if prepared.empty:
        return prepared.copy(), {"n": 0}
    rows = V._apply_calibration_blend(prepared, policy["calibration_blend_alpha"])
    return _score_prepared_policy(rows, policy)


def _passes_tuning_constraints(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        metrics.get("n", 0) >= args.min_tune_gated
        and metrics.get("ece", 1.0) <= args.max_tune_ece
        and metrics.get("side_share_max", 1.0) <= args.max_tune_side_share
        and metrics.get("market_share_max", 1.0) <= args.max_tune_market_share
    )


def _policy_score(metrics: dict[str, Any], args: argparse.Namespace) -> tuple[float, float, int]:
    coverage_penalty = max(0, args.target_tune_gated - int(metrics.get("n", 0))) * args.coverage_penalty
    ece_penalty = max(0.0, float(metrics.get("ece", 0.0)) - args.target_tune_ece) * args.ece_penalty
    return (
        float(metrics.get("brier", 1.0)) + coverage_penalty + ece_penalty,
        float(metrics.get("ece", 1.0)),
        -int(metrics.get("n", 0)),
    )


def _choose_policy(tune_rows: pd.DataFrame, policies: list[dict[str, float]], args: argparse.Namespace) -> dict[str, Any]:
    best: tuple[tuple[float, float, int], dict[str, float], dict[str, Any]] | None = None
    evaluated = 0
    passed = 0
    alpha_cache: dict[float, pd.DataFrame] = {}
    for policy in policies:
        alpha = policy["calibration_blend_alpha"]
        if alpha not in alpha_cache:
            alpha_cache[alpha] = V._apply_calibration_blend(tune_rows, alpha)
        _, metrics = _score_prepared_policy(alpha_cache[alpha], policy)
        evaluated += 1
        if not _passes_tuning_constraints(metrics, args):
            continue
        passed += 1
        score = _policy_score(metrics, args)
        if best is None or score < best[0]:
            best = (score, policy, metrics)
    if best is None:
        fallback = {
            "calibration_blend_alpha": args.fallback_alpha,
            "over_min_edge": args.fallback_over_edge,
            "under_min_edge": args.fallback_under_edge,
            "max_uncertainty_percentile": args.fallback_uncertainty,
        }
        _, metrics = _score_policy(tune_rows, fallback)
        return {"policy": fallback, "tune_metrics": metrics, "selection_status": "fallback", "evaluated": evaluated, "passed": passed}
    return {"policy": best[1], "tune_metrics": best[2], "selection_status": "optimized", "evaluated": evaluated, "passed": passed}


def _aggregate_gated(frames: list[pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        return {"n": 0}
    gated = pd.concat(frames, ignore_index=True)
    metrics = V._metrics(gated, "p_selected", outcome_col="selected_outcome")
    metrics["side_share"] = {str(k): float(v) for k, v in gated["side"].value_counts(normalize=True).to_dict().items()}
    metrics["market_share"] = {str(k): float(v) for k, v in gated["market"].value_counts(normalize=True).to_dict().items()}
    metrics["largest_player_share"] = float(gated["player"].value_counts(normalize=True).iloc[0]) if len(gated) else 1.0
    return metrics


def run_walk_forward(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = V._load_manifest(manifest_path)
    all_rows = V._load_rows(manifest, manifest_path)
    policies = _policy_grid(args)
    folds = []
    gated_frames = []
    fold_starts = pd.date_range(start=pd.Timestamp(args.start), end=pd.Timestamp(args.end), freq="MS")
    for fold_start in fold_starts:
        fold_end = min(fold_start + pd.offsets.MonthEnd(0), pd.Timestamp(args.end))
        tune_end = fold_start - pd.Timedelta(days=1)
        tune_start = tune_end.replace(day=1)
        tune_rows = _prepare_window(all_rows, manifest, manifest_path, tune_start, tune_end)
        test_rows = _prepare_window(all_rows, manifest, manifest_path, fold_start, fold_end)
        if tune_rows.empty or test_rows.empty:
            continue
        selected = _choose_policy(tune_rows, policies, args)
        gated, metrics = _score_policy(test_rows, selected["policy"])
        gated_frames.append(gated)
        folds.append(
            {
                "fold_start": str(fold_start.date()),
                "fold_end": str(fold_end.date()),
                "tune_start": str(tune_start.date()),
                "tune_end": str(tune_end.date()),
                "test_rows": int(len(test_rows)),
                "gated_rows": int(len(gated)),
                "selection_status": selected["selection_status"],
                "policy": selected["policy"],
                "tune_metrics": selected["tune_metrics"],
                "test_metrics": metrics,
                "grid_evaluated": selected["evaluated"],
                "grid_passed": selected["passed"],
            }
        )
    latest_tune_end = pd.Timestamp(args.end)
    latest_tune_start = latest_tune_end.replace(day=1)
    latest_tune_rows = _prepare_window(all_rows, manifest, manifest_path, latest_tune_start, latest_tune_end)
    next_policy = _choose_policy(latest_tune_rows, policies, args) if not latest_tune_rows.empty else None
    return {
        "mode": "walk_forward_policy_optimization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "date_range": f"{args.start}_to_{args.end}",
        "grid_size": len(policies),
        "folds": folds,
        "aggregate_gated": _aggregate_gated(gated_frames),
        "recommended_next_policy": next_policy,
        "constraints": {
            "min_tune_gated": args.min_tune_gated,
            "max_tune_ece": args.max_tune_ece,
            "max_tune_side_share": args.max_tune_side_share,
            "max_tune_market_share": args.max_tune_market_share,
            "target_tune_gated": args.target_tune_gated,
            "target_tune_ece": args.target_tune_ece,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize v9.3 selection/calibration policy")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.75)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--over-edge-min", type=float, default=0.08)
    parser.add_argument("--over-edge-max", type=float, default=0.18)
    parser.add_argument("--under-edge-min", type=float, default=0.14)
    parser.add_argument("--under-edge-max", type=float, default=0.30)
    parser.add_argument("--edge-step", type=float, default=0.01)
    parser.add_argument("--uncertainty-min", type=float, default=0.45)
    parser.add_argument("--uncertainty-max", type=float, default=0.90)
    parser.add_argument("--uncertainty-step", type=float, default=0.05)
    parser.add_argument("--min-tune-gated", type=int, default=200)
    parser.add_argument("--target-tune-gated", type=int, default=350)
    parser.add_argument("--max-tune-ece", type=float, default=0.05)
    parser.add_argument("--target-tune-ece", type=float, default=0.03)
    parser.add_argument("--max-tune-side-share", type=float, default=0.70)
    parser.add_argument("--max-tune-market-share", type=float, default=0.65)
    parser.add_argument("--coverage-penalty", type=float, default=0.00005)
    parser.add_argument("--ece-penalty", type=float, default=0.05)
    parser.add_argument("--fallback-alpha", type=float, default=1.0)
    parser.add_argument("--fallback-over-edge", type=float, default=0.10)
    parser.add_argument("--fallback-under-edge", type=float, default=0.22)
    parser.add_argument("--fallback-uncertainty", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_walk_forward(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"aggregate_gated": report["aggregate_gated"], "folds": len(report["folds"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

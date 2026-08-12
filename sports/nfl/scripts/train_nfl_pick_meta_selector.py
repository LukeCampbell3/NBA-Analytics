#!/usr/bin/env python3
"""Learn and validate an NFL-only outcome-aware pick policy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_selector import summarize_market_rows  # noqa: E402
from sports.nfl.predictions.pick_meta import (  # noqa: E402
    ADVANTAGE_CANDIDATES,
    ARTIFACT_TYPE,
    CONFIDENCE_CANDIDATES,
    MAXIMUM_PRICE_CANDIDATES,
    MINIMUM_PRICE_CANDIDATES,
    MODEL_VERSION,
    WEEKLY_CAP_CANDIDATES,
    apply_meta_policy,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recent-pool",
        type=Path,
        default=NFL_ROOT / "data/evaluation/recent_selector_pool_2025.csv",
    )
    parser.add_argument(
        "--stress-pools",
        type=Path,
        nargs="*",
        default=[
            NFL_ROOT / "data/evaluation/market_selector_pool_2021.csv",
            NFL_ROOT / "data/evaluation/market_selector_pool_2022.csv",
        ],
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_pick_meta_selector.joblib",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data/evaluation/pick_meta_backtest.json",
    )
    return parser.parse_args()


def evaluate(rows: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    return summarize_market_rows(apply_meta_policy(rows, **policy))


def select_policy(development: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    leaderboard: list[dict[str, Any]] = []
    for confidence in CONFIDENCE_CANDIDATES:
        for advantage in ADVANTAGE_CANDIDATES:
            for minimum_price in MINIMUM_PRICE_CANDIDATES:
                for maximum_price in MAXIMUM_PRICE_CANDIDATES:
                    for weekly_cap in WEEKLY_CAP_CANDIDATES:
                        policy = {
                            "minimum_side_probability": confidence,
                            "minimum_no_vig_advantage": advantage,
                            "minimum_price": minimum_price,
                            "maximum_price": maximum_price,
                            "weekly_cap": weekly_cap,
                        }
                        summary = evaluate(development, policy)
                        eligible = bool(
                            summary["graded_decisions"] >= 48
                            and summary["distinct_weeks"] >= 10
                            and summary.get("roi") is not None
                            and summary["roi"] > 0.0
                        )
                        leaderboard.append({"eligible": eligible, **policy, **summary})
    candidates = [row for row in leaderboard if row["eligible"]]
    if not candidates:
        raise RuntimeError("No recent NFL meta-policy candidate passed the development gate.")
    selected = max(
        candidates,
        key=lambda row: (
            row["hit_rate_wilson_95"][0],
            row["hit_rate"],
            row["roi"],
            -row["graded_decisions"],
        ),
    )
    policy_keys = (
        "minimum_side_probability",
        "minimum_no_vig_advantage",
        "minimum_price",
        "maximum_price",
        "weekly_cap",
    )
    return {key: selected[key] for key in policy_keys}, leaderboard


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _calibration_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return {
        "rows": int(len(actual)),
        "mean_confidence": round(float(clipped.mean()), 6),
        "realized_hit_rate": round(float(actual.mean()), 6),
        "calibration_gap": round(float(clipped.mean() - actual.mean()), 6),
        "brier_score": round(float(brier_score_loss(actual, clipped)), 6),
        "log_loss": round(float(log_loss(actual, clipped)), 6),
    }


def confidence_calibration_report(
    development: pd.DataFrame, locked: pd.DataFrame, policy: dict[str, Any]
) -> dict[str, Any]:
    selected_development = apply_meta_policy(development, **policy).copy()
    selected_locked = apply_meta_policy(locked, **policy).copy()
    selected_development["won"] = selected_development["result"].eq("win").astype(int)
    selected_locked["won"] = selected_locked["result"].eq("win").astype(int)
    raw = selected_development["estimated_side_probability"].to_numpy(dtype=float)
    actual = selected_development["won"].to_numpy(dtype=int)
    weeks = selected_development["week"].to_numpy(dtype=int)
    candidate_names = ("identity", "shrinkage_25", "platt", "beta", "isotonic", "offset")
    candidate_probabilities = {
        name: np.zeros(len(selected_development), dtype=float) for name in candidate_names
    }
    for week in sorted(set(weeks)):
        train = weeks != week
        validation = weeks == week
        candidate_probabilities["identity"][validation] = raw[validation]
        candidate_probabilities["shrinkage_25"][validation] = (
            0.75 * raw[validation] + 0.25 * actual[train].mean()
        )
        platt = LogisticRegression(C=0.1, max_iter=2_000, random_state=42).fit(
            _logit(raw[train]).reshape(-1, 1), actual[train]
        )
        candidate_probabilities["platt"][validation] = platt.predict_proba(
            _logit(raw[validation]).reshape(-1, 1)
        )[:, 1]
        beta_features = np.column_stack(
            [np.log(np.clip(raw, 1e-5, 1.0)), np.log(np.clip(1.0 - raw, 1e-5, 1.0))]
        )
        beta = LogisticRegression(C=0.1, max_iter=2_000, random_state=42).fit(
            beta_features[train], actual[train]
        )
        candidate_probabilities["beta"][validation] = beta.predict_proba(
            beta_features[validation]
        )[:, 1]
        isotonic = IsotonicRegression(
            y_min=0.05, y_max=0.95, out_of_bounds="clip"
        ).fit(raw[train], actual[train])
        candidate_probabilities["isotonic"][validation] = isotonic.predict(raw[validation])
        candidate_probabilities["offset"][validation] = np.clip(
            raw[validation] + actual[train].mean() - raw[train].mean(), 0.05, 0.95
        )
    candidates = [
        {"method": name, **_calibration_metrics(actual, candidate_probabilities[name])}
        for name in candidate_names
    ]
    candidates.sort(key=lambda row: (row["brier_score"], row["log_loss"]))
    selected_method = candidates[0]["method"]
    locked_actual = selected_locked["won"].to_numpy(dtype=int)
    locked_raw = selected_locked["estimated_side_probability"].to_numpy(dtype=float)
    full = pd.concat([selected_development, selected_locked], ignore_index=True)
    support = [
        round(float(full["estimated_side_probability"].min()) - 1e-6, 6),
        round(float(full["estimated_side_probability"].max()) + 1e-6, 6),
    ]
    return {
        "status": "passed" if selected_method == "identity" else "failed",
        "selected_method": selected_method,
        "selection_rule": "lowest grouped-development Brier score, then log loss",
        "development_grouped_comparison": candidates,
        "locked_identity_evaluation": _calibration_metrics(locked_actual, locked_raw),
        "full_recent_identity_evaluation": _calibration_metrics(
            full["result"].eq("win").astype(int).to_numpy(),
            full["estimated_side_probability"].to_numpy(dtype=float),
        ),
        "historical_support": support,
    }


def main() -> int:
    args = parse_args()
    if "mlb" in str(args.artifact).lower():
        raise ValueError("NFL meta-policy artifacts cannot be written to an MLB model path.")
    recent = pd.read_csv(args.recent_pool, low_memory=False)
    development = recent.loc[recent["week"].le(12)].copy()
    locked = recent.loc[recent["week"].ge(13)].copy()
    policy, leaderboard = select_policy(development)
    development_summary = evaluate(development, policy)
    locked_summary = evaluate(locked, policy)
    full_recent_summary = evaluate(recent, policy)
    calibration = confidence_calibration_report(development, locked, policy)
    stress_results = {
        path.stem: evaluate(pd.read_csv(path, low_memory=False), policy)
        for path in args.stress_pools
    }
    passed = bool(
        locked_summary["graded_decisions"] >= 30
        and locked_summary["distinct_weeks"] >= 6
        and locked_summary["hit_rate"] >= 0.60
        and locked_summary["hit_rate_wilson_95"][0] > 0.55
        and locked_summary["roi"] > 0.0
    )
    trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data_cutoff = pd.to_datetime(
        recent["commence_time_utc"], utc=True, errors="coerce"
    ).max()
    evidence_as_of = (
        data_cutoff.isoformat().replace("+00:00", "Z")
        if pd.notna(data_cutoff)
        else "2025-12-31T00:00:00Z"
    )
    report = {
        "schema_version": 2,
        "sport": "NFL",
        "artifact_type": ARTIFACT_TYPE,
        "model_version": MODEL_VERSION,
        "evidence_as_of_utc": evidence_as_of,
        "design": {
            "historical_seasons_available": [2021, 2022, 2025],
            "recent_development_period": "2025 weeks 1-12",
            "locked_recent_period": "2025 weeks 13-18",
            "locked_results_used_for_policy_selection": False,
            "identity_features_used": False,
            "policy_family_size": len(leaderboard),
            "eligible_policy_candidates": sum(row["eligible"] for row in leaderboard),
            "selection_metric": "development Wilson lower bound, hit rate, ROI, then coverage",
            "line_scope": "2025 explicit SportsGameOdds provider consensus closes",
            "book_execution_scope": "research only; consensus closes are not named-book execution proof",
        },
        "selected_policy": policy,
        "development": development_summary,
        "locked_recent_validation": {
            "status": "passed" if passed else "failed",
            **locked_summary,
        },
        "full_recent_season": full_recent_summary,
        "confidence_calibration": calibration,
        "older_stress_periods": stress_results,
        "deployment": {
            "status": "shadow_only",
            "prospective_certificate_active": False,
            "staking_enabled": False,
        },
        "leaderboard_top_25": sorted(
            leaderboard,
            key=lambda row: (
                bool(row["eligible"]),
                row["hit_rate_wilson_95"][0] if row["hit_rate_wilson_95"] else -1.0,
                row["hit_rate"] if row["hit_rate"] is not None else -1.0,
                row["roi"] if row["roi"] is not None else -1.0,
            ),
            reverse=True,
        )[:25],
    }
    artifact = {
        "schema_version": 1,
        "sport": "NFL",
        "artifact_type": ARTIFACT_TYPE,
        "model_version": MODEL_VERSION,
        "trained_at_utc": trained_at,
        "learned_from": "settled 2025 weeks 1-12 passing candidate wins and losses",
        "policy": policy,
        "confidence_calibration": {
            "method": calibration["selected_method"],
            "status": calibration["status"],
            "historical_support": calibration["historical_support"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    print(json.dumps({"policy": policy, "locked": report["locked_recent_validation"]}, indent=2))
    print(f"NFL-only meta-policy artifact: {args.artifact}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

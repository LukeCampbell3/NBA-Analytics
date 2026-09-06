#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.game_conditioned_moe import (
    EXPERT_NAMES,
    MODEL_VERSION,
    SCHEMA_VERSION,
    TARGETS,
    build_expert_state,
    logistic,
    logit,
)
from sports.mlb.advanced.schema import AdvancedCandidateContext
from sports.mlb.advanced.sequential_pa_model import simulate_hitter_market
from sports.mlb.scripts.backtest_sequential_pa_hitter_model import (
    DEFAULT_DATA_ROOT,
    brier,
    ece,
    finite,
    legacy_projection_probability,
    logloss,
    prior_batter_profile,
    prior_pitcher_proxy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "game_conditioned_hitter_moe_v2.json"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.md"

TARGET_SPECS = {
    "H": (0.5, "H"),
    "TB": (1.5, "TB"),
    "HR": (0.5, "HR"),
}


def _clip_probability(value: float) -> float:
    return max(1e-5, min(1.0 - 1e-5, float(value)))


def _row_has_legacy_signal(row: pd.Series, target: str) -> bool:
    return any(
        key in row.index and finite(row.get(key)) is not None
        for key in (f"{target}_market_gap", f"{target}_rolling_avg", f"Market_{target}")
    )


def _historical_prior(row: pd.Series, target: str, line: float, sequential_probability: float) -> tuple[float, str]:
    if _row_has_legacy_signal(row, target):
        _, prior = legacy_projection_probability(row, target, line)
        return _clip_probability(prior), "LEGACY_PROJECTION_PROXY"
    return _clip_probability(sequential_probability), "SEQUENTIAL_STRUCTURAL_FALLBACK"


def _collect_examples(data_root: Path, *, season: int, max_games: int, trials: int, min_history: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    games_collected = 0
    files = sorted(data_root.glob(f"*/{season}_processed_processed.csv"))
    for file_index, path in enumerate(files):
        if games_collected >= max_games:
            break
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or str(frame.iloc[0].get("Player_Type") or "").lower() != "hitter":
            continue
        frame["_date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
        frame = frame.loc[frame["_date"].notna()].sort_values("_date").reset_index(drop=True)
        if len(frame) <= min_history:
            continue
        player_name = str(frame.iloc[0].get("Player") or path.parent.name).replace("_", " ")
        player_id = int(finite(frame.iloc[-1].get("Player_MLBAM_ID"), file_index + 1) or file_index + 1)
        candidate_indices = list(range(min_history, len(frame)))
        if len(candidate_indices) > 10:
            candidate_indices = candidate_indices[-10:]
        for idx in candidate_indices:
            if games_collected >= max_games:
                break
            row = frame.iloc[idx]
            history = frame.iloc[:idx]
            as_of_date = row["_date"].date().isoformat()
            batter = prior_batter_profile(history, player_id=player_id, player_name=player_name, as_of_date=as_of_date)
            pitcher = prior_pitcher_proxy(row, as_of_date=as_of_date)
            batting_order = int(finite(row.get("Batting_Order"), 6) or 6)
            team_runs = finite(row.get("Team_Expected_Runs"), finite(row.get("Expected_Team_Runs")))
            context = AdvancedCandidateContext(
                game_id=str(row.get("Game_ID") or f"hist-{file_index}-{idx}"),
                run_date=as_of_date,
                batter=batter,
                pitcher=pitcher,
                direct_matchup=None,
                batting_order=batting_order,
                is_home=str(row.get("Is_Home") or "0").strip().lower() in {"1", "true", "yes"},
                team_expected_runs=team_runs,
                park_factor=float(finite(row.get("Park_Factor"), 1.0) or 1.0),
                defense_residual=0.0,
                defense_status="HISTORICAL_AVERAGE_CONTEXT_ONLY",
                data_freshness_status="FRESH",
                missing_components=(
                    "HISTORICAL_FULL_STATCAST_PITCH_CONTEXT_NOT_PRESERVED",
                    "HISTORICAL_XFIP_SIERA_PARTIAL",
                ),
                temperature_f=finite(row.get("Temperature_F"), finite(row.get("Temperature"))),
            )
            added = False
            for target in TARGETS:
                line, actual_column = TARGET_SPECS[target]
                actual = finite(row.get(actual_column))
                if actual is None:
                    continue
                sequential = simulate_hitter_market(context, target=target, market_line=line, trials=trials)
                prior, prior_source = _historical_prior(
                    row,
                    target,
                    line,
                    sequential.raw_structural_probability,
                )
                state = build_expert_state(
                    context,
                    sequential,
                    target=target,
                    pitch_compatibility_score=0.0,
                )
                examples.append(
                    {
                        "date": as_of_date,
                        "player": player_name,
                        "target": target,
                        "outcome": 1 if actual > line else 0,
                        "prior_probability": prior,
                        "prior_source": prior_source,
                        "sequential_probability": sequential.raw_structural_probability,
                        "sequential_uncertainty": sequential.uncertainty,
                        "evidence_strength": state.evidence_strength,
                        "features": {
                            name: float(state.effective_features[name])
                            for name in EXPERT_NAMES
                        },
                    }
                )
                added = True
            if added:
                games_collected += 1
    return examples


def _fit_target_parameters(train: list[dict[str, Any]], *, ridge: float):
    matrix = np.array(
        [[float(row["features"][name]) for name in EXPERT_NAMES] for row in train],
        dtype=float,
    )
    outcomes = np.array([float(row["outcome"]) for row in train], dtype=float)
    offsets = np.array([logit(float(row["prior_probability"])) for row in train], dtype=float)
    means = matrix.mean(axis=0) if len(matrix) else np.zeros(len(EXPERT_NAMES))
    scales = matrix.std(axis=0) if len(matrix) else np.ones(len(EXPERT_NAMES))
    scales = np.where(scales < 1e-6, 1.0, scales)
    standardized = (matrix - means) / scales

    def objective(params: np.ndarray) -> float:
        logits = offsets + float(params[0]) + standardized.dot(params[1:])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        loss = -np.mean(
            outcomes * np.log(probabilities)
            + (1.0 - outcomes) * np.log(1.0 - probabilities)
        )
        penalty = float(ridge) * float(np.sum(params[1:] ** 2)) / max(1, len(train))
        return float(loss + penalty)

    fitted = minimize(
        objective,
        np.zeros(1 + len(EXPERT_NAMES)),
        method="L-BFGS-B",
    )
    params = fitted.x if fitted.success else np.zeros(1 + len(EXPERT_NAMES))
    return (
        float(params[0]),
        {name: float(params[i + 1]) for i, name in enumerate(EXPERT_NAMES)},
        {name: float(means[i]) for i, name in enumerate(EXPERT_NAMES)},
        {name: float(scales[i]) for i, name in enumerate(EXPERT_NAMES)},
    )


def _predict(rows, *, intercept, coefficients, means, scales):
    predicted = []
    for row in rows:
        residual = float(intercept)
        for name in EXPERT_NAMES:
            residual += float(coefficients[name]) * (
                (float(row["features"][name]) - float(means[name]))
                / max(1e-6, abs(float(scales[name])))
            )
        residual = max(
            -0.35,
            min(
                0.35,
                residual
                * max(0.0, min(1.0, float(row["evidence_strength"]))),
            ),
        )
        probability = logistic(logit(float(row["prior_probability"])) + residual)
        predicted.append(
            {
                **row,
                "candidate_probability": probability,
                "residual_logit": residual,
            }
        )
    return predicted


def _metrics(rows, key):
    if not rows:
        return {"rows": 0, "brier": None, "log_loss": None, "ece": None}
    translated = [{"outcome": row["outcome"], key: row[key]} for row in rows]
    return {
        "rows": len(rows),
        "brier": brier(translated, key),
        "log_loss": logloss(translated, key),
        "ece": ece(translated, key),
    }


def _expanding_window_predictions(
    rows: list[dict[str, Any]],
    *,
    ridge: float,
    folds: int,
    min_train_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = sorted({str(row["date"]) for row in rows})
    if len(dates) < 4:
        return [], []

    first_eligible_index = None
    for index, date in enumerate(dates):
        train_rows = [row for row in rows if str(row["date"]) < date]
        if len(train_rows) >= min_train_rows:
            first_eligible_index = index
            break
    if first_eligible_index is None or first_eligible_index >= len(dates) - 1:
        return [], []

    validation_dates = dates[first_eligible_index:]
    fold_count = max(1, min(int(folds), len(validation_dates)))
    date_blocks = [
        [str(value) for value in block.tolist()]
        for block in np.array_split(np.array(validation_dates, dtype=object), fold_count)
        if len(block)
    ]

    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for fold_index, block in enumerate(date_blocks, start=1):
        first_date = block[0]
        last_date = block[-1]
        block_set = set(block)
        train = [row for row in rows if str(row["date"]) < first_date]
        validation = [row for row in rows if str(row["date"]) in block_set]
        if len(train) < min_train_rows or not validation:
            continue
        intercept, coefficients, means, scales = _fit_target_parameters(train, ridge=ridge)
        predicted = _predict(
            validation,
            intercept=intercept,
            coefficients=coefficients,
            means=means,
            scales=scales,
        )
        predictions.extend(predicted)
        prior = _metrics(predicted, "prior_probability")
        candidate = _metrics(predicted, "candidate_probability")
        both_improved = (
            candidate["brier"] < prior["brier"]
            and candidate["log_loss"] < prior["log_loss"]
        )
        diagnostics.append(
            {
                "fold": fold_index,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "validation_start": first_date,
                "validation_end": last_date,
                "prior_brier": prior["brier"],
                "candidate_brier": candidate["brier"],
                "prior_log_loss": prior["log_loss"],
                "candidate_log_loss": candidate["log_loss"],
                "both_improved": both_improved,
            }
        )
    return predictions, diagnostics


def _target_fit(rows, *, target, ridge, folds, min_train_rows):
    subset = [row for row in rows if row["target"] == target]
    predictions, fold_diagnostics = _expanding_window_predictions(
        subset,
        ridge=ridge,
        folds=folds,
        min_train_rows=min_train_rows,
    )
    if len(subset) < min_train_rows or len(predictions) < 10:
        return {
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in EXPERT_NAMES},
            "feature_means": {name: 0.0 for name in EXPERT_NAMES},
            "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
            "prior_legacy_weight": 0.72,
            "positive_authority": False,
            "validation": {
                "status": "INSUFFICIENT_EXPANDING_WINDOW_HOLDOUT",
                "train_rows": len(subset),
                "validation_rows": len(predictions),
                "folds": fold_diagnostics,
                "statistical_gate_passed": False,
            },
        }

    prior_metrics = _metrics(predictions, "prior_probability")
    candidate_metrics = _metrics(predictions, "candidate_probability")
    brier_improved = candidate_metrics["brier"] < prior_metrics["brier"]
    logloss_improved = candidate_metrics["log_loss"] < prior_metrics["log_loss"]
    fold_pass_count = sum(bool(fold["both_improved"]) for fold in fold_diagnostics)
    fold_pass_rate = fold_pass_count / max(1, len(fold_diagnostics))
    statistical_pass = (
        len(predictions) >= 50
        and brier_improved
        and logloss_improved
        and len(fold_diagnostics) >= 3
        and fold_pass_rate >= 0.60
    )

    intercept, coefficients, means, scales = _fit_target_parameters(subset, ridge=ridge)
    prior_sources: dict[str, int] = {}
    for row in subset:
        prior_sources[row["prior_source"]] = prior_sources.get(row["prior_source"], 0) + 1

    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "feature_means": means,
        "feature_scales": scales,
        "prior_legacy_weight": 0.72,
        "positive_authority": False,
        "validation": {
            "status": (
                "IMPROVED_DIAGNOSTIC_ONLY"
                if statistical_pass
                else "DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE"
            ),
            "fit_rows": len(subset),
            "validation_rows": len(predictions),
            "fold_count": len(fold_diagnostics),
            "folds_both_improved": fold_pass_count,
            "fold_pass_rate": fold_pass_rate,
            "folds": fold_diagnostics,
            "prior_sources": prior_sources,
            "prior_brier": prior_metrics["brier"],
            "candidate_brier": candidate_metrics["brier"],
            "prior_log_loss": prior_metrics["log_loss"],
            "candidate_log_loss": candidate_metrics["log_loss"],
            "prior_ece": prior_metrics["ece"],
            "candidate_ece": candidate_metrics["ece"],
            "brier_improved": brier_improved,
            "log_loss_improved": logloss_improved,
            "statistical_gate_passed": statistical_pass,
            "negative_authority_allowed": statistical_pass,
            "positive_authority_blocker": "FULL_EXACT_POINT_IN_TIME_ADVANCED_FEATURE_SNAPSHOTS_NOT_AVAILABLE_FOR_THIS_FIT",
        },
    }


def _markdown(payload):
    lines = [
        "# MLB Game-Conditioned Hitter MoE Validation",
        "",
        f"Model: `{payload['model_version']}`",
        "",
        f"Evidence class: `{payload['evidence_class']}`",
        "",
        "The model fits target-specific residuals in logit space around the legacy/structural prior. Every validation prediction comes from an expanding-window fit using strictly earlier dates.",
        "",
        "Positive publication authority remains disabled because this corpus does not preserve exact pregame snapshots of every advanced live feature. A target receives negative-only authority only if aggregate Brier and log-loss improve and at least 60% of expanding-window folds improve both.",
        "",
        "| Target | Fit rows | OOF rows | Folds pass | Prior Brier | Candidate Brier | Prior LogLoss | Candidate LogLoss | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in TARGETS:
        val = payload["targets"][target]["validation"]
        fmt = lambda value: "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {target} | {val.get('fit_rows', val.get('train_rows', 0))} | "
            f"{val.get('validation_rows', 0)} | "
            f"{val.get('folds_both_improved', 0)}/{val.get('fold_count', 0)} | "
            f"{fmt(val.get('prior_brier'))} | {fmt(val.get('candidate_brier'))} | "
            f"{fmt(val.get('prior_log_loss'))} | {fmt(val.get('candidate_log_loss'))} | "
            f"{val.get('status')} |"
        )
    lines += [
        "",
        "## Experts",
        "",
        "- strikeout/contact compatibility",
        "- contact quality / expected contact",
        "- power / total-base / home-run tail",
        "- specific defensive conversion residual",
        "- plate-appearance opportunity",
        "- starter-removal / bullpen transition",
        "",
        "Live production additionally uses exact-day Savant pitch-type matchup information, xFIP/SIERA when available, team scoring state, park/weather state, and support/uncertainty shrinkage. This historical fit is diagnostic initialization, not certification.",
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--max-games", type=int, default=300)
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=1.5)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _collect_examples(
        args.data_root,
        season=args.season,
        max_games=args.max_games,
        trials=args.trials,
        min_history=args.min_history,
    )
    targets = {
        target: _target_fit(
            rows,
            target=target,
            ridge=args.ridge,
            folds=args.folds,
            min_train_rows=args.min_train_rows,
        )
        for target in TARGETS
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "training_status": "FITTED_EXPANDING_WINDOW_RESIDUAL_MOE",
        "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        "season": args.season,
        "rows": len(rows),
        "games": len({(row["date"], row["player"]) for row in rows}),
        "max_abs_residual_logit": 0.35,
        "architecture": "global_residual_coefficients_x_game_specific_expert_activations",
        "prior": "legacy_probability_when_preserved_else_structural_probability; live blends no-vig market when available",
        "validation_design": "expanding_window_strictly_prior_dates",
        "targets": targets,
        "promotion_rule": "negative authority requires OOF Brier+log-loss improvement and >=60% fold pass; positive authority additionally requires exact point-in-time advanced-feature evidence",
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output_model.write_text(encoded, encoding="utf-8")
    args.output_report.write_text(encoded, encoding="utf-8")
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(args.output_model),
                "rows": len(rows),
                "games": payload["games"],
                "targets": {key: value["validation"] for key, value in targets.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

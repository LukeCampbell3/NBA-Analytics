#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.advanced.game_conditioned_moe import EXPERT_NAMES, MODEL_VERSION, SCHEMA_VERSION, TARGETS
from sports.mlb.scripts import fit_game_conditioned_hitter_moe as base

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "game_conditioned_hitter_moe_v2.json"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.md"

SELECTION_THRESHOLDS = {
    "H": (0.55, 0.58, 0.60, 0.65, 0.70),
    "TB": (0.40, 0.50, 0.55, 0.58, 0.60),
    "HR": (0.05, 0.10, 0.15, 0.20, 0.30),
}
TOP_FRACTIONS = (0.10, 0.20, 0.30)


def _game_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["date"]), str(row["player"])


def _cross_player_sample(rows: list[dict[str, Any]], max_games: int) -> list[dict[str, Any]]:
    """Take a deterministic sample across the full scanned corpus.

    The legacy fitter stops once its game limit is reached while walking sorted
    player paths, which can over-represent the first alphabetic players. This
    second-stage sampler scans broadly first and hash-ranks unique player-games.
    """
    keys = sorted({_game_key(row) for row in rows})
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{key[0]}|{key[1]}".encode("utf-8")).hexdigest(),
    )
    if max_games > 0:
        ranked = ranked[:max_games]
    keep = set(ranked)
    sampled = [row for row in rows if _game_key(row) in keep]
    return sorted(sampled, key=lambda row: (str(row["date"]), str(row["player"]), str(row["target"])))


def _apply_production_guard(
    rows: list[dict[str, Any]],
    *,
    candidate_gate_passed: bool,
    calibration_risk: float,
) -> list[dict[str, Any]]:
    """Replay live negative-only authority exactly enough for non-regression tests."""
    guarded: list[dict[str, Any]] = []
    for row in rows:
        prior = max(1e-5, min(1.0 - 1e-5, float(row["prior_probability"])))
        candidate = max(1e-5, min(1.0 - 1e-5, float(row["candidate_probability"])))
        if candidate_gate_passed:
            uncertainty = max(0.0, min(1.0, float(row.get("sequential_uncertainty", 0.0))))
            haircut = min(0.10, 0.035 * uncertainty + max(0.0, min(0.08, float(calibration_risk))))
            lower_bound = max(0.0, min(1.0, candidate - haircut))
            production = min(prior, lower_bound)
        else:
            lower_bound = prior
            production = prior
        guarded.append({**row, "production_probability": production, "production_lower_bound": lower_bound})
    return guarded


def _hit_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(sum(int(row["outcome"]) for row in rows) / len(rows))


def _selection_non_regression(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    threshold_checks: list[dict[str, Any]] = []
    supported_passes: list[bool] = []
    for threshold in SELECTION_THRESHOLDS[target]:
        prior_rows = [row for row in rows if float(row["prior_probability"]) >= threshold]
        prod_rows = [row for row in rows if float(row["production_probability"]) >= threshold]
        vetoed = [
            row for row in rows
            if float(row["prior_probability"]) >= threshold > float(row["production_probability"])
        ]
        prior_rate = _hit_rate(prior_rows)
        prod_rate = _hit_rate(prod_rows)
        supported = len(prior_rows) >= 30 and len(prod_rows) >= 15
        non_regressive = bool(
            supported
            and prior_rate is not None
            and prod_rate is not None
            and prod_rate + 1e-12 >= prior_rate
        )
        if supported:
            supported_passes.append(non_regressive)
        threshold_checks.append(
            {
                "threshold": threshold,
                "prior_selected": len(prior_rows),
                "production_selected": len(prod_rows),
                "vetoed": len(vetoed),
                "prior_hit_rate": prior_rate,
                "production_hit_rate": prod_rate,
                "vetoed_hit_rate": _hit_rate(vetoed),
                "supported": supported,
                "non_regressive": non_regressive if supported else None,
            }
        )

    rank_checks: list[dict[str, Any]] = []
    rank_passes: list[bool] = []
    for fraction in TOP_FRACTIONS:
        count = max(20, int(math.ceil(len(rows) * fraction)))
        if len(rows) < count:
            continue
        prior_top = sorted(rows, key=lambda row: float(row["prior_probability"]), reverse=True)[:count]
        prod_top = sorted(rows, key=lambda row: float(row["production_probability"]), reverse=True)[:count]
        prior_rate = _hit_rate(prior_top)
        prod_rate = _hit_rate(prod_top)
        non_regressive = bool(
            prior_rate is not None and prod_rate is not None and prod_rate + 1e-12 >= prior_rate
        )
        rank_passes.append(non_regressive)
        rank_checks.append(
            {
                "top_fraction": fraction,
                "count": count,
                "prior_hit_rate": prior_rate,
                "production_hit_rate": prod_rate,
                "non_regressive": non_regressive,
            }
        )

    passed = bool(supported_passes) and all(supported_passes) and bool(rank_passes) and all(rank_passes)
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "supported_threshold_count": len(supported_passes),
        "threshold_checks": threshold_checks,
        "rank_checks": rank_checks,
    }


def _target_fit(rows: list[dict[str, Any]], *, target: str, ridge: float, folds: int, min_train_rows: int) -> dict[str, Any]:
    subset = [row for row in rows if row["target"] == target]
    predictions, fold_diagnostics = base._expanding_window_predictions(
        subset,
        ridge=ridge,
        folds=folds,
        min_train_rows=min_train_rows,
    )
    if len(subset) < min_train_rows or len(predictions) < 50:
        return {
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in EXPERT_NAMES},
            "feature_means": {name: 0.0 for name in EXPERT_NAMES},
            "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
            "prior_legacy_weight": 0.72,
            "positive_authority": False,
            "validation": {
                "status": "INSUFFICIENT_NON_REGRESSION_HOLDOUT",
                "fit_rows": len(subset),
                "validation_rows": len(predictions),
                "candidate_improvement_gate_passed": False,
                "production_non_regression_gate_passed": False,
                "statistical_gate_passed": False,
                "negative_authority_allowed": False,
            },
        }

    prior_metrics = base._metrics(predictions, "prior_probability")
    candidate_metrics = base._metrics(predictions, "candidate_probability")
    fold_pass_count = sum(bool(fold["both_improved"]) for fold in fold_diagnostics)
    fold_pass_rate = fold_pass_count / max(1, len(fold_diagnostics))
    candidate_gate = bool(
        candidate_metrics["brier"] < prior_metrics["brier"]
        and candidate_metrics["log_loss"] < prior_metrics["log_loss"]
        and len(fold_diagnostics) >= 3
        and fold_pass_rate >= 0.60
    )

    calibration_risk = max(0.0, float(candidate_metrics["brier"]) - float(prior_metrics["brier"]))
    guarded = _apply_production_guard(
        predictions,
        candidate_gate_passed=candidate_gate,
        calibration_risk=calibration_risk,
    )
    production_metrics = base._metrics(guarded, "production_probability")
    selection = _selection_non_regression(guarded, target)
    probability_non_regression = bool(
        production_metrics["brier"] <= prior_metrics["brier"] + 1e-12
        and production_metrics["log_loss"] <= prior_metrics["log_loss"] + 1e-12
    )
    production_gate = bool(candidate_gate and probability_non_regression and selection["passed"])

    intercept, coefficients, means, scales = base._fit_target_parameters(subset, ridge=ridge)
    sources: dict[str, int] = {}
    for row in subset:
        source = str(row.get("prior_source") or "UNKNOWN")
        sources[source] = sources.get(source, 0) + 1

    if production_gate:
        status = "IMPROVED_AND_PICK_NON_REGRESSIVE_DIAGNOSTIC_ONLY"
    elif candidate_gate:
        status = "CANDIDATE_IMPROVED_BUT_PRODUCTION_NON_REGRESSION_FAILED"
    else:
        status = "DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE"

    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "feature_means": means,
        "feature_scales": scales,
        "prior_legacy_weight": 0.72,
        "positive_authority": False,
        "validation": {
            "status": status,
            "fit_rows": len(subset),
            "validation_rows": len(predictions),
            "fold_count": len(fold_diagnostics),
            "folds_both_improved": fold_pass_count,
            "fold_pass_rate": fold_pass_rate,
            "folds": fold_diagnostics,
            "prior_sources": sources,
            "prior_brier": prior_metrics["brier"],
            "candidate_brier": candidate_metrics["brier"],
            "production_brier": production_metrics["brier"],
            "prior_log_loss": prior_metrics["log_loss"],
            "candidate_log_loss": candidate_metrics["log_loss"],
            "production_log_loss": production_metrics["log_loss"],
            "prior_ece": prior_metrics["ece"],
            "candidate_ece": candidate_metrics["ece"],
            "production_ece": production_metrics["ece"],
            "candidate_improvement_gate_passed": candidate_gate,
            "production_probability_non_regression": probability_non_regression,
            "selection_non_regression": selection,
            "production_non_regression_gate_passed": production_gate,
            "statistical_gate_passed": production_gate,
            "negative_authority_allowed": production_gate,
            "positive_authority_blocker": "FULL_EXACT_POINT_IN_TIME_ADVANCED_FEATURE_SNAPSHOTS_NOT_AVAILABLE_FOR_THIS_FIT",
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Game-Conditioned Hitter MoE Non-Regression Validation",
        "",
        f"Model: `{payload['model_version']}`",
        "",
        f"Evidence: `{payload['evidence_class']}`",
        "",
        "This report requires the new residual model to beat the prior in rolling-origin probability scoring and to preserve or improve supported pick hit-rate slices after replaying the live negative-only authority rule.",
        "",
        "| Target | OOF | Folds pass | Prior Brier | Candidate | Production | Prior LL | Candidate | Production | Pick guard | Authority |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for target in TARGETS:
        val = payload["targets"][target]["validation"]
        fmt = lambda value: "n/a" if value is None else f"{float(value):.4f}"
        selection = val.get("selection_non_regression") or {}
        lines.append(
            f"| {target} | {val.get('validation_rows', 0)} | {val.get('folds_both_improved', 0)}/{val.get('fold_count', 0)} | "
            f"{fmt(val.get('prior_brier'))} | {fmt(val.get('candidate_brier'))} | {fmt(val.get('production_brier'))} | "
            f"{fmt(val.get('prior_log_loss'))} | {fmt(val.get('candidate_log_loss'))} | {fmt(val.get('production_log_loss'))} | "
            f"{selection.get('status', 'n/a')} | {val.get('negative_authority_allowed', False)} |"
        )
    lines += [
        "",
        "A target that fails any gate has zero production authority, so its production probability is the previous prior unchanged. Positive/bidirectional authority remains disabled until exact point-in-time locked or prospective advanced-feature evidence exists.",
        "",
        "No ROI claim is made because this processed-history replay does not preserve exact decision-time prices for every observation.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--scan-max-games", type=int, default=100000)
    parser.add_argument("--max-games", type=int, default=1200)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--min-history", type=int, default=20)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--min-train-rows", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=1.5)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scanned = base._collect_examples(
        args.data_root,
        season=args.season,
        max_games=args.scan_max_games,
        trials=args.trials,
        min_history=args.min_history,
    )
    rows = _cross_player_sample(scanned, args.max_games)
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
    game_keys = {_game_key(row) for row in rows}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "training_status": "FITTED_EXPANDING_WINDOW_RESIDUAL_MOE_NON_REGRESSION_GATED",
        "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        "season": args.season,
        "rows": len(rows),
        "games": len(game_keys),
        "players": len({player for _, player in game_keys}),
        "dates": len({date for date, _ in game_keys}),
        "scanned_rows": len(scanned),
        "scanned_games": len({_game_key(row) for row in scanned}),
        "max_abs_residual_logit": 0.35,
        "architecture": "global_residual_coefficients_x_game_specific_expert_activations",
        "prior": "legacy_probability_when_preserved_else_structural_probability; live blends no-vig market when available",
        "validation_design": "broad_scan_deterministic_cross_player_sample_plus_expanding_window_strictly_prior_dates",
        "targets": targets,
        "promotion_rule": "negative authority requires OOF Brier+log-loss lift, >=60% fold pass, guarded production probability non-regression, and supported pick threshold/rank hit-rate non-regression; positive authority additionally requires exact point-in-time advanced-feature evidence",
        "economic_evidence": {
            "roi_claim": False,
            "reason": "exact decision-time prices are not preserved for every processed-history observation",
        },
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output_model.write_text(encoded, encoding="utf-8")
    args.output_report.write_text(encoded, encoding="utf-8")
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "games": len(game_keys),
        "players": payload["players"],
        "dates": payload["dates"],
        "targets": {target: targets[target]["validation"] for target in TARGETS},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

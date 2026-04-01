#!/usr/bin/env python3
"""
Daily side-by-side monitor for unified shadow meta veto profiles.

Runs evaluate_cutoff_meta_append twice over the same window:
1) live challenger profile (default unified-veto-corr-score=1.25)
2) research comparator profile (default unified-veto-corr-score=999)

Writes per-profile evaluator outputs plus a combined comparison JSON/CSV.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def _corr_token(value: float) -> str:
    raw = f"{float(value):g}"
    return raw.replace("-", "m").replace(".", "p")


def _safe_metric(summary: dict[str, Any], variant: str, key: str) -> float:
    node = summary.get(variant, {})
    value = node.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(summary: dict[str, Any], variant: str, key: str) -> int:
    node = summary.get(variant, {})
    value = node.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily unified veto side-by-side monitor.")
    parser.add_argument(
        "--daily-runs-dir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs",
        help="Path to daily run folders (YYYYMMDD).",
    )
    parser.add_argument("--run-date", type=str, default=None, help="Reference run date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--start-date", type=str, default=None, help="Explicit monitor window start (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Explicit monitor window end (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=21, help="Inclusive lookback span used when start-date is omitted.")
    parser.add_argument("--board-size", type=int, default=12, help="Baseline edge board size.")
    parser.add_argument("--cutoff-rank-low", type=int, default=8, help="Inclusive lower rank for cutoff-band rows.")
    parser.add_argument("--cutoff-rank-high", type=int, default=18, help="Inclusive upper rank for cutoff-band rows.")
    parser.add_argument("--append-window", type=int, default=6, help="Append candidate window from B+1..B+window.")
    parser.add_argument("--min-train-resolved", type=int, default=20, help="Minimum resolved append-pool rows before training.")
    parser.add_argument("--model-uplift-threshold", type=float, default=0.03, help="Meta append uplift threshold.")
    parser.add_argument("--model-uplift-margin", type=float, default=0.015, help="Meta append top1-top2 uplift margin.")
    parser.add_argument("--gap-quantile-threshold", type=float, default=0.60, help="Cutoff-gap quantile threshold.")
    parser.add_argument("--max-corr-score", type=float, default=1.25, help="Meta append correlation ceiling.")
    parser.add_argument(
        "--unified-shadow-uplift-floor",
        type=float,
        default=-0.03,
        help="Unified veto floor for modeled uplift.",
    )
    parser.add_argument("--live-unified-veto-corr-score", type=float, default=1.25, help="Live unified veto corr score.")
    parser.add_argument("--research-unified-veto-corr-score", type=float, default=999.0, help="Research unified veto corr score.")
    parser.add_argument(
        "--unified-require-shallow-day",
        action="store_true",
        help="Forward shallow-day requirement to evaluator unified gate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for monitor artifacts. Defaults to daily_runs/<run_stamp>/shadow/unified_cutoff_meta_monitor.",
    )
    parser.add_argument(
        "--comparison-json-out",
        type=Path,
        default=None,
        help="Optional explicit path for combined comparison JSON.",
    )
    parser.add_argument(
        "--comparison-csv-out",
        type=Path,
        default=None,
        help="Optional explicit path for combined comparison CSV.",
    )
    parser.add_argument(
        "--skip-stage1-daily-model",
        action="store_true",
        help="Skip Stage-1 daily permission model evaluation.",
    )
    parser.add_argument(
        "--stage1-target-mode",
        type=str,
        default="best_pool_positive",
        choices=[
            "best_pool_positive",
            "shadow_improves_edge",
            "best_feasible_append_value",
            "top1_feasible_positive",
            "top1_feasible_value",
        ],
        help="Stage-1 daily permission target mode.",
    )
    parser.add_argument("--stage1-threshold", type=float, default=0.55, help="Stage-1 permission probability threshold.")
    parser.add_argument("--stage1-min-train-days", type=int, default=12, help="Stage-1 minimum labeled training days.")
    parser.add_argument(
        "--exclude-snapshot-modes",
        nargs="*",
        default=[],
        help="Optional snapshot modes to exclude from evaluation/training windows (for example: stale_fallback).",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for child evaluator runs.")
    return parser.parse_args()


def _run_eval(
    *,
    args: argparse.Namespace,
    profile_name: str,
    unified_veto_corr_score: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    output_dir: Path,
) -> dict[str, Any]:
    window_token = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    corr_token = _corr_token(unified_veto_corr_score)
    stem = f"cutoff_meta_append_{window_token}_b{int(args.board_size)}_{profile_name}_corr{corr_token}"

    dataset_out = output_dir / f"{stem}_dataset.csv"
    rows_out = output_dir / f"{stem}_rows.csv"
    daily_out = output_dir / f"{stem}_daily.csv"
    daily_context_out = output_dir / f"{stem}_daily_context.csv"
    abstain_out = output_dir / f"{stem}_abstain.csv"
    summary_out = output_dir / f"{stem}_summary.json"

    cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "evaluate_cutoff_meta_append.py"),
        "--daily-runs-dir",
        str(args.daily_runs_dir),
        "--start-date",
        start_date.strftime("%Y-%m-%d"),
        "--end-date",
        end_date.strftime("%Y-%m-%d"),
        "--board-size",
        str(int(args.board_size)),
        "--cutoff-rank-low",
        str(int(args.cutoff_rank_low)),
        "--cutoff-rank-high",
        str(int(args.cutoff_rank_high)),
        "--append-window",
        str(int(args.append_window)),
        "--min-train-resolved",
        str(int(args.min_train_resolved)),
        "--model-uplift-threshold",
        str(float(args.model_uplift_threshold)),
        "--model-uplift-margin",
        str(float(args.model_uplift_margin)),
        "--gap-quantile-threshold",
        str(float(args.gap_quantile_threshold)),
        "--max-corr-score",
        str(float(args.max_corr_score)),
        "--unified-veto-corr-score",
        str(float(unified_veto_corr_score)),
        "--unified-shadow-uplift-floor",
        str(float(args.unified_shadow_uplift_floor)),
        "--dataset-out",
        str(dataset_out),
        "--rows-out",
        str(rows_out),
        "--daily-out",
        str(daily_out),
        "--daily-context-out",
        str(daily_context_out),
        "--abstain-out",
        str(abstain_out),
        "--summary-out",
        str(summary_out),
    ]
    if args.unified_require_shallow_day:
        cmd.append("--unified-require-shallow-day")
    if args.exclude_snapshot_modes:
        cmd += ["--exclude-snapshot-modes", *[str(x) for x in args.exclude_snapshot_modes]]

    print("\n" + "=" * 96)
    print(f"RUN CUTOFF META MONITOR PROFILE [{profile_name}]")
    print("=" * 96)
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    stage1_paths: dict[str, str] = {}
    stage1_summary: dict[str, Any] = {}
    if not args.skip_stage1_daily_model:
        stage1_scored_out = output_dir / f"{stem}_stage1_daily_scored.csv"
        stage1_summary_out = output_dir / f"{stem}_stage1_daily_summary.json"
        stage1_cmd = [
            args.python,
            str(REPO_ROOT / "scripts" / "evaluate_stage1_daily_append_permission.py"),
            "--daily-context-csv",
            str(daily_context_out),
            "--target-mode",
            str(args.stage1_target_mode),
            "--threshold",
            str(float(args.stage1_threshold)),
            "--min-train-days",
            str(int(args.stage1_min_train_days)),
            "--out-scored-csv",
            str(stage1_scored_out),
            "--out-summary-json",
            str(stage1_summary_out),
        ]
        if args.exclude_snapshot_modes:
            stage1_cmd += ["--exclude-snapshot-modes", *[str(x) for x in args.exclude_snapshot_modes]]
        print("\n" + "-" * 96)
        print(f"RUN STAGE1 DAILY MODEL [{profile_name}]")
        print("-" * 96)
        print("Command:", " ".join(stage1_cmd))
        subprocess.run(stage1_cmd, cwd=REPO_ROOT, check=True)
        stage1_paths = {
            "stage1_scored": str(stage1_scored_out),
            "stage1_summary": str(stage1_summary_out),
        }
        stage1_summary = json.loads(stage1_summary_out.read_text(encoding="utf-8"))

    return {
        "profile_name": profile_name,
        "unified_veto_corr_score": float(unified_veto_corr_score),
        "summary": summary,
        "stage1_summary": stage1_summary,
        "paths": {
            "dataset": str(dataset_out),
            "rows": str(rows_out),
            "daily": str(daily_out),
            "daily_context": str(daily_context_out),
            "abstain": str(abstain_out),
            "summary": str(summary_out),
            **stage1_paths,
        },
    }


def _profile_row(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    profile_name = str(payload["profile_name"])
    veto_corr = float(payload["unified_veto_corr_score"])
    edge = summary.get("edge_baseline", {})
    shadow = summary.get("shadow_append_a1_p90_x1", {})
    unified = summary.get("unified_shadow_meta_x1", {})
    delta = summary.get("delta_unified_minus_edge", {})
    unified_diag = summary.get("unified_gate_diagnostics", {})
    stage1_summary = payload.get("stage1_summary", {}) if isinstance(payload.get("stage1_summary"), dict) else {}
    stage1_actionable = stage1_summary.get("metrics_actionable_days", {}) if isinstance(stage1_summary, dict) else {}

    return {
        "profile_name": profile_name,
        "unified_veto_corr_score": veto_corr,
        "edge_resolved": _safe_int(summary, "edge_baseline", "resolved"),
        "edge_hit_rate": _safe_metric(summary, "edge_baseline", "resolved_hit_rate"),
        "edge_ev_per_resolved": _safe_metric(summary, "edge_baseline", "ev_per_resolved"),
        "shadow_resolved": _safe_int(summary, "shadow_append_a1_p90_x1", "resolved"),
        "shadow_hit_rate": _safe_metric(summary, "shadow_append_a1_p90_x1", "resolved_hit_rate"),
        "shadow_ev_per_resolved": _safe_metric(summary, "shadow_append_a1_p90_x1", "ev_per_resolved"),
        "unified_resolved": _safe_int(summary, "unified_shadow_meta_x1", "resolved"),
        "unified_hit_rate": _safe_metric(summary, "unified_shadow_meta_x1", "resolved_hit_rate"),
        "unified_ev_per_resolved": _safe_metric(summary, "unified_shadow_meta_x1", "ev_per_resolved"),
        "delta_unified_vs_edge_hit_rate_pp": float(delta.get("delta_hit_rate_pp", 0.0) or 0.0),
        "delta_unified_vs_edge_ev_per_resolved": float(delta.get("delta_ev_per_resolved", 0.0) or 0.0),
        "unified_appended_rows_total": int(unified_diag.get("appended_rows_total", 0) or 0),
        "unified_appended_resolved_total": int(unified_diag.get("appended_resolved_total", 0) or 0),
        "unified_abstain_reasons": json.dumps(unified_diag.get("abstain_reasons", {}), sort_keys=True),
        "stage1_target_mode": str(stage1_summary.get("target_mode", "")) if stage1_summary else "",
        "stage1_effective_target_mode": str(stage1_summary.get("effective_target_mode", "")) if stage1_summary else "",
        "stage1_fallback_used": bool(stage1_summary.get("fallback_used", False)) if stage1_summary else False,
        "stage1_target_class_count_primary": int(stage1_summary.get("target_class_count_primary", 0) or 0) if stage1_summary else 0,
        "stage1_threshold": float(stage1_summary.get("threshold", np.nan)) if stage1_summary else np.nan,
        "stage1_rows_total": int(stage1_summary.get("rows_total", 0) or 0) if stage1_summary else 0,
        "stage1_rows_model_ready": int(stage1_summary.get("rows_model_ready", 0) or 0) if stage1_summary else 0,
        "stage1_actionable_labeled_days": int(stage1_actionable.get("labeled_days", 0) or 0) if stage1_summary else 0,
        "stage1_actionable_permit_days": int(stage1_actionable.get("permit_days", 0) or 0) if stage1_summary else 0,
        "stage1_actionable_precision": float(stage1_actionable.get("precision_on_permit", np.nan)) if stage1_summary else np.nan,
        "stage1_actionable_recall": float(stage1_actionable.get("recall_on_positive", np.nan)) if stage1_summary else np.nan,
        "stage1_actionable_false_permit_rate": float(stage1_actionable.get("false_permit_rate", np.nan)) if stage1_summary else np.nan,
        "stage1_actionable_veto_regret_rate": float(stage1_actionable.get("veto_regret_rate", np.nan)) if stage1_summary else np.nan,
        "stage1_actionable_permit_mean_value_delta": float(stage1_actionable.get("permit_mean_value_delta", np.nan)) if stage1_summary else np.nan,
        "summary_path": str(payload["paths"]["summary"]),
        "rows_path": str(payload["paths"]["rows"]),
        "daily_path": str(payload["paths"]["daily"]),
        "abstain_path": str(payload["paths"]["abstain"]),
        "daily_context_path": str(payload["paths"]["daily_context"]),
        "stage1_scored_path": str(payload["paths"].get("stage1_scored", "")),
        "stage1_summary_path": str(payload["paths"].get("stage1_summary", "")),
        "dataset_path": str(payload["paths"]["dataset"]),
    }


def main() -> None:
    args = parse_args()

    run_date = pd.Timestamp(args.run_date).normalize() if args.run_date else pd.Timestamp.now().normalize()
    if args.end_date:
        end_date = pd.Timestamp(args.end_date).normalize()
    else:
        end_date = run_date
    if args.start_date:
        start_date = pd.Timestamp(args.start_date).normalize()
    else:
        lookback = int(args.lookback_days)
        if lookback < 1:
            raise ValueError("--lookback-days must be >= 1 when --start-date is omitted.")
        start_date = (end_date - pd.Timedelta(days=lookback - 1)).normalize()
    if start_date > end_date:
        raise ValueError(f"Invalid window: start_date {start_date.date()} is after end_date {end_date.date()}.")

    run_stamp = run_date.strftime("%Y%m%d")
    default_output_dir = args.daily_runs_dir / run_stamp / "shadow" / "unified_cutoff_meta_monitor"
    output_dir = (args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    window_token = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    compare_json = args.comparison_json_out or (output_dir / f"cutoff_meta_monitor_compare_{window_token}_b{int(args.board_size)}.json")
    compare_csv = args.comparison_csv_out or (output_dir / f"cutoff_meta_monitor_compare_{window_token}_b{int(args.board_size)}.csv")

    live_payload = _run_eval(
        args=args,
        profile_name="live",
        unified_veto_corr_score=float(args.live_unified_veto_corr_score),
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )
    research_payload = _run_eval(
        args=args,
        profile_name="research",
        unified_veto_corr_score=float(args.research_unified_veto_corr_score),
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )

    live_summary = live_payload["summary"]
    research_summary = research_payload["summary"]
    live_unified = live_summary.get("unified_shadow_meta_x1", {})
    research_unified = research_summary.get("unified_shadow_meta_x1", {})

    comparison = {
        "run_date": run_date.strftime("%Y-%m-%d"),
        "window_start": start_date.strftime("%Y-%m-%d"),
        "window_end": end_date.strftime("%Y-%m-%d"),
        "lookback_days": int((end_date - start_date).days + 1),
        "board_size": int(args.board_size),
        "policy": {
            "live_unified_veto_corr_score": float(args.live_unified_veto_corr_score),
            "research_unified_veto_corr_score": float(args.research_unified_veto_corr_score),
            "unified_shadow_uplift_floor": float(args.unified_shadow_uplift_floor),
            "meta_max_corr_score": float(args.max_corr_score),
            "meta_fallback_enabled": False,
            "live_default_uses_research_corr": False,
            "stage1_daily_model_enabled": bool(not args.skip_stage1_daily_model),
            "stage1_target_mode": str(args.stage1_target_mode),
            "stage1_threshold": float(args.stage1_threshold),
            "stage1_min_train_days": int(args.stage1_min_train_days),
            "exclude_snapshot_modes": [str(x) for x in args.exclude_snapshot_modes],
        },
        "profiles": {
            "live": live_payload,
            "research": research_payload,
        },
        "research_minus_live_unified": {
            "delta_resolved": int(research_unified.get("resolved", 0) or 0) - int(live_unified.get("resolved", 0) or 0),
            "delta_wins": int(research_unified.get("wins", 0) or 0) - int(live_unified.get("wins", 0) or 0),
            "delta_losses": int(research_unified.get("losses", 0) or 0) - int(live_unified.get("losses", 0) or 0),
            "delta_hit_rate_pp": 100.0
            * (
                float(research_unified.get("resolved_hit_rate", 0.0) or 0.0)
                - float(live_unified.get("resolved_hit_rate", 0.0) or 0.0)
            ),
            "delta_ev_per_resolved": float(research_unified.get("ev_per_resolved", 0.0) or 0.0)
            - float(live_unified.get("ev_per_resolved", 0.0) or 0.0),
        },
        "comparison_outputs": {
            "json": str(compare_json),
            "csv": str(compare_csv),
        },
    }

    compare_rows = pd.DataFrame.from_records([_profile_row(live_payload), _profile_row(research_payload)])
    compare_json.parent.mkdir(parents=True, exist_ok=True)
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    compare_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    compare_rows.to_csv(compare_csv, index=False)

    print("\n" + "=" * 96)
    print("DAILY CUTOFF META MONITOR COMPLETE")
    print("=" * 96)
    print(f"Window:              {start_date.date()} -> {end_date.date()}")
    print(f"Board size:          {int(args.board_size)}")
    print(f"Live veto corr:      {float(args.live_unified_veto_corr_score)}")
    print(f"Research veto corr:  {float(args.research_unified_veto_corr_score)}")
    print(f"Output directory:    {output_dir}")
    print(f"Compare JSON:        {compare_json}")
    print(f"Compare CSV:         {compare_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backfill protocol for Stage-1 daily append-permission research.

Protocol steps:
1) Ensure daily run folders exist across a requested date range (policy frozen).
2) Build cutoff-meta evaluation artifacts including daily_context dataset.
3) Run Stage-1 threshold sweep over the resulting daily_context sample.
4) Emit sample-health diagnostics and sweep summary artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_RUNS_ROOT = REPO_ROOT / "model" / "analysis" / "daily_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 backfill protocol.")
    parser.add_argument("--start-date", type=str, required=True, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, required=True, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--board-size", type=int, default=12, help="Board size for cutoff-meta evaluation.")
    parser.add_argument("--cutoff-rank-low", type=int, default=8, help="Cutoff-band lower rank.")
    parser.add_argument("--cutoff-rank-high", type=int, default=18, help="Cutoff-band upper rank.")
    parser.add_argument("--append-window", type=int, default=6, help="Append window for cutoff-meta evaluation.")
    parser.add_argument("--min-train-resolved", type=int, default=20, help="Minimum resolved rows for cutoff-meta model fit.")
    parser.add_argument("--research-pool-min-agreement", type=float, default=1.0, help="Research pool min agreement for opportunity diagnostics.")
    parser.add_argument(
        "--research-corr-mode",
        type=str,
        default="percentile",
        choices=["auto", "none", "absolute", "percentile", "zscore"],
        help="Research corr feasibility mode for append diagnostics.",
    )
    parser.add_argument(
        "--research-pool-max-corr-score",
        type=float,
        default=None,
        help="Optional research pool corr ceiling for opportunity diagnostics.",
    )
    parser.add_argument(
        "--research-corr-percentile-max",
        type=float,
        default=0.25,
        help="Optional within-day corr percentile cap [0,1] for research diagnostics.",
    )
    parser.add_argument(
        "--research-corr-zscore-max",
        type=float,
        default=None,
        help="Optional within-day corr z-score cap for research diagnostics.",
    )
    parser.add_argument(
        "--stage1-target-mode",
        type=str,
        default="top1_feasible_positive",
        choices=[
            "best_pool_positive",
            "shadow_improves_edge",
            "best_feasible_append_value",
            "top1_feasible_positive",
            "top1_feasible_improves_edge",
            "top1_feasible_value",
        ],
    )
    parser.add_argument(
        "--stage1-candidate-mode",
        type=str,
        default="top1_feasible",
        choices=["shadow", "top1_feasible", "none"],
        help="Candidate availability gate used by Stage-1 permit decisions.",
    )
    parser.add_argument("--stage1-min-train-days", type=int, default=12, help="Stage-1 minimum labeled train days.")
    parser.add_argument("--stage1-thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60], help="Stage-1 permission thresholds.")
    parser.add_argument("--daily-runs-dir", type=Path, default=DAILY_RUNS_ROOT, help="Daily runs root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Protocol output directory.")
    parser.add_argument("--policy-profile", type=str, default="production_edge_b12", help="Frozen primary policy profile for backfill generation.")
    parser.add_argument(
        "--shadow-policy-profiles",
        nargs="*",
        default=["shadow_edge_append_agree1_p90_x1"],
        help="Frozen shadow policy profiles for backfill generation.",
    )
    parser.add_argument("--skip-generate-runs", action="store_true", help="Skip generating missing daily run folders.")
    parser.add_argument("--force-regenerate-existing", action="store_true", help="Regenerate days even when selector already exists.")
    parser.add_argument("--max-days", type=int, default=None, help="Optional cap on number of dates processed from the range.")
    parser.add_argument(
        "--exclude-snapshot-modes",
        nargs="*",
        default=["stale_fallback"],
        help="Snapshot modes to exclude from evaluation/training (default: stale_fallback).",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for child scripts.")
    return parser.parse_args()


def _infer_season(dt: pd.Timestamp) -> int:
    return dt.year + 1 if dt.month >= 9 else dt.year


def _date_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.date_range(start=start, end=end, freq="D")
    return [pd.Timestamp(d).normalize() for d in days]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
        if pd.isna(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def main() -> None:
    args = parse_args()
    start = pd.to_datetime(args.start_date, errors="coerce")
    end = pd.to_datetime(args.end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid start or end date.")
    if end < start:
        raise ValueError("end-date must be on/after start-date.")

    dates = _date_range(start.normalize(), end.normalize())
    if args.max_days is not None:
        dates = dates[: max(0, int(args.max_days))]
    if not dates:
        raise RuntimeError("No dates to process.")

    window_token = f"{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}"
    default_out = args.daily_runs_dir / f"stage1_backfill_protocol_{window_token}_b{int(args.board_size)}"
    out_dir = (args.output_dir or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generation_rows: list[dict[str, Any]] = []
    if not args.skip_generate_runs:
        for dt in dates:
            run_stamp = dt.strftime("%Y%m%d")
            run_dir = args.daily_runs_dir / run_stamp
            selector = run_dir / f"upcoming_market_play_selector_{run_stamp}.csv"
            exists = selector.exists()
            action = "skip_existing"
            status = "ok"
            if (not exists) or bool(args.force_regenerate_existing):
                action = "generate"
                season = _infer_season(dt)
                cmd = [
                    args.python,
                    "scripts/run_daily_market_pipeline.py",
                    "--run-date",
                    dt.strftime("%Y-%m-%d"),
                    "--season",
                    str(int(season)),
                    "--policy-profile",
                    str(args.policy_profile),
                    "--shadow-policy-profiles",
                    *[str(x) for x in args.shadow_policy_profiles],
                    "--skip-update-data",
                    "--skip-collect-market",
                    "--skip-align",
                    "--skip-backtest",
                    "--skip-cutoff-meta-monitor",
                    "--skip-export-web",
                    "--skip-build-site",
                ]
                try:
                    _run(cmd)
                except Exception as exc:  # pragma: no cover
                    status = "error"
                    generation_rows.append(
                        {
                            "run_date": dt.strftime("%Y-%m-%d"),
                            "run_stamp": run_stamp,
                            "action": action,
                            "status": status,
                            "selector_exists_after": selector.exists(),
                            "error": str(exc),
                        }
                    )
                    continue
            generation_rows.append(
                {
                    "run_date": dt.strftime("%Y-%m-%d"),
                    "run_stamp": run_stamp,
                    "action": action,
                    "status": status,
                    "selector_exists_after": selector.exists(),
                    "error": "",
                }
            )

    gen_df = pd.DataFrame.from_records(generation_rows)
    if not gen_df.empty:
        gen_df.to_csv(out_dir / "run_generation_log.csv", index=False)

    # Evaluate cutoff meta on the full requested window.
    dataset_out = out_dir / "cutoff_meta_dataset.csv"
    rows_out = out_dir / "cutoff_meta_rows.csv"
    daily_out = out_dir / "cutoff_meta_daily.csv"
    daily_context_out = out_dir / "cutoff_meta_daily_context.csv"
    abstain_out = out_dir / "cutoff_meta_abstain.csv"
    shadow_miss_out = out_dir / "cutoff_meta_shadow_top1_miss_reasons.csv"
    shadow_miss_summary_out = out_dir / "top1_shadow_miss_reasons_summary.json"
    stage2_table_out = out_dir / "stage2_proposal_table.csv"
    stage2_pairs_out = out_dir / "stage2_proposal_pairs.csv"
    stage2_summary_out = out_dir / "stage2_proposal_summary.json"
    summary_out = out_dir / "cutoff_meta_summary.json"

    eval_cmd = [
        args.python,
        str(REPO_ROOT / "scripts" / "evaluate_cutoff_meta_append.py"),
        "--daily-runs-dir",
        str(args.daily_runs_dir),
        "--start-date",
        dates[0].strftime("%Y-%m-%d"),
        "--end-date",
        dates[-1].strftime("%Y-%m-%d"),
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
        "--research-pool-min-agreement",
        str(float(args.research_pool_min_agreement)),
        "--research-corr-mode",
        str(args.research_corr_mode),
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
        "--shadow-top1-miss-out",
        str(shadow_miss_out),
        "--top1-shadow-miss-summary-out",
        str(shadow_miss_summary_out),
        "--stage2-proposal-table-out",
        str(stage2_table_out),
        "--stage2-proposal-pairs-out",
        str(stage2_pairs_out),
        "--stage2-proposal-summary-out",
        str(stage2_summary_out),
        "--summary-out",
        str(summary_out),
    ]
    if args.research_pool_max_corr_score is not None:
        eval_cmd += ["--research-pool-max-corr-score", str(float(args.research_pool_max_corr_score))]
    if args.research_corr_percentile_max is not None:
        eval_cmd += ["--research-corr-percentile-max", str(float(args.research_corr_percentile_max))]
    if args.research_corr_zscore_max is not None:
        eval_cmd += ["--research-corr-zscore-max", str(float(args.research_corr_zscore_max))]
    if args.exclude_snapshot_modes:
        eval_cmd += ["--exclude-snapshot-modes", *[str(x) for x in args.exclude_snapshot_modes]]
    _run(eval_cmd)

    if not daily_context_out.exists():
        raise RuntimeError("Expected daily_context output was not produced.")
    daily_context_df = pd.read_csv(daily_context_out)

    # Stage-1 threshold sweep.
    sweep_rows: list[dict[str, Any]] = []
    for threshold in [float(t) for t in args.stage1_thresholds]:
        token = str(threshold).replace(".", "p")
        scored_out = out_dir / f"stage1_scored_t{token}.csv"
        stage1_summary_out = out_dir / f"stage1_summary_t{token}.json"
        stage1_cmd = [
            args.python,
            str(REPO_ROOT / "scripts" / "evaluate_stage1_daily_append_permission.py"),
            "--daily-context-csv",
            str(daily_context_out),
            "--target-mode",
            str(args.stage1_target_mode),
            "--candidate-mode",
            str(args.stage1_candidate_mode),
            "--min-train-days",
            str(int(args.stage1_min_train_days)),
            "--threshold",
            str(float(threshold)),
            "--out-scored-csv",
            str(scored_out),
            "--out-summary-json",
            str(stage1_summary_out),
        ]
        if args.exclude_snapshot_modes:
            stage1_cmd += ["--exclude-snapshot-modes", *[str(x) for x in args.exclude_snapshot_modes]]
        _run(stage1_cmd)
        payload = json.loads(stage1_summary_out.read_text(encoding="utf-8"))
        actionable = payload.get("metrics_actionable_days", {})
        overall = payload.get("metrics_all_days", {})
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "target_mode": payload.get("target_mode"),
                "effective_target_mode": payload.get("effective_target_mode"),
                "candidate_mode": payload.get("candidate_mode"),
                "fallback_used": bool(payload.get("fallback_used", False)),
                "target_class_count_primary": int(payload.get("target_class_count_primary", 0) or 0),
                "rows_total": int(payload.get("rows_total", 0) or 0),
                "rows_model_ready": int(payload.get("rows_model_ready", 0) or 0),
                "rows_candidate_exists": int(payload.get("rows_candidate_exists", 0) or 0),
                "rows_shadow_candidate_exists": int(payload.get("rows_shadow_candidate_exists", 0) or 0),
                "actionable_labeled_days": int(actionable.get("labeled_days", 0) or 0),
                "actionable_permit_days": int(actionable.get("permit_days", 0) or 0),
                "actionable_precision": _safe_float(actionable.get("precision_on_permit")),
                "actionable_recall": _safe_float(actionable.get("recall_on_positive")),
                "actionable_false_permit_rate": _safe_float(actionable.get("false_permit_rate")),
                "actionable_veto_regret_rate": _safe_float(actionable.get("veto_regret_rate")),
                "actionable_permit_mean_value_delta": _safe_float(actionable.get("permit_mean_value_delta")),
                "all_labeled_days": int(overall.get("labeled_days", 0) or 0),
                "all_permit_days": int(overall.get("permit_days", 0) or 0),
                "all_false_permit_rate": _safe_float(overall.get("false_permit_rate")),
                "all_veto_regret_rate": _safe_float(overall.get("veto_regret_rate")),
                "summary_json": str(stage1_summary_out),
                "scored_csv": str(scored_out),
            }
        )

    sweep_df = pd.DataFrame.from_records(sweep_rows).sort_values("threshold")
    sweep_csv = out_dir / "stage1_threshold_sweep_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    # Dataset health diagnostics.
    def _nn(col: str) -> int:
        return int(pd.to_numeric(daily_context_df.get(col), errors="coerce").notna().sum())

    def _sum_bool(col: str) -> int:
        return int(pd.to_numeric(daily_context_df.get(col), errors="coerce").fillna(0).astype(bool).sum())

    health = {
        "window_start": dates[0].strftime("%Y-%m-%d"),
        "window_end": dates[-1].strftime("%Y-%m-%d"),
        "window_days_requested": int(len(dates)),
        "daily_context_rows": int(len(daily_context_df)),
        "model_ready_days": _sum_bool("model_ready"),
        "shadow_candidate_days": _sum_bool("shadow_candidate_exists"),
        "pool_resolved_days": int(pd.to_numeric(daily_context_df.get("pool_resolved_count"), errors="coerce").fillna(0).gt(0).sum()),
        "pool_feasible_days": int(pd.to_numeric(daily_context_df.get("pool_feasible_count"), errors="coerce").fillna(0).gt(0).sum()),
        "pool_feasible_resolved_days": int(pd.to_numeric(daily_context_df.get("pool_feasible_resolved_count"), errors="coerce").fillna(0).gt(0).sum()),
        "top1_feasible_days": int(pd.to_numeric(daily_context_df.get("top1_feasible_exists"), errors="coerce").fillna(0).astype(bool).sum()),
        "top1_feasible_resolved_days": int(pd.to_numeric(daily_context_df.get("top1_feasible_resolved"), errors="coerce").fillna(0).astype(bool).sum()),
        "top1_shadow_missed_positive_days": int((pd.to_numeric(daily_context_df.get("label_shadow_missed_positive_top1"), errors="coerce") == 1).sum()),
        "top1_shadow_generation_blocked_days": int(pd.to_numeric(daily_context_df.get("shadow_top1_generation_blocked"), errors="coerce").fillna(0).sum()),
        "top1_shadow_safety_blocked_days": int(pd.to_numeric(daily_context_df.get("shadow_top1_safety_blocked"), errors="coerce").fillna(0).sum()),
        "label_non_null": {
            "label_best_append_positive": _nn("label_best_append_positive"),
            "label_shadow_candidate_positive": _nn("label_shadow_candidate_positive"),
            "label_abstain_correct": _nn("label_abstain_correct"),
            "label_shadow_day_improves_edge": _nn("label_shadow_day_improves_edge"),
            "label_unified_day_improves_edge": _nn("label_unified_day_improves_edge"),
            "label_unified_veto_correct": _nn("label_unified_veto_correct"),
            "label_any_feasible_positive": _nn("label_any_feasible_positive"),
            "label_any_feasible_improves_edge": _nn("label_any_feasible_improves_edge"),
            "label_shadow_missed_feasible_positive": _nn("label_shadow_missed_feasible_positive"),
            "label_top1_feasible_positive": _nn("label_top1_feasible_positive"),
            "label_top1_feasible_improves_edge": _nn("label_top1_feasible_improves_edge"),
            "shadow_matches_top1_feasible": _nn("shadow_matches_top1_feasible"),
            "label_shadow_missed_positive_top1": _nn("label_shadow_missed_positive_top1"),
        },
        "label_balance": {
            "label_best_append_positive_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_best_append_positive"), errors="coerce").mean()),
            "label_shadow_day_improves_edge_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_shadow_day_improves_edge"), errors="coerce").mean()),
            "label_any_feasible_positive_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_any_feasible_positive"), errors="coerce").mean()),
            "label_any_feasible_improves_edge_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_any_feasible_improves_edge"), errors="coerce").mean()),
            "label_top1_feasible_positive_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_top1_feasible_positive"), errors="coerce").mean()),
            "label_top1_feasible_improves_edge_mean": _safe_float(pd.to_numeric(daily_context_df.get("label_top1_feasible_improves_edge"), errors="coerce").mean()),
        },
        "snapshot_mode_counts": daily_context_df.get("snapshot_mode", pd.Series(dtype="object")).fillna("").astype(str).value_counts(dropna=False).to_dict(),
        "excluded_snapshot_modes": [str(x) for x in args.exclude_snapshot_modes],
        "research_pool_config": {
            "min_agreement": float(args.research_pool_min_agreement),
            "corr_mode": str(args.research_corr_mode),
            "max_corr_score": (None if args.research_pool_max_corr_score is None else float(args.research_pool_max_corr_score)),
            "corr_percentile_max": (None if args.research_corr_percentile_max is None else float(args.research_corr_percentile_max)),
            "corr_zscore_max": (None if args.research_corr_zscore_max is None else float(args.research_corr_zscore_max)),
        },
        "stage1_config": {
            "target_mode": str(args.stage1_target_mode),
            "candidate_mode": str(args.stage1_candidate_mode),
            "thresholds": [float(t) for t in args.stage1_thresholds],
            "min_train_days": int(args.stage1_min_train_days),
        },
    }

    if not gen_df.empty:
        health["generation"] = {
            "rows": int(len(gen_df)),
            "generated_days": int((gen_df["action"] == "generate").sum()),
            "skipped_existing_days": int((gen_df["action"] == "skip_existing").sum()),
            "error_days": int((gen_df["status"] == "error").sum()),
            "selector_exists_after_days": int(pd.to_numeric(gen_df["selector_exists_after"], errors="coerce").fillna(0).astype(bool).sum()),
        }

    protocol_summary = {
        "protocol": {
            "policy_profile": str(args.policy_profile),
            "shadow_policy_profiles": [str(x) for x in args.shadow_policy_profiles],
            "live_policy_changed": False,
            "stage1_shadow_only": True,
        },
        "health": health,
        "artifacts": {
            "output_dir": str(out_dir),
            "run_generation_log": str(out_dir / "run_generation_log.csv") if not gen_df.empty else "",
            "cutoff_meta_dataset": str(dataset_out),
            "cutoff_meta_rows": str(rows_out),
            "cutoff_meta_daily": str(daily_out),
            "cutoff_meta_daily_context": str(daily_context_out),
            "cutoff_meta_abstain": str(abstain_out),
            "cutoff_meta_shadow_top1_miss_reasons": str(shadow_miss_out),
            "top1_shadow_miss_reasons_summary": str(shadow_miss_summary_out),
            "stage2_proposal_table": str(stage2_table_out),
            "stage2_proposal_pairs": str(stage2_pairs_out),
            "stage2_proposal_summary": str(stage2_summary_out),
            "cutoff_meta_summary": str(summary_out),
            "stage1_threshold_sweep_summary": str(sweep_csv),
        },
    }

    summary_json = out_dir / "stage1_backfill_protocol_summary.json"
    summary_json.write_text(json.dumps(protocol_summary, indent=2), encoding="utf-8")

    print("=" * 96)
    print("STAGE1 BACKFILL PROTOCOL COMPLETE")
    print("=" * 96)
    print(f"Window: {health['window_start']} -> {health['window_end']}")
    print(f"Requested days: {health['window_days_requested']}")
    print(f"Daily context rows: {health['daily_context_rows']}")
    print(f"Model-ready days: {health['model_ready_days']}")
    print(f"Shadow-candidate days: {health['shadow_candidate_days']}")
    print(f"Pool-resolved days: {health['pool_resolved_days']}")
    print(f"Stage1 sweep: {sweep_csv}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()

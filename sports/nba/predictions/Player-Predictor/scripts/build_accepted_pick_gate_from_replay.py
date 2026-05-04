#!/usr/bin/env python3
"""
Build an accepted-pick gate artifact from archived replay rows.

This wraps the existing research commands into one repeatable promotion check:
1) convert replay rows into accepted-pick history
2) train the shadow gate candidate
3) replay the candidate against the same archived rows in shadow mode
4) write a single summary that says promote_live or shadow_only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
SPORT_ROOT = PLAYER_PREDICTOR_ROOT.parents[1]
WORKSPACE_ROOT = SPORT_ROOT.parents[1]
VALIDATION_ROOT = WORKSPACE_ROOT / "sports" / "validation"
RESEARCH_SCRIPT = Path(__file__).resolve().with_name("run_accepted_pick_gate_research.py")
DEFAULT_OUTPUT_ROOT = PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "accepted_pick_gate"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out else default


def _mode_row_count(path: Path, mode: str) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "mode" not in (reader.fieldnames or []):
                return sum(1 for _ in reader)
            mode_token = str(mode or "").strip().lower()
            return sum(1 for row in reader if str(row.get("mode", "")).strip().lower() == mode_token)
    except Exception:
        return 0


def _latest_replay_rows(mode: str) -> Path:
    candidates = sorted(
        VALIDATION_ROOT.glob("robust_reranker_mode_compare_*_rows.csv"),
        key=lambda path: (_mode_row_count(path, mode), path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No robust reranker replay rows found. Expected sports/validation/"
            "robust_reranker_mode_compare_*_rows.csv"
        )
    selected = candidates[0]
    if _mode_row_count(selected, mode) <= 0:
        raise FileNotFoundError(f"No replay rows found for mode={mode!r} under {VALIDATION_ROOT}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accepted-pick gate artifacts from replay rows.")
    parser.add_argument("--rows-csv", type=Path, default=None, help="Archived replay rows CSV.")
    parser.add_argument("--mode", type=str, default="robust_reranker", help="Replay mode to train/evaluate.")
    parser.add_argument("--date-col", type=str, default="market_date")
    parser.add_argument("--result-col", type=str, default="result")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-train-rows", type=int, default=10)
    parser.add_argument("--min-test-rows", type=int, default=3)
    parser.add_argument("--train-window-days", type=int, default=21)
    parser.add_argument("--test-window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    rows_csv = (args.rows_csv or _latest_replay_rows(str(args.mode))).resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")

    mode_token = str(args.mode or "all").strip().lower().replace(" ", "_")
    output_root = args.output_root.resolve()
    history_csv = output_root / "history" / f"accepted_pick_history_replay_{mode_token}.csv"
    history_report = output_root / "history" / f"accepted_pick_history_replay_{mode_token}_build_report.json"
    model_json = output_root / "candidates" / f"accepted_pick_gate_candidate_{mode_token}.json"
    report_json = output_root / "reports" / f"accepted_pick_gate_report_{mode_token}.json"
    train_rows_csv = output_root / "reports" / f"accepted_pick_gate_scored_rows_{mode_token}.csv"
    paired_eval_json = output_root / "eval" / f"paired_eval_summary_{mode_token}.json"
    paired_eval_rows_csv = output_root / "eval" / f"paired_eval_scored_rows_{mode_token}.csv"
    summary_json = output_root / "reports" / f"accepted_pick_gate_replay_artifact_summary_{mode_token}.json"

    date_filters: list[str] = []
    if args.start_date:
        date_filters.extend(["--start-date", str(args.start_date)])
    if args.end_date:
        date_filters.extend(["--end-date", str(args.end_date)])

    prepare_command = [
        sys.executable,
        str(RESEARCH_SCRIPT),
        "prepare-replay-history",
        "--rows-csv",
        str(rows_csv),
        "--output-csv",
        str(history_csv),
        "--report-out",
        str(history_report),
        "--mode",
        str(args.mode),
        "--date-col",
        str(args.date_col),
        "--result-col",
        str(args.result_col),
        *date_filters,
    ]
    train_command = [
        sys.executable,
        str(RESEARCH_SCRIPT),
        "train-shadow",
        "--history-csv",
        str(history_csv),
        "--run-date-col",
        "market_date",
        "--model-family",
        "logistic",
        "--train-window-days",
        str(args.train_window_days),
        "--test-window-days",
        str(args.test_window_days),
        "--step-days",
        str(args.step_days),
        "--min-train-rows",
        str(args.min_train_rows),
        "--min-test-rows",
        str(args.min_test_rows),
        "--threshold-quantiles",
        "0.05,0.10,0.15,0.20,0.25,0.30",
        "--threshold-max-candidates",
        "16",
        "--recent-days",
        "14",
        "--rolling-window-days",
        "14",
        "--rolling-step-days",
        "7",
        "--lopo-min-player-rows",
        "4",
        "--bootstrap-iterations",
        str(args.bootstrap_iterations),
        "--model-out",
        str(model_json),
        "--report-out",
        str(report_json),
        "--scored-rows-out",
        str(train_rows_csv),
    ]
    paired_eval_command = [
        sys.executable,
        str(RESEARCH_SCRIPT),
        "paired-eval",
        "--rows-csv",
        str(rows_csv),
        "--gate-json",
        str(model_json),
        "--summary-out",
        str(paired_eval_json),
        "--scored-rows-out",
        str(paired_eval_rows_csv),
        "--mode",
        str(args.mode),
        "--date-col",
        str(args.date_col),
        "--result-col",
        str(args.result_col),
        "--short-days",
        "35",
        "--recent-days",
        "14",
        "--rolling-window-days",
        "14",
        "--rolling-step-days",
        "7",
        "--lopo-min-player-rows",
        "4",
        "--bootstrap-iterations",
        str(args.bootstrap_iterations),
        *date_filters,
    ]

    for command in (prepare_command, train_command, paired_eval_command):
        _run(command, dry_run=bool(args.dry_run))

    if args.dry_run:
        return

    history_payload = _read_json(history_report)
    model_payload = _read_json(model_json)
    train_payload = _read_json(report_json)
    paired_payload = _read_json(paired_eval_json)

    adaptive = model_payload.get("adaptive_live_control", {})
    oof_recommendation = model_payload.get("oof_promotion_recommendation", {})
    promotion = model_payload.get("promotion_recommendation", {})
    paired_recommendation = paired_payload.get("promotion_recommendation", {})
    model_live_ready = bool(
        adaptive.get("pass")
        and (oof_recommendation.get("pass") if isinstance(oof_recommendation, dict) else promotion.get("pass"))
    )
    paired_live_ready = bool(paired_recommendation.get("pass"))
    recommendation = "promote_live" if model_live_ready and paired_live_ready else "shadow_only"

    summary = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "recommendation": recommendation,
        "rows_csv": str(rows_csv),
        "mode": str(args.mode),
        "artifacts": {
            "history_csv": str(history_csv),
            "history_report_json": str(history_report),
            "model_json": str(model_json),
            "train_report_json": str(report_json),
            "train_scored_rows_csv": str(train_rows_csv),
            "paired_eval_json": str(paired_eval_json),
            "paired_eval_rows_csv": str(paired_eval_rows_csv),
        },
        "history": {
            "rows_written": int(history_payload.get("rows_written", 0) or 0),
            "rows_resolved": int(history_payload.get("rows_resolved", 0) or 0),
            "rows_win": int(history_payload.get("rows_win", 0) or 0),
            "rows_loss": int(history_payload.get("rows_loss", 0) or 0),
        },
        "model_live_ready": model_live_ready,
        "paired_live_ready": paired_live_ready,
        "model_adaptive_live_control": adaptive,
        "model_oof_promotion_pass": bool(oof_recommendation.get("pass")) if isinstance(oof_recommendation, dict) else None,
        "model_promotion_pass": bool(promotion.get("pass")) if isinstance(promotion, dict) else None,
        "paired_promotion_pass": bool(paired_recommendation.get("pass")),
        "paired_broad_profit_delta_units": _safe_float(
            paired_payload.get("broad_summary", {}).get("delta", {}).get("profit_units")
        ),
        "paired_recent_profit_delta_units": _safe_float(
            paired_payload.get("recent_summary", {}).get("delta", {}).get("profit_units")
        ),
        "paired_recent_hit_rate_delta_pp": _safe_float(
            paired_payload.get("recent_summary", {}).get("delta", {}).get("hit_rate_pp")
        ),
        "train_report_model_id": train_payload.get("model_id"),
        "notes": (
            "promote_live requires both the trained artifact adaptive/OOS controls and the deterministic "
            "paired replay evaluation to pass; otherwise production should keep the gate in shadow."
        ),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Replay artifact summary: {summary_json}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__":
    main()

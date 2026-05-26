#!/usr/bin/env python3
"""
Aggregate rebound diagnostics validation windows into a promotion-safe report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_rebound_diagnostics import (  # noqa: E402
    ACTIVE_WINDOW,
    BASELINE_VARIANT,
    MIXED_WINDOW,
    NO_OP_WINDOW,
    PROMOTION_TARGET_VARIANT,
    SEGMENTS,
    VARIANT_ORDER,
)


DEFAULT_BROADER_WINDOW_MIN_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate rebound diagnostics validation windows into a promotion gate report.")
    parser.add_argument(
        "--validation-summary-json",
        type=Path,
        action="append",
        required=True,
        help="Validation summary JSON emitted by validate_rebound_diagnostics.py. Pass multiple times.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_promotion_gate.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_promotion_gate.md",
    )
    parser.add_argument(
        "--variant-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_variant_compare.csv",
    )
    parser.add_argument(
        "--segments-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_segments.csv",
    )
    parser.add_argument("--broader-window-min-count", type=int, default=DEFAULT_BROADER_WINDOW_MIN_COUNT)
    return parser.parse_args()


def _safe_int(value: Any) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    return int(round(float(numeric)))


def _safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    return float(numeric)


def _flatten_summary_records(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    window = payload.get("window", {})
    records: list[dict[str, Any]] = []
    for row in payload.get("summary", []):
        out = dict(row)
        out["source_summary_json"] = str(source_path)
        out["window_start_run_date"] = str(window.get("start_run_date", ""))
        out["window_end_run_date"] = str(window.get("end_run_date", ""))
        records.append(out)
    return records


def _flatten_segment_records(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    window = payload.get("window", {})
    records: list[dict[str, Any]] = []
    for row in payload.get("segments", []):
        out = dict(row)
        out["source_summary_json"] = str(source_path)
        out["window_start_run_date"] = str(window.get("start_run_date", ""))
        out["window_end_run_date"] = str(window.get("end_run_date", ""))
        records.append(out)
    return records


def _flatten_window_reports(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    window = payload.get("window", {})
    records: list[dict[str, Any]] = []
    for row in payload.get("window_reports", []):
        no_op = dict(row.get("no_op_narrowness_validation", {}))
        active = dict(row.get("active_improvement_validation", {}))
        opposite = dict(row.get("opposite_under_discovery", {}))
        under_results = dict(opposite.get("under_candidate_results", {}))
        out = {
            "source_summary_json": str(source_path),
            "window_start_run_date": str(window.get("start_run_date", "")),
            "window_end_run_date": str(window.get("end_run_date", "")),
            "validation_mode": str(row.get("validation_mode", "")),
            "variant": str(row.get("variant", "")),
            "validation_window_type": str(row.get("validation_window_type", "")),
            "final_board_trb_over_count_baseline": _safe_int(row.get("final_board_trb_over_count_baseline")),
            "final_board_trb_over_count_full_diagnostics": _safe_int(row.get("final_board_trb_over_count_full_diagnostics")),
            "candidate_pool_trb_over_count": _safe_int(row.get("candidate_pool_trb_over_count")),
            "risky_trb_over_candidate_count": _safe_int(row.get("risky_trb_over_candidate_count")),
            "active_rebound_risk_present": bool(row.get("active_rebound_risk_present", False)),
            "no_op_day_count": _safe_int(row.get("no_op_day_count")),
            "active_day_count": _safe_int(row.get("active_day_count")),
            "status_label": str(row.get("status_label", "")),
            "no_op_narrowness_passed": bool(no_op.get("passed", False)),
            "no_op_narrowness_reason": str(no_op.get("reason", "")),
            "no_op_board_change_count": _safe_int(no_op.get("board_change_count")),
            "no_op_non_rebound_board_change_count": _safe_int(no_op.get("non_rebound_board_change_count")),
            "no_op_non_rebound_hit_rate_delta": _safe_float(no_op.get("non_rebound_hit_rate_delta")),
            "no_op_coverage_retained": _safe_float(no_op.get("coverage_retained")),
            "no_op_final_board_trb_over_count": _safe_int(no_op.get("final_board_trb_over_count")),
            "no_op_diagnostics_trigger_count": _safe_int(no_op.get("diagnostics_trigger_count")),
            "no_op_overtrigger_warning": bool(no_op.get("overtrigger_warning", False)),
            "active_improvement_passed": bool(active.get("passed", False)),
            "active_improvement_reason": str(active.get("reason", "")),
            "removed_trb_over_wins": _safe_int(active.get("removed_trb_over_wins")),
            "removed_trb_over_losses": _safe_int(active.get("removed_trb_over_losses")),
            "kept_trb_over_wins": _safe_int(active.get("kept_trb_over_wins")),
            "kept_trb_over_losses": _safe_int(active.get("kept_trb_over_losses")),
            "win_preservation_rate": _safe_float(active.get("win_preservation_rate")),
            "loss_removal_rate": _safe_float(active.get("loss_removal_rate")),
            "active_board_change_count": _safe_int(active.get("board_change_count")),
            "active_non_rebound_board_change_count": _safe_int(active.get("non_rebound_board_change_count")),
            "active_coverage_retained": _safe_float(active.get("coverage_retained")),
            "roi_delta": _safe_float(active.get("roi_delta")),
            "brier_delta": _safe_float(active.get("brier_delta")),
            "ece_delta": _safe_float(active.get("ece_delta")),
            "hit_rate_delta": _safe_float(active.get("hit_rate_delta")),
            "profit_units_delta": _safe_float(active.get("profit_units_delta")),
            "active_non_rebound_hit_rate_delta": _safe_float(active.get("non_rebound_hit_rate_delta")),
            "opposite_under_enabled": bool(opposite.get("enabled", False)),
            "opposite_under_flagged_over_count": _safe_int(opposite.get("flagged_over_count")),
            "synthetic_under_candidates_created": _safe_int(opposite.get("synthetic_under_candidates_created")),
            "under_candidates_with_valid_price": _safe_int(opposite.get("under_candidates_with_valid_price")),
            "under_candidates_passing_break_even": _safe_int(opposite.get("under_candidates_passing_break_even")),
            "under_candidates_added_to_board": _safe_int(opposite.get("under_candidates_added_to_board")),
            "under_candidates_rejected_price": _safe_int(opposite.get("under_candidates_rejected_price")),
            "under_candidates_rejected_forecastability": _safe_int(opposite.get("under_candidates_rejected_forecastability")),
            "under_candidates_rejected_stress": _safe_int(opposite.get("under_candidates_rejected_stress")),
            "under_candidate_resolved_picks": _safe_int(under_results.get("resolved_picks")),
            "under_candidate_wins": _safe_int(under_results.get("wins")),
            "under_candidate_losses": _safe_int(under_results.get("losses")),
            "under_candidate_pushes": _safe_int(under_results.get("pushes")),
            "under_candidate_hit_rate": _safe_float(under_results.get("hit_rate")),
            "under_candidate_profit_units": _safe_float(under_results.get("profit_units")),
            "under_candidate_roi": _safe_float(under_results.get("roi")),
            "added_under_rows": opposite.get("added_under_rows", []),
        }
        out["window_key"] = (
            f"{out['window_start_run_date']}:{out['window_end_run_date']}:"
            f"{out['validation_mode']}:{out['variant']}:{Path(source_path).name}"
        )
        records.append(out)
    return records


def _section_windows(window_reports: pd.DataFrame, column_name: str) -> dict[str, Any]:
    if window_reports.empty:
        return {
            "passed_window_count": 0,
            "failed_window_count": 0,
            "artifact_free_heuristic_passed": False,
            "trained_bundle_passed": False,
            "passed_windows": [],
            "failed_windows": [],
        }
    passed = window_reports.loc[window_reports[column_name].astype(bool)].copy()
    failed = window_reports.loc[~window_reports[column_name].astype(bool)].copy()
    return {
        "passed_window_count": int(len(passed)),
        "failed_window_count": int(len(failed)),
        "artifact_free_heuristic_passed": bool(
            passed["validation_mode"].astype(str).eq("artifact_free_heuristic").any()
        ),
        "trained_bundle_passed": bool(
            passed["validation_mode"].astype(str).eq("trained_bundle").any()
        ),
        "passed_windows": passed[
            ["window_key", "window_start_run_date", "window_end_run_date", "validation_mode", "validation_window_type", "status_label", "source_summary_json"]
        ].to_dict(orient="records"),
        "failed_windows": failed[
            ["window_key", "window_start_run_date", "window_end_run_date", "validation_mode", "validation_window_type", "status_label", "source_summary_json"]
        ].to_dict(orient="records"),
    }


def _aggregate_opposite_under(window_reports: pd.DataFrame) -> dict[str, Any]:
    if window_reports.empty:
        return {
            "enabled": False,
            "flagged_over_count": 0,
            "synthetic_under_candidates_created": 0,
            "under_candidates_with_valid_price": 0,
            "under_candidates_passing_break_even": 0,
            "under_candidates_added_to_board": 0,
            "under_candidates_rejected_price": 0,
            "under_candidates_rejected_forecastability": 0,
            "under_candidates_rejected_stress": 0,
            "under_candidate_results": {
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "hit_rate": np.nan,
                "roi": np.nan,
            },
            "added_under_rows": [],
        }
    added_under_rows: list[dict[str, Any]] = []
    for rows in window_reports["added_under_rows"].tolist():
        if isinstance(rows, list):
            added_under_rows.extend(rows)
    resolved = int(window_reports["under_candidate_resolved_picks"].sum())
    wins = int(window_reports["under_candidate_wins"].sum())
    losses = int(window_reports["under_candidate_losses"].sum())
    pushes = int(window_reports["under_candidate_pushes"].sum())
    profit_units = float(pd.to_numeric(window_reports["under_candidate_profit_units"], errors="coerce").fillna(0.0).sum())
    return {
        "enabled": bool(window_reports["opposite_under_enabled"].astype(bool).any()),
        "flagged_over_count": int(window_reports["opposite_under_flagged_over_count"].sum()),
        "synthetic_under_candidates_created": int(window_reports["synthetic_under_candidates_created"].sum()),
        "under_candidates_with_valid_price": int(window_reports["under_candidates_with_valid_price"].sum()),
        "under_candidates_passing_break_even": int(window_reports["under_candidates_passing_break_even"].sum()),
        "under_candidates_added_to_board": int(window_reports["under_candidates_added_to_board"].sum()),
        "under_candidates_rejected_price": int(window_reports["under_candidates_rejected_price"].sum()),
        "under_candidates_rejected_forecastability": int(window_reports["under_candidates_rejected_forecastability"].sum()),
        "under_candidates_rejected_stress": int(window_reports["under_candidates_rejected_stress"].sum()),
        "under_candidate_results": {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "hit_rate": float(wins / max(1, wins + losses)) if (wins + losses) > 0 else np.nan,
            "roi": float(profit_units / max(1, resolved)) if resolved > 0 else np.nan,
        },
        "added_under_rows": added_under_rows,
    }


def build_promotion_gate(window_reports: pd.DataFrame, *, broader_window_min_count: int) -> dict[str, Any]:
    target_reports = window_reports.loc[window_reports["variant"].astype(str) == PROMOTION_TARGET_VARIANT].copy()
    heuristic_reports = target_reports.loc[target_reports["validation_mode"].astype(str) == "artifact_free_heuristic"].copy()
    trained_reports = target_reports.loc[target_reports["validation_mode"].astype(str) == "trained_bundle"].copy()

    heuristic_no_op_pass = bool(heuristic_reports["no_op_narrowness_passed"].astype(bool).any()) if not heuristic_reports.empty else False
    heuristic_active_pass = bool(heuristic_reports["active_improvement_passed"].astype(bool).any()) if not heuristic_reports.empty else False
    trained_no_op_pass = bool(trained_reports["no_op_narrowness_passed"].astype(bool).any()) if not trained_reports.empty else False
    trained_active_pass = bool(trained_reports["active_improvement_passed"].astype(bool).any()) if not trained_reports.empty else False

    shadow_validated_logic = bool(heuristic_no_op_pass and heuristic_active_pass)
    trained_bundle_validated = bool(trained_no_op_pass and trained_active_pass)

    no_op_pass_windows = set(target_reports.loc[target_reports["no_op_narrowness_passed"].astype(bool), "window_key"].tolist())
    active_pass_windows = set(target_reports.loc[target_reports["active_improvement_passed"].astype(bool), "window_key"].tolist())
    broader_pass_windows = no_op_pass_windows | active_pass_windows
    broader_walk_forward_validated = bool(
        trained_bundle_validated
        and len(broader_pass_windows) >= int(broader_window_min_count)
        and bool(no_op_pass_windows)
        and bool(active_pass_windows)
    )

    blocked_reasons: list[str] = []
    if not heuristic_no_op_pass:
        blocked_reasons.append("no_op_narrowness_window_required")
    if not heuristic_active_pass:
        blocked_reasons.append("active_rebound_improvement_window_required")
    if not trained_bundle_validated:
        blocked_reasons.append("trained_bundle_replay_required")
    if not broader_walk_forward_validated:
        blocked_reasons.append("broader_walk_forward_required")
    if bool((target_reports["active_non_rebound_board_change_count"] > 0).any() or (target_reports["no_op_non_rebound_board_change_count"] > 0).any()):
        blocked_reasons.append("unexpected_non_rebound_board_changes")
    opposite_under_problem = bool(
        (
            (target_reports["under_candidates_added_to_board"] > target_reports["under_candidates_with_valid_price"])
            | (target_reports["under_candidates_added_to_board"] > target_reports["under_candidates_passing_break_even"])
        ).any()
    )
    if opposite_under_problem:
        blocked_reasons.append("opposite_under_audit_failed")

    promotion_ready = bool(shadow_validated_logic and trained_bundle_validated and broader_walk_forward_validated and not opposite_under_problem and "unexpected_non_rebound_board_changes" not in blocked_reasons)

    if promotion_ready:
        promotion_status_label = "promotion_candidate"
    elif shadow_validated_logic and not trained_bundle_validated:
        promotion_status_label = "trained_bundle_required"
    elif bool(target_reports["status_label"].astype(str).eq("rejected_overfit").any()):
        promotion_status_label = "rejected_overfit"
    else:
        promotion_status_label = "needs_more_sample"

    next_steps_map = {
        "no_op_narrowness_window_required": "Run a no-op narrowness replay window and confirm zero non-rebound board changes with >=95% coverage retained.",
        "active_rebound_improvement_window_required": "Run an active rebound-risk replay window and confirm more losing TRB_OVER removals than winning removals.",
        "trained_bundle_replay_required": "Replay the same rebound validation windows with the trained model bundle enabled.",
        "broader_walk_forward_required": "Run broader walk-forward validation across additional windows before promotion.",
        "unexpected_non_rebound_board_changes": "Investigate and eliminate non-rebound board changes before promotion.",
        "opposite_under_audit_failed": "Tighten opposite-side under discovery so only priced, break-even-positive candidates are added.",
    }
    required_next_steps = [next_steps_map[key] for key in blocked_reasons if key in next_steps_map]
    return {
        "shadow_validated_logic": shadow_validated_logic,
        "trained_bundle_validated": trained_bundle_validated,
        "broader_walk_forward_validated": broader_walk_forward_validated,
        "promotion_ready": promotion_ready,
        "promotion_status_label": promotion_status_label,
        "blocked_reason": blocked_reasons,
        "required_next_steps": required_next_steps,
        "artifact_free_heuristic_no_op_passed": heuristic_no_op_pass,
        "artifact_free_heuristic_active_passed": heuristic_active_pass,
        "trained_bundle_no_op_passed": trained_no_op_pass,
        "trained_bundle_active_passed": trained_active_pass,
        "broader_pass_window_count": int(len(broader_pass_windows)),
        "broader_window_min_count": int(broader_window_min_count),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame.loc[:, columns].copy()
    headers = columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        rendered = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if np.isnan(value):
                    rendered.append("")
                else:
                    rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _build_markdown_report(
    promotion_gate: dict[str, Any],
    window_reports: pd.DataFrame,
    variant_df: pd.DataFrame,
    opposite_under: dict[str, Any],
) -> str:
    target_reports = window_reports.loc[window_reports["variant"].astype(str) == PROMOTION_TARGET_VARIANT].copy()
    lines: list[str] = []
    lines.append("# Rebound Diagnostics Promotion Gate")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- `shadow_validated_logic`: `{promotion_gate['shadow_validated_logic']}`")
    lines.append(f"- `trained_bundle_validated`: `{promotion_gate['trained_bundle_validated']}`")
    lines.append(f"- `broader_walk_forward_validated`: `{promotion_gate['broader_walk_forward_validated']}`")
    lines.append(f"- `promotion_ready`: `{promotion_gate['promotion_ready']}`")
    lines.append(f"- `promotion_status_label`: `{promotion_gate['promotion_status_label']}`")
    lines.append("")
    lines.append("Blocked reasons:")
    for reason in promotion_gate["blocked_reason"]:
        lines.append(f"- `{reason}`")
    lines.append("")
    lines.append("Next steps:")
    for step in promotion_gate["required_next_steps"]:
        lines.append(f"- {step}")
    lines.append("")

    lines.append("## No-Op Narrowness")
    lines.append("")
    no_op_rows = target_reports.loc[
        target_reports["validation_window_type"].astype(str).isin([NO_OP_WINDOW, MIXED_WINDOW])
    ].copy()
    lines.append(
        _markdown_table(
            no_op_rows,
            [
                "window_start_run_date",
                "window_end_run_date",
                "validation_mode",
                "validation_window_type",
                "no_op_narrowness_passed",
                "no_op_board_change_count",
                "no_op_non_rebound_board_change_count",
                "no_op_coverage_retained",
                "no_op_diagnostics_trigger_count",
                "status_label",
            ],
        )
    )
    lines.append("")
    lines.append("## Active Improvement")
    lines.append("")
    active_rows = target_reports.loc[
        target_reports["validation_window_type"].astype(str).isin([ACTIVE_WINDOW, MIXED_WINDOW])
    ].copy()
    lines.append(
        _markdown_table(
            active_rows,
            [
                "window_start_run_date",
                "window_end_run_date",
                "validation_mode",
                "validation_window_type",
                "active_improvement_passed",
                "removed_trb_over_losses",
                "removed_trb_over_wins",
                "roi_delta",
                "brier_delta",
                "ece_delta",
                "active_non_rebound_board_change_count",
                "status_label",
            ],
        )
    )
    lines.append("")

    lines.append("## Variant Comparison")
    lines.append("")
    variant_preview = variant_df[
        [
            "window_start_run_date",
            "window_end_run_date",
            "validation_mode",
            "variant",
            "hit_rate",
            "roi",
            "coverage_retained",
            "board_change_count",
            "non_rebound_board_change_count",
            "final_board_trb_over_count",
            "final_board_trb_under_count",
            "status_label",
        ]
    ].copy()
    lines.append(_markdown_table(variant_preview, list(variant_preview.columns)))
    lines.append("")

    lines.append("## Opposite Under Audit")
    lines.append("")
    lines.append(f"- `flagged_over_count`: `{opposite_under['flagged_over_count']}`")
    lines.append(f"- `synthetic_under_candidates_created`: `{opposite_under['synthetic_under_candidates_created']}`")
    lines.append(f"- `under_candidates_with_valid_price`: `{opposite_under['under_candidates_with_valid_price']}`")
    lines.append(f"- `under_candidates_passing_break_even`: `{opposite_under['under_candidates_passing_break_even']}`")
    lines.append(f"- `under_candidates_added_to_board`: `{opposite_under['under_candidates_added_to_board']}`")
    lines.append(f"- `under_candidates_rejected_price`: `{opposite_under['under_candidates_rejected_price']}`")
    lines.append(f"- `under_candidates_rejected_forecastability`: `{opposite_under['under_candidates_rejected_forecastability']}`")
    lines.append(f"- `under_candidates_rejected_stress`: `{opposite_under['under_candidates_rejected_stress']}`")
    lines.append("")
    if opposite_under["added_under_rows"]:
        lines.append(_markdown_table(pd.DataFrame(opposite_under["added_under_rows"]), ["player", "game_date", "line", "odds", "reason", "result"]))
    else:
        lines.append("_No synthetic under rows were added to the board._")
    lines.append("")

    lines.append("## Promotion Status")
    lines.append("")
    lines.append(f"- `promotion_ready`: `{promotion_gate['promotion_ready']}`")
    lines.append(f"- `promotion_status_label`: `{promotion_gate['promotion_status_label']}`")
    lines.append("")
    lines.append("Required next validation:")
    for step in promotion_gate["required_next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payloads: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for path in args.validation_summary_json:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        payloads.append(payload)
        summary_rows.extend(_flatten_summary_records(payload, resolved))
        segment_rows.extend(_flatten_segment_records(payload, resolved))
        window_rows.extend(_flatten_window_reports(payload, resolved))

    variant_df = pd.DataFrame.from_records(summary_rows)
    segment_df = pd.DataFrame.from_records(segment_rows)
    window_reports_df = pd.DataFrame.from_records(window_rows)
    opposite_under = _aggregate_opposite_under(
        window_reports_df.loc[window_reports_df["variant"].astype(str) == PROMOTION_TARGET_VARIANT].copy()
        if not window_reports_df.empty
        else pd.DataFrame()
    )
    promotion_gate = build_promotion_gate(
        window_reports_df,
        broader_window_min_count=int(args.broader_window_min_count),
    )
    no_op_validation = _section_windows(
        window_reports_df.loc[window_reports_df["variant"].astype(str) == PROMOTION_TARGET_VARIANT].copy(),
        "no_op_narrowness_passed",
    )
    active_validation = _section_windows(
        window_reports_df.loc[window_reports_df["variant"].astype(str) == PROMOTION_TARGET_VARIANT].copy(),
        "active_improvement_passed",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summaries": [str(path.resolve()) for path in args.validation_summary_json],
        "variant_summaries": variant_df.to_dict(orient="records"),
        "segment_report": segment_df.to_dict(orient="records"),
        "window_reports": window_reports_df.to_dict(orient="records"),
        "no_op_narrowness_validation": no_op_validation,
        "active_improvement_validation": active_validation,
        "opposite_under_discovery": opposite_under,
        "promotion_gate": promotion_gate,
        "blocked_reason": promotion_gate["blocked_reason"],
        "required_next_steps": promotion_gate["required_next_steps"],
    }

    markdown = _build_markdown_report(promotion_gate, window_reports_df, variant_df, opposite_under)

    for path in [args.out_json, args.out_md, args.variant_csv_out, args.segments_csv_out]:
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out_json.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.resolve().write_text(markdown, encoding="utf-8")
    variant_df.to_csv(args.variant_csv_out.resolve(), index=False)
    segment_df.to_csv(args.segments_csv_out.resolve(), index=False)

    print("REBOUND DIAGNOSTICS PROMOTION GATE")
    print(f"Promotion status:      {promotion_gate['promotion_status_label']}")
    print(f"Shadow validated:      {promotion_gate['shadow_validated_logic']}")
    print(f"Trained bundle ready:  {promotion_gate['trained_bundle_validated']}")
    print(f"Broader walk-forward:  {promotion_gate['broader_walk_forward_validated']}")
    print(f"Promotion ready:       {promotion_gate['promotion_ready']}")
    print(f"Output JSON:           {args.out_json.resolve()}")
    print(f"Output Markdown:       {args.out_md.resolve()}")
    print(f"Variant CSV:           {args.variant_csv_out.resolve()}")
    print(f"Segments CSV:          {args.segments_csv_out.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import write_json


DEFAULT_VARIANT_ORDER = [
    "baseline_no_rebound_diagnostics",
    "upper_band_only",
    "full_rebound_diagnostics",
    "full_rebound_diagnostics_plus_opposite_under",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize intervention validation artifacts into a general report format.")
    parser.add_argument("--intervention-family", type=str, required=True)
    parser.add_argument("--intervention-id", type=str, required=True)
    parser.add_argument("--failure-mode-id", type=str, default="")
    parser.add_argument("--validation-summary-json", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--variant-csv-out", type=Path, required=True)
    parser.add_argument("--window-csv-out", type=Path, required=True)
    parser.add_argument("--segments-csv-out", type=Path, required=True)
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


def _flatten_variant_rows(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    window = payload.get("window", {})
    records: list[dict[str, Any]] = []
    for row in payload.get("summary", []):
        out = dict(row)
        out["source_summary_json"] = str(source_path)
        out["window_start_run_date"] = str(window.get("start_run_date", ""))
        out["window_end_run_date"] = str(window.get("end_run_date", ""))
        records.append(out)
    return records


def _flatten_segment_rows(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    window = payload.get("window", {})
    records: list[dict[str, Any]] = []
    for row in payload.get("segments", []):
        out = dict(row)
        out["source_summary_json"] = str(source_path)
        out["window_start_run_date"] = str(window.get("start_run_date", ""))
        out["window_end_run_date"] = str(window.get("end_run_date", ""))
        records.append(out)
    return records


def _flatten_window_rows(payload: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
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
            "status_label": str(row.get("status_label", "")),
            "active_rebound_risk_present": bool(row.get("active_rebound_risk_present", False)),
            "final_board_trb_over_count_baseline": _safe_int(row.get("final_board_trb_over_count_baseline")),
            "final_board_trb_over_count_full_diagnostics": _safe_int(row.get("final_board_trb_over_count_full_diagnostics")),
            "candidate_pool_trb_over_count": _safe_int(row.get("candidate_pool_trb_over_count")),
            "risky_trb_over_candidate_count": _safe_int(row.get("risky_trb_over_candidate_count")),
            "no_op_narrowness_passed": bool(no_op.get("passed", False)),
            "no_op_reason": str(no_op.get("reason", "")),
            "no_op_board_change_count": _safe_int(no_op.get("board_change_count")),
            "no_op_non_target_board_change_count": _safe_int(no_op.get("non_rebound_board_change_count")),
            "no_op_non_target_hit_rate_delta": _safe_float(no_op.get("non_rebound_hit_rate_delta")),
            "no_op_coverage_retained": _safe_float(no_op.get("coverage_retained")),
            "no_op_final_target_count": _safe_int(no_op.get("final_board_trb_over_count")),
            "no_op_diagnostics_trigger_count": _safe_int(no_op.get("diagnostics_trigger_count")),
            "no_op_overtrigger_warning": bool(no_op.get("overtrigger_warning", False)),
            "active_improvement_passed": bool(active.get("passed", False)),
            "active_reason": str(active.get("reason", "")),
            "removed_wins": _safe_int(active.get("removed_trb_over_wins", active.get("removed_wins"))),
            "removed_losses": _safe_int(active.get("removed_trb_over_losses", active.get("removed_losses"))),
            "kept_wins": _safe_int(active.get("kept_trb_over_wins", active.get("kept_wins"))),
            "kept_losses": _safe_int(active.get("kept_trb_over_losses", active.get("kept_losses"))),
            "win_preservation_rate": _safe_float(active.get("win_preservation_rate")),
            "loss_removal_rate": _safe_float(active.get("loss_removal_rate")),
            "active_board_change_count": _safe_int(active.get("board_change_count")),
            "active_non_target_board_change_count": _safe_int(active.get("non_rebound_board_change_count")),
            "active_coverage_retained": _safe_float(active.get("coverage_retained")),
            "roi_delta": _safe_float(active.get("roi_delta")),
            "brier_delta": _safe_float(active.get("brier_delta")),
            "ece_delta": _safe_float(active.get("ece_delta")),
            "calibration_gap_delta": _safe_float(active.get("calibration_gap_delta")),
            "hit_rate_delta": _safe_float(active.get("hit_rate_delta")),
            "profit_units_delta": _safe_float(active.get("profit_units_delta")),
            "active_non_target_hit_rate_delta": _safe_float(active.get("non_rebound_hit_rate_delta")),
            "opposite_side_enabled": bool(opposite.get("enabled", False)),
            "flagged_over_count": _safe_int(opposite.get("flagged_over_count")),
            "synthetic_under_candidates_created": _safe_int(opposite.get("synthetic_under_candidates_created")),
            "under_candidates_with_valid_price": _safe_int(opposite.get("under_candidates_with_valid_price")),
            "under_candidates_passing_break_even": _safe_int(opposite.get("under_candidates_passing_break_even")),
            "under_candidates_added_to_board": _safe_int(opposite.get("under_candidates_added_to_board")),
            "under_candidates_rejected_price": _safe_int(opposite.get("under_candidates_rejected_price")),
            "under_candidates_rejected_forecastability": _safe_int(opposite.get("under_candidates_rejected_forecastability")),
            "under_candidates_rejected_stress": _safe_int(opposite.get("under_candidates_rejected_stress")),
            "under_candidate_wins": _safe_int(under_results.get("wins")),
            "under_candidate_losses": _safe_int(under_results.get("losses")),
            "under_candidate_pushes": _safe_int(under_results.get("pushes")),
            "under_candidate_hit_rate": _safe_float(under_results.get("hit_rate")),
            "under_candidate_roi": _safe_float(under_results.get("roi")),
            "added_under_rows": opposite.get("added_under_rows", []),
        }
        out["window_key"] = (
            f"{out['window_start_run_date']}:{out['window_end_run_date']}:"
            f"{out['validation_mode']}:{out['variant']}:{Path(source_path).name}"
        )
        records.append(out)
    return records


def build_intervention_validation_payload(
    *,
    intervention_family: str,
    intervention_id: str,
    failure_mode_id: str,
    summary_payloads: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    variant_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for source_path, payload in summary_payloads:
        variant_rows.extend(_flatten_variant_rows(payload, source_path))
        segment_rows.extend(_flatten_segment_rows(payload, source_path))
        window_rows.extend(_flatten_window_rows(payload, source_path))
    variants = pd.DataFrame(variant_rows)
    segments = pd.DataFrame(segment_rows)
    windows = pd.DataFrame(window_rows)
    return {
        "intervention_family": str(intervention_family).strip(),
        "intervention_id": str(intervention_id).strip(),
        "failure_mode_id": str(failure_mode_id).strip(),
        "variant_order": DEFAULT_VARIANT_ORDER,
        "variant_summaries": variants.to_dict(orient="records"),
        "window_reports": windows.to_dict(orient="records"),
        "segment_report": segments.to_dict(orient="records"),
        "validation_modes_present": sorted(windows.get("validation_mode", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()),
    }


def main() -> None:
    args = parse_args()
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in args.validation_summary_json:
        payloads.append((summary_path, json.loads(summary_path.resolve().read_text(encoding="utf-8"))))
    out_payload = build_intervention_validation_payload(
        intervention_family=args.intervention_family,
        intervention_id=args.intervention_id,
        failure_mode_id=args.failure_mode_id,
        summary_payloads=payloads,
    )
    variants = pd.DataFrame(out_payload["variant_summaries"])
    windows = pd.DataFrame(out_payload["window_reports"])
    segments = pd.DataFrame(out_payload["segment_report"])
    args.variant_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    variants.to_csv(args.variant_csv_out, index=False)
    windows.to_csv(args.window_csv_out, index=False)
    segments.to_csv(args.segments_csv_out, index=False)
    write_json(args.out_json, out_payload)


if __name__ == "__main__":
    main()

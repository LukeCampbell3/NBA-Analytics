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


DEFAULT_BROADER_WINDOW_MIN_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a promotion-safe report for a generic intervention validation payload.")
    parser.add_argument("--validation-summary-json", type=Path, action="append", required=True)
    parser.add_argument("--intervention-id", type=str, required=True)
    parser.add_argument("--target-variant", type=str, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--variant-csv-out", type=Path, required=True)
    parser.add_argument("--segments-csv-out", type=Path, required=True)
    parser.add_argument("--broader-window-min-count", type=int, default=DEFAULT_BROADER_WINDOW_MIN_COUNT)
    return parser.parse_args()


def _load_generic_validation_payloads(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        variant_rows.extend(payload.get("variant_summaries", []))
        window_rows.extend(payload.get("window_reports", []))
        segment_rows.extend(payload.get("segment_report", []))
    return pd.DataFrame(variant_rows), pd.DataFrame(window_rows), pd.DataFrame(segment_rows)


def build_intervention_promotion_gate(
    window_reports: pd.DataFrame,
    *,
    target_variant: str,
    broader_window_min_count: int = DEFAULT_BROADER_WINDOW_MIN_COUNT,
) -> dict[str, Any]:
    target_reports = window_reports.loc[window_reports.get("variant", pd.Series(dtype="object")).astype(str) == str(target_variant)].copy()
    if target_reports.empty:
        return {
            "shadow_validated_logic": False,
            "trained_bundle_validated": False,
            "broader_walk_forward_validated": False,
            "promotion_ready": False,
            "promotion_status_label": "needs_more_sample",
            "blocked_reasons": ["no_validation_windows_loaded"],
            "required_next_steps": ["Run no-op and active-risk validation windows before evaluating promotion."],
            "rollback_plan": "Disable the intervention ablation flag and revert to the baseline selector.",
        }

    heuristic_reports = target_reports.loc[target_reports["validation_mode"].astype(str) == "artifact_free_heuristic"].copy()
    trained_reports = target_reports.loc[target_reports["validation_mode"].astype(str) == "trained_bundle"].copy()

    heuristic_no_op = bool(heuristic_reports.get("no_op_narrowness_passed", pd.Series(dtype=bool)).astype(bool).any())
    heuristic_active = bool(heuristic_reports.get("active_improvement_passed", pd.Series(dtype=bool)).astype(bool).any())
    trained_no_op = bool(trained_reports.get("no_op_narrowness_passed", pd.Series(dtype=bool)).astype(bool).any())
    trained_active = bool(trained_reports.get("active_improvement_passed", pd.Series(dtype=bool)).astype(bool).any())

    shadow_validated_logic = bool(heuristic_no_op and heuristic_active)
    trained_bundle_validated = bool(trained_no_op and trained_active)

    no_op_windows = set(target_reports.loc[target_reports.get("no_op_narrowness_passed", pd.Series(dtype=bool)).astype(bool), "window_key"].astype(str).tolist())
    active_windows = set(target_reports.loc[target_reports.get("active_improvement_passed", pd.Series(dtype=bool)).astype(bool), "window_key"].astype(str).tolist())
    broader_walk_forward_validated = bool(
        trained_bundle_validated
        and len(no_op_windows | active_windows) >= int(broader_window_min_count)
        and bool(no_op_windows)
        and bool(active_windows)
    )

    blocked_reasons: list[str] = []
    if not heuristic_no_op:
        blocked_reasons.append("no_op_narrowness_window_required")
    if not heuristic_active:
        blocked_reasons.append("active_risk_improvement_window_required")
    if not trained_bundle_validated:
        blocked_reasons.append("trained_bundle_replay_required")
    if not broader_walk_forward_validated:
        blocked_reasons.append("broader_walk_forward_required")
    if bool((target_reports.get("active_non_target_board_change_count", pd.Series(dtype="float64")).fillna(0) > 0).any()):
        blocked_reasons.append("unexpected_non_target_market_damage")
    if bool((target_reports.get("no_op_non_target_board_change_count", pd.Series(dtype="float64")).fillna(0) > 0).any()):
        blocked_reasons.append("unexpected_non_target_market_damage")
    if bool((target_reports.get("removed_losses", pd.Series(dtype="float64")).fillna(0) < target_reports.get("removed_wins", pd.Series(dtype="float64")).fillna(0)).any()):
        blocked_reasons.append("removed_more_wins_than_losses")
    if bool((target_reports.get("active_coverage_retained", pd.Series(dtype="float64")).fillna(1.0) < 0.95).any()):
        blocked_reasons.append("coverage_below_threshold")
    if bool((target_reports.get("roi_delta", pd.Series(dtype="float64")).fillna(0.0) < -1e-9).any()):
        blocked_reasons.append("roi_worsened")
    if bool((target_reports.get("brier_delta", pd.Series(dtype="float64")).fillna(0.0) > 1e-9).any()):
        blocked_reasons.append("brier_worsened")
    if bool((target_reports.get("ece_delta", pd.Series(dtype="float64")).fillna(0.0) > 1e-9).any()):
        blocked_reasons.append("ece_worsened")
    if bool(
        (
            target_reports.get("under_candidates_added_to_board", pd.Series(dtype="float64")).fillna(0)
            > target_reports.get("under_candidates_with_valid_price", pd.Series(dtype="float64")).fillna(0)
        ).any()
    ):
        blocked_reasons.append("opposite_side_added_without_valid_price")
    if bool(
        (
            target_reports.get("under_candidates_added_to_board", pd.Series(dtype="float64")).fillna(0)
            > target_reports.get("under_candidates_passing_break_even", pd.Series(dtype="float64")).fillna(0)
        ).any()
    ):
        blocked_reasons.append("opposite_side_added_without_break_even")

    promotion_ready = bool(not blocked_reasons and shadow_validated_logic and trained_bundle_validated and broader_walk_forward_validated)
    if promotion_ready:
        status_label = "promotion_candidate"
    elif "unexpected_non_target_market_damage" in blocked_reasons:
        status_label = "rejected_damages_non_target"
    elif "removed_more_wins_than_losses" in blocked_reasons:
        status_label = "rejected_overfit"
    elif shadow_validated_logic and not trained_bundle_validated:
        status_label = "trained_bundle_required"
    elif trained_bundle_validated and not broader_walk_forward_validated:
        status_label = "broader_walk_forward_required"
    else:
        status_label = "needs_more_sample"

    next_steps_map = {
        "no_op_narrowness_window_required": "Run no-op windows and confirm the intervention does not over-trigger.",
        "active_risk_improvement_window_required": "Run active-risk windows and confirm more losses are removed than wins.",
        "trained_bundle_replay_required": "Replay the intervention with the trained model bundle enabled.",
        "broader_walk_forward_required": "Expand the replay to broader walk-forward windows before promotion.",
        "unexpected_non_target_market_damage": "Remove or justify non-target market board changes before promotion.",
        "removed_more_wins_than_losses": "Tighten the trigger so win preservation improves.",
        "coverage_below_threshold": "Reduce coverage loss or adjust the intervention into a softer downgrade.",
        "roi_worsened": "Revisit the penalty strength because ROI worsened.",
        "brier_worsened": "Revisit the intervention because calibration quality worsened.",
        "ece_worsened": "Revisit the intervention because calibration quality worsened.",
        "opposite_side_added_without_valid_price": "Tighten opposite-side eligibility so only priced candidates are added.",
        "opposite_side_added_without_break_even": "Tighten opposite-side eligibility so only break-even-positive candidates are added.",
    }
    return {
        "shadow_validated_logic": shadow_validated_logic,
        "trained_bundle_validated": trained_bundle_validated,
        "broader_walk_forward_validated": broader_walk_forward_validated,
        "promotion_ready": promotion_ready,
        "promotion_status_label": status_label,
        "blocked_reasons": blocked_reasons,
        "required_next_steps": [next_steps_map[key] for key in blocked_reasons if key in next_steps_map],
        "rollback_plan": "Disable the intervention ablation flag and delete the generated failure_mode_adjustments.csv sidecar.",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    lines = [
        "| " + " | ".join(table.columns.tolist()) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        rendered: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                rendered.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _build_markdown_report(
    *,
    intervention_id: str,
    gate: dict[str, Any],
    variant_rows: pd.DataFrame,
    segment_rows: pd.DataFrame,
) -> str:
    lines = [
        "# Intervention Promotion Gate",
        "",
        f"- intervention_id: `{intervention_id}`",
        f"- shadow_validated_logic: `{gate['shadow_validated_logic']}`",
        f"- trained_bundle_validated: `{gate['trained_bundle_validated']}`",
        f"- broader_walk_forward_validated: `{gate['broader_walk_forward_validated']}`",
        f"- promotion_ready: `{gate['promotion_ready']}`",
        f"- promotion_status_label: `{gate['promotion_status_label']}`",
        "",
        "## Blocked Reasons",
        "",
    ]
    blocked = gate.get("blocked_reasons", [])
    if blocked:
        lines.extend([f"- {reason}" for reason in blocked])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Required Next Steps",
            "",
        ]
    )
    next_steps = gate.get("required_next_steps", [])
    if next_steps:
        lines.extend([f"- {step}" for step in next_steps])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Variant Comparison",
            "",
            _markdown_table(
                variant_rows,
                [
                    "variant",
                    "validation_mode",
                    "resolved_picks",
                    "hit_rate",
                    "roi",
                    "brier",
                    "ece",
                    "status_label",
                ],
            ),
            "",
            "## Segments",
            "",
            _markdown_table(
                segment_rows,
                [
                    "segment",
                    "candidate_count",
                    "final_board_count",
                    "removed_count",
                    "wins_removed",
                    "losses_removed",
                    "avg_penalty",
                ],
            ),
            "",
            "## Rollback Plan",
            "",
            gate["rollback_plan"],
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    variant_rows, window_rows, segment_rows = _load_generic_validation_payloads(args.validation_summary_json)
    gate = build_intervention_promotion_gate(
        window_rows,
        target_variant=args.target_variant,
        broader_window_min_count=int(args.broader_window_min_count),
    )
    args.variant_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    variant_rows.to_csv(args.variant_csv_out, index=False)
    segment_rows.to_csv(args.segments_csv_out, index=False)
    markdown = _build_markdown_report(
        intervention_id=args.intervention_id,
        gate=gate,
        variant_rows=variant_rows,
        segment_rows=segment_rows,
    )
    args.out_md.resolve().write_text(markdown, encoding="utf-8")
    write_json(
        args.out_json,
        {
            "intervention_id": args.intervention_id,
            "variant_summaries": variant_rows.to_dict(orient="records"),
            "window_reports": window_rows.to_dict(orient="records"),
            "segment_report": segment_rows.to_dict(orient="records"),
            "promotion_gate": gate,
        },
    )


if __name__ == "__main__":
    main()

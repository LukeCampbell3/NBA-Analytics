from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.build_needs_more_sample_queue import build_needs_more_sample_queue
from research.safe_state.evaluate_safe_state_shadow_results import evaluate_safe_state_shadow_results
from research.safe_state.expand_comparable_state_sampling import expand_comparable_state_sampling
from research.safe_state.lock_true_unstable_shadow_rejections import lock_true_unstable_shadow_rejections
from research.safe_state.recheck_needs_more_sample_candidates import recheck_needs_more_sample_candidates
from research.safe_state.safe_state_evidence_ledger import append_safe_state_evidence_ledger


def run_safe_state_evidence_lifecycle(
    *,
    output_dir: Path,
    annotated_candidates_csv: Path,
    blocker_resolution_rows_csv: Path,
    root_cause_rows_csv: Path,
    candidate_blockers_csv: Path | None = None,
    data_proc_dir: Path | None = None,
    run_id: str | None = None,
    evaluate_settlement: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"safe_state_lifecycle_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    true_unstable_report = lock_true_unstable_shadow_rejections(
        output_dir=output_dir,
        annotated_candidates_csv=annotated_candidates_csv,
        blocker_resolution_rows_csv=blocker_resolution_rows_csv,
        root_cause_rows_csv=root_cause_rows_csv,
        candidate_blockers_csv=candidate_blockers_csv,
    )
    queue_report = build_needs_more_sample_queue(
        output_dir=output_dir,
        blocker_resolution_rows_csv=blocker_resolution_rows_csv,
        root_cause_rows_csv=root_cause_rows_csv,
        annotated_candidates_csv=annotated_candidates_csv,
    )
    expansion_report = expand_comparable_state_sampling(
        output_dir=output_dir,
        needs_more_sample_queue_csv=output_dir / "needs_more_sample_queue.csv",
        annotated_candidates_csv=annotated_candidates_csv,
        data_proc_dir=data_proc_dir,
    )
    recheck_report = recheck_needs_more_sample_candidates(
        output_dir=output_dir,
        needs_more_sample_queue_csv=output_dir / "needs_more_sample_queue.csv",
        comparable_state_expansion_rows_csv=output_dir / "comparable_state_expansion_rows.csv",
        annotated_candidates_csv=annotated_candidates_csv,
    )
    ledger_report = append_safe_state_evidence_ledger(
        ledger_path=output_dir / "safe_state_evidence_ledger.jsonl",
        run_id=run_id,
        true_unstable_csv=output_dir / "true_unstable_shadow_rejections.csv",
        needs_more_sample_queue_csv=output_dir / "needs_more_sample_queue.csv",
        recheck_csv=output_dir / "needs_more_sample_recheck.csv",
    )
    settlement_report = evaluate_safe_state_shadow_results(board_dir=output_dir, output_dir=output_dir) if evaluate_settlement else {}
    shadow_boards_path = output_dir / "safe_state_shadow_boards.csv"
    variant_summary_path = output_dir / "safe_state_shadow_variant_summary.csv"
    if variant_summary_path.exists():
        shutil.copyfile(variant_summary_path, shadow_boards_path)
    elif not shadow_boards_path.exists():
        shadow_boards_path.write_text("variant,board_size,status\n", encoding="utf-8")

    statuses = []
    if true_unstable_report.get("locked_true_unstable_count", 0):
        statuses.append("TRUE_UNSTABLE_REJECTIONS_TRACKED")
    if queue_report.get("needs_more_sample_count", 0):
        statuses.append("NEEDS_MORE_SAMPLE_TRACKING")
    if recheck_report.get("recheck_status_counts", {}).get("PROMOTED_TO_SAFE_STATE_CORE_SHADOW", 0):
        statuses.append("SAFE_STATE_CORE_FOUND_SHADOW")
    if evaluate_settlement:
        statuses.append("SETTLEMENT_EVALUATED")
    else:
        statuses.append("SETTLEMENT_PENDING")
    statuses.append("MULTI_SLATE_EVIDENCE_ACCUMULATING")
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "annotated_candidates_csv": str(annotated_candidates_csv),
            "blocker_resolution_rows_csv": str(blocker_resolution_rows_csv),
            "root_cause_rows_csv": str(root_cause_rows_csv),
            "candidate_blockers_csv": str(candidate_blockers_csv) if candidate_blockers_csv else "",
            "data_proc_dir": str(data_proc_dir) if data_proc_dir else "",
        },
        "output_paths": {
            "json": str(output_dir / "safe_state_lifecycle_report.json"),
            "markdown": str(output_dir / "safe_state_lifecycle_report.md"),
            "ledger": str(output_dir / "safe_state_evidence_ledger.jsonl"),
            "needs_more_sample_queue": str(output_dir / "needs_more_sample_queue.csv"),
            "true_unstable_shadow_rejections": str(output_dir / "true_unstable_shadow_rejections.csv"),
            "safe_state_shadow_boards": str(shadow_boards_path),
            "safe_state_settlement_evaluation": str(output_dir / "safe_state_settlement_rows.csv"),
        },
        "status_labels": statuses,
        "true_unstable": true_unstable_report,
        "needs_more_sample_queue": queue_report,
        "comparable_state_expansion": expansion_report,
        "needs_more_sample_recheck": recheck_report,
        "ledger": ledger_report,
        "settlement": settlement_report,
        "future_promotion_criteria": [
            "multiple_settled_slates",
            "enough_safe_core_or_near_core_rows",
            "true_unstable_rejections_remove_more_losses_than_wins",
            "needs_more_sample_candidates_mature_or_reject",
            "paired_comparison_improves_roi_brier_ece",
            "no_coverage_collapse",
            "no_hidden_player_team_date_overfit",
            "trained_bundle_validation_passes",
            "broader_walk_forward_validation_passes",
            "rollback_rule_exists",
        ],
        "promotion_status": "NOT_PROMOTION_ELIGIBLE",
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    (output_dir / "safe_state_lifecycle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "safe_state_lifecycle_report.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Safe-State Evidence Lifecycle Report",
        "",
        f"- Run ID: {report['run_id']}",
        f"- Promotion status: {report['promotion_status']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Status Labels",
    ]
    lines.extend([f"- {status}" for status in report["status_labels"]])
    lines.extend(
        [
            "",
            "## Summary",
            f"- True-unstable locked: {report['true_unstable'].get('locked_true_unstable_count', 0)}",
            f"- Needs-more-sample queued: {report['needs_more_sample_queue'].get('needs_more_sample_count', 0)}",
            f"- Ledger entries appended: {report['ledger'].get('entries_appended', 0)}",
            "",
            "No selector gate, production sidecar, threshold relaxation, or promotion was created.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shadow-only safe-state evidence lifecycle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--blocker-resolution-rows-csv", type=Path, required=True)
    parser.add_argument("--root-cause-rows-csv", type=Path, required=True)
    parser.add_argument("--candidate-blockers-csv", type=Path)
    parser.add_argument("--data-proc-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--evaluate-settlement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_safe_state_evidence_lifecycle(
        output_dir=args.output_dir,
        annotated_candidates_csv=args.annotated_candidates_csv,
        blocker_resolution_rows_csv=args.blocker_resolution_rows_csv,
        root_cause_rows_csv=args.root_cause_rows_csv,
        candidate_blockers_csv=args.candidate_blockers_csv,
        data_proc_dir=args.data_proc_dir,
        run_id=args.run_id,
        evaluate_settlement=bool(args.evaluate_settlement),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

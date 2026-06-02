from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _config_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("safe_state_production_test", config)
    return section if isinstance(section, dict) else {}


def _promotion_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _config_section(config).get("promotion", {})
    return section if isinstance(section, dict) else {}


def evaluate_safe_state_promotion_gate(
    *,
    aggregate_metrics_csv: Path | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    metrics = _read_csv(aggregate_metrics_csv)
    config = _load_config(config_path)
    promotion = _promotion_config(config)
    min_slates = int(promotion.get("min_settled_slates", 20) or 20)
    min_safe_rows = int(promotion.get("min_resolved_safe_state_rows", 50) or 50)

    settled_slates = 0
    resolved_safe_rows = 0
    if not metrics.empty:
        if "run_date" in metrics.columns and "resolved_rows" in metrics.columns:
            settled_slates = int(metrics.loc[pd.to_numeric(metrics["resolved_rows"], errors="coerce").fillna(0).gt(0), "run_date"].nunique())
        variant_col = "variant" if "variant" in metrics.columns else "board_variant" if "board_variant" in metrics.columns else ""
        if variant_col:
            safe_mask = metrics[variant_col].fillna("").astype(str).isin({"safe_state_core_board", "safe_state_near_core_board"})
            resolved_safe_rows = int(pd.to_numeric(metrics.loc[safe_mask, "resolved_rows"], errors="coerce").fillna(0).sum()) if "resolved_rows" in metrics.columns else 0

    blocked_reasons: list[str] = []
    if settled_slates < min_slates:
        blocked_reasons.append("min_settled_slates_not_met")
    if resolved_safe_rows < min_safe_rows:
        blocked_reasons.append("min_resolved_safe_state_rows_not_met")
    if promotion.get("require_trained_bundle_validation", True):
        blocked_reasons.append("trained_bundle_validation_required")
    if promotion.get("require_broader_walk_forward", True):
        blocked_reasons.append("broader_walk_forward_required")
    blocked_reasons.append("ring_3_enforcement_not_enabled")

    report = {
        "promotion_ready": False,
        "promotion_status": "NOT_PROMOTION_ELIGIBLE",
        "blocked_reasons": sorted(set(blocked_reasons)),
        "minimum_evidence_requirements": {
            "min_settled_slates": min_slates,
            "min_resolved_safe_state_rows": min_safe_rows,
            "trained_bundle_validation": bool(promotion.get("require_trained_bundle_validation", True)),
            "broader_walk_forward_validation": bool(promotion.get("require_broader_walk_forward", True)),
            "loss_removal_gt_win_removal": bool(promotion.get("require_loss_removal_gt_win_removal", True)),
            "brier_ece_roi_not_worse": bool(
                promotion.get("require_brier_not_worse", True)
                and promotion.get("require_ece_not_worse", True)
                and promotion.get("require_roi_not_worse", True)
            ),
            "no_coverage_collapse": bool(promotion.get("require_no_coverage_collapse", True)),
        },
        "current_evidence_summary": {
            "settled_slates": settled_slates,
            "resolved_safe_state_rows": resolved_safe_rows,
            "aggregate_metrics_csv": str(aggregate_metrics_csv) if aggregate_metrics_csv else "",
        },
        "next_required_steps": [
            "continue_ring_1_production_shadow_collection",
            "wait_for_settlement_and_update_outcomes",
            "accumulate_multi_slate_evidence",
            "run_trained_bundle_validation",
            "run_broader_walk_forward_validation",
        ],
        "rollback_plan": "Keep safe-state enforcement disabled; remove shadow runner schedule if evidence quality regresses.",
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "safe_state_promotion_gate_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the disabled safe-state production promotion gate.")
    parser.add_argument("--aggregate-metrics-csv", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_safe_state_promotion_gate(
        aggregate_metrics_csv=args.aggregate_metrics_csv,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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

from research.market_quality.common import compute_price_quality_frame, merge_selected_with_candidate_pool  # noqa: E402


SELECTED_ROW_COLUMNS = [
    "player",
    "market_type",
    "side",
    "line",
    "market_side_price",
    "market_side_break_even",
    "stress_probability",
    "lcb_probability",
    "stress_edge",
    "lcb_edge",
    "forecastability_score",
    "scenario_agreement",
    "edge_defendability_tier",
    "timestamp_safety_basis",
    "timestamp_safety_blocked_reason",
    "price_validity_status",
    "price_source",
    "price_source_type",
    "book",
    "event_time_source",
    "event_time_confidence",
]


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_selected_detail() -> pd.DataFrame:
    return pd.DataFrame(columns=SELECTED_ROW_COLUMNS)


def _load_ledger(
    *,
    price_audit_csv: Path | None,
    selected_board_csv: Path | None,
    candidate_pool_csv: Path | None,
) -> pd.DataFrame:
    audit = _read_csv(price_audit_csv)
    if not audit.empty and "record_scope" in audit.columns:
        return audit
    selected = _read_csv(selected_board_csv)
    candidates = _read_csv(candidate_pool_csv)
    selected = merge_selected_with_candidate_pool(selected, candidates)
    candidate_ledger = compute_price_quality_frame(candidates, record_scope="candidate") if not candidates.empty else pd.DataFrame()
    selected_ledger = compute_price_quality_frame(selected, record_scope="selected") if not selected.empty else pd.DataFrame()
    return pd.concat([candidate_ledger, selected_ledger], ignore_index=True, sort=False)


def _tier_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "edge_defendability_tier" not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame["edge_defendability_tier"].fillna("").astype(str).value_counts().to_dict().items()}


def _selected_detail(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return _empty_selected_detail()
    detail = selected.copy()
    for column in SELECTED_ROW_COLUMNS:
        if column not in detail.columns:
            detail[column] = pd.NA
    return detail[SELECTED_ROW_COLUMNS].copy()


def build_edge_defense_report(
    *,
    output_dir: Path,
    price_audit_csv: Path | None = None,
    selected_board_csv: Path | None = None,
    candidate_pool_csv: Path | None = None,
    recency_diagnosis_json: Path | None = None,
) -> dict[str, Any]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(
        price_audit_csv=price_audit_csv,
        selected_board_csv=selected_board_csv,
        candidate_pool_csv=candidate_pool_csv,
    )
    if ledger.empty:
        candidates = pd.DataFrame()
        selected = pd.DataFrame()
    else:
        scope = ledger.get("record_scope", pd.Series("", index=ledger.index)).fillna("").astype(str)
        candidates = ledger.loc[scope.eq("candidate")].copy()
        selected = ledger.loc[scope.eq("selected")].copy()
    recency = _read_json(recency_diagnosis_json)
    selected_validity = selected.get("price_validity_status", pd.Series("", index=selected.index)).fillna("").astype(str)
    selected_tier = selected.get("edge_defendability_tier", pd.Series("", index=selected.index)).fillna("").astype(str)
    selected_lcb_edge = pd.to_numeric(selected.get("lcb_edge"), errors="coerce")
    selected_stress_edge = pd.to_numeric(selected.get("stress_edge"), errors="coerce")
    selected_blocked_reason = selected.get("timestamp_safety_blocked_reason", pd.Series("", index=selected.index)).fillna("").astype(str)
    selected_event_time_blocks = selected_blocked_reason.str.contains("missing_event_time|event_start|explicit_prelock", case=False, na=False)

    recency_blocks = bool(
        int(recency.get("rows_before_recency", 0) or 0) > 0
        and int(recency.get("rows_after_recency", 0) or 0) == 0
    )
    report = {
        "input_paths": {
            "price_audit_csv": str(price_audit_csv) if price_audit_csv else "",
            "selected_board_csv": str(selected_board_csv) if selected_board_csv else "",
            "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv else "",
            "recency_diagnosis_json": str(recency_diagnosis_json) if recency_diagnosis_json else "",
        },
        "output_paths": {
            "edge_defense_report_json": str(output_dir / "edge_defense_report.json"),
            "edge_defense_report_md": str(output_dir / "edge_defense_report.md"),
            "edge_defense_selected_rows_csv": str(output_dir / "edge_defense_selected_rows.csv"),
        },
        "total_candidate_rows": int(len(candidates)),
        "total_selected_rows": int(len(selected)),
        "selected_rows_with_PRICE_VALID": int(selected_validity.eq("PRICE_VALID").sum()),
        "selected_rows_with_EDGE_DEFENDABLE": int(selected_tier.eq("EDGE_DEFENDABLE").sum()),
        "selected_rows_with_EDGE_PRICE_DEPENDENT": int(selected_tier.eq("EDGE_PRICE_DEPENDENT").sum()),
        "selected_rows_with_EDGE_UNTRUSTED_PRICE": int(selected_tier.eq("EDGE_UNTRUSTED_PRICE").sum()),
        "selected_rows_with_EDGE_DIAGNOSTIC_ONLY": int(selected_tier.eq("EDGE_DIAGNOSTIC_ONLY").sum()),
        "selected_rows_where_lcb_edge_gt_0": int(selected_lcb_edge.gt(0.0).sum()),
        "selected_rows_where_stress_edge_gt_0": int(selected_stress_edge.gt(0.0).sum()),
        "selected_rows_where_price_blocks_validation": int(
            selected.get("price_gap_blocks_validation", pd.Series(False, index=selected.index)).fillna(False).astype(bool).sum()
        )
        if not selected.empty
        else 0,
        "selected_rows_where_event_time_blocks_validation": int(selected_event_time_blocks.sum()),
        "selected_rows_where_recency_blocks_production_selection": int(len(selected)) if recency_blocks and len(selected) else 0,
        "candidate_rows_by_edge_defendability_tier": _tier_counts(candidates),
        "selected_rows_by_edge_defendability_tier": _tier_counts(selected),
        "recency_blocks_production_selection": recency_blocks,
        "recency_root_cause": recency.get("freshness_root_cause", ""),
        "recency_recommended_repair": recency.get("recommended_repair", ""),
        "production_behavior_changed": False,
        "promotion_claim": False,
    }
    detail = _selected_detail(selected)
    detail.to_csv(output_dir / "edge_defense_selected_rows.csv", index=False)
    (output_dir / "edge_defense_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Selected-Row Edge Defense Report",
        "",
        f"- Candidate rows: {report['total_candidate_rows']}",
        f"- Selected rows: {report['total_selected_rows']}",
        f"- Selected PRICE_VALID rows: {report['selected_rows_with_PRICE_VALID']}",
        f"- Selected EDGE_DEFENDABLE rows: {report['selected_rows_with_EDGE_DEFENDABLE']}",
        f"- Selected EDGE_PRICE_DEPENDENT rows: {report['selected_rows_with_EDGE_PRICE_DEPENDENT']}",
        f"- Selected EDGE_UNTRUSTED_PRICE rows: {report['selected_rows_with_EDGE_UNTRUSTED_PRICE']}",
        f"- Selected EDGE_DIAGNOSTIC_ONLY rows: {report['selected_rows_with_EDGE_DIAGNOSTIC_ONLY']}",
        f"- Recency blocks production selection: {report['recency_blocks_production_selection']}",
        f"- Recency root cause: {report['recency_root_cause']}",
        "",
        "This report is audit-only. It does not rerank, veto, downgrade, or promote picks.",
    ]
    (output_dir / "edge_defense_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report selected-row price-valid edge defense status.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--price-audit-csv", type=Path, default=None)
    parser.add_argument("--selected-board-csv", type=Path, default=None)
    parser.add_argument("--candidate-pool-csv", type=Path, default=None)
    parser.add_argument("--recency-diagnosis-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_edge_defense_report(
        output_dir=args.output_dir,
        price_audit_csv=args.price_audit_csv,
        selected_board_csv=args.selected_board_csv,
        candidate_pool_csv=args.candidate_pool_csv,
        recency_diagnosis_json=args.recency_diagnosis_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

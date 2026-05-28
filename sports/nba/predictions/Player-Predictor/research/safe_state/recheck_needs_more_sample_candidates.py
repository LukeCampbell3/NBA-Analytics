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

from research.market_quality.common import candidate_identity_columns


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ensure_direction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "direction" not in out.columns and "side" in out.columns:
        out["direction"] = out["side"]
    return out


def _best_expansion(expansions: pd.DataFrame) -> pd.DataFrame:
    if expansions.empty:
        return pd.DataFrame()
    out_rows = []
    priority = {
        "SUFFICIENT_TIGHT": 0,
        "SUFFICIENT_SCATTERED": 1,
        "INSUFFICIENT_SAMPLE": 2,
        "CONTEXT_MISMATCH": 3,
        "NOT_AVAILABLE": 4,
    }
    working = expansions.copy()
    working["_status_order"] = working.get("expansion_status", "").fillna("").astype(str).map(priority).fillna(9)
    working["_fallback_level"] = pd.to_numeric(working.get("fallback_level", 999), errors="coerce").fillna(999)
    tightness_source = working["tightness_score"] if "tightness_score" in working.columns else pd.Series(0, index=working.index)
    working["_tightness"] = pd.to_numeric(tightness_source, errors="coerce").fillna(0)
    for _, group in working.sort_values(["_status_order", "_fallback_level", "_tightness"], ascending=[True, True, False]).groupby("candidate_id", sort=False):
        out_rows.append(group.iloc[0].drop(labels=["_status_order", "_fallback_level", "_tightness"], errors="ignore").to_dict())
    return pd.DataFrame.from_records(out_rows)


def recheck_needs_more_sample_candidates(
    *,
    output_dir: Path,
    needs_more_sample_queue_csv: Path,
    comparable_state_expansion_rows_csv: Path,
    annotated_candidates_csv: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue = candidate_identity_columns(_ensure_direction(_read_csv(needs_more_sample_queue_csv)))
    expansions = candidate_identity_columns(_ensure_direction(_read_csv(comparable_state_expansion_rows_csv)))
    annotated = candidate_identity_columns(_read_csv(annotated_candidates_csv))
    best = _best_expansion(expansions)

    working = queue.copy()
    if not best.empty:
        working = working.merge(best, on="candidate_id", how="left", suffixes=("", "_expansion"))
    if not annotated.empty:
        keep = [
            "candidate_id",
            "edge_defendability_tier",
            "price_validity_status",
            "stress_edge",
            "lcb_edge",
            "forecastability_gap_primary",
            "forecastability_gap_secondary",
            "forecastability_gap_severity",
            "forecastability_gap_fixability",
            "safe_state_tier",
            "structural_mispricing_tier",
            "forecastability_tier",
        ]
        working = working.merge(annotated[[col for col in keep if col in annotated.columns]], on="candidate_id", how="left", suffixes=("", "_annotated"))

    records: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        status = str(row.get("expansion_status", "") or "")
        reliability = str(row.get("comparable_state_reliability_tier", "") or "")
        if status == "SUFFICIENT_SCATTERED" or reliability == "SCATTERED":
            recheck = "REJECTED_SIMILAR_STATE_SCATTER"
        elif status == "SUFFICIENT_TIGHT":
            structural = str(row.get("structural_mispricing_tier", "") or "").upper()
            forecast = str(row.get("forecastability_tier", "") or "").upper()
            if structural in {"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"} and forecast in {
                "HIGH_FORECASTABILITY",
                "MEDIUM_FORECASTABILITY",
            }:
                recheck = "PROMOTED_TO_SAFE_STATE_CORE_SHADOW"
            else:
                recheck = "PROMOTED_TO_SAFE_STATE_NEAR_CORE"
        elif status == "INSUFFICIENT_SAMPLE" or not status:
            recheck = "REMAINS_NEEDS_MORE_SAMPLE"
        else:
            recheck = "REMAINS_NEEDS_MORE_SAMPLE"
        records.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "player": row.get("player", row.get("player_name", "")),
                "game_id": row.get("game_id", ""),
                "market_date": row.get("market_date", row.get("game_date", "")),
                "target": row.get("target", ""),
                "market_type": row.get("market_type", ""),
                "side": row.get("side", row.get("direction", "")),
                "line": row.get("line", row.get("market_line", "")),
                "best_fallback_level": row.get("fallback_level", ""),
                "best_fallback_label": row.get("fallback_label", ""),
                "match_count": row.get("match_count", ""),
                "tightness_score": row.get("tightness_score", ""),
                "comparable_state_reliability_tier": reliability,
                "expansion_status": status,
                "recheck_status": recheck,
                "queue_status_after_recheck": {
                    "PROMOTED_TO_SAFE_STATE_NEAR_CORE": "PROMOTED_TO_NEAR_CORE",
                    "PROMOTED_TO_SAFE_STATE_CORE_SHADOW": "PROMOTED_TO_SAFE_CORE",
                    "REJECTED_SIMILAR_STATE_SCATTER": "REJECTED_SCATTERED",
                }.get(recheck, "NEEDS_MORE_SAMPLE"),
                "shadow_only": True,
                "production_eligible": False,
            }
        )

    rows = pd.DataFrame.from_records(records)
    csv_path = output_dir / "needs_more_sample_recheck.csv"
    json_path = output_dir / "needs_more_sample_recheck.json"
    md_path = output_dir / "needs_more_sample_recheck.md"
    rows.to_csv(csv_path, index=False)
    report = {
        "input_paths": {
            "needs_more_sample_queue_csv": str(needs_more_sample_queue_csv),
            "comparable_state_expansion_rows_csv": str(comparable_state_expansion_rows_csv),
            "annotated_candidates_csv": str(annotated_candidates_csv),
        },
        "output_paths": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
        "rechecked_candidates": int(len(rows)),
        "recheck_status_counts": rows.get("recheck_status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict() if not rows.empty else {},
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report, rows)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Needs-More-Sample Recheck",
        "",
        f"- Rechecked candidates: {report['rechecked_candidates']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Status Counts",
    ]
    for status, count in report["recheck_status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "All promotions here are shadow-only evidence states, not production eligibility."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recheck NEEDS_MORE_SAMPLE rows after comparable-state expansion.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--needs-more-sample-queue-csv", type=Path, required=True)
    parser.add_argument("--comparable-state-expansion-rows-csv", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = recheck_needs_more_sample_candidates(
        output_dir=args.output_dir,
        needs_more_sample_queue_csv=args.needs_more_sample_queue_csv,
        comparable_state_expansion_rows_csv=args.comparable_state_expansion_rows_csv,
        annotated_candidates_csv=args.annotated_candidates_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

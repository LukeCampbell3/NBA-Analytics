from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import candidate_identity_columns
from research.safe_state.analyze_minutes_forecastability_gap import annotate_minutes_gap_decomposition
from research.safe_state.analyze_usage_forecastability_gap import annotate_usage_gap_decomposition


MINUTES_GAP = "FORECASTABILITY_GAP_MINUTES_STATE"
USAGE_GAP = "FORECASTABILITY_GAP_USAGE_STATE"
FIXABLE_LABELS = {"FIXABLE_WITH_EXISTING_LOGS", "FIXABLE_WITH_NEW_PIPELINE_DATA", "NEEDS_MORE_SAMPLE"}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _num(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def _gap_text(row: pd.Series) -> str:
    return ";".join(
        [
            _text(row, "forecastability_gap_primary"),
            _text(row, "forecastability_gap_secondary"),
            _text(row, "primary_blocker"),
            _text(row, "secondary_blockers"),
        ]
    ).upper()


def _merge_analysis(base: pd.DataFrame, analysis: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if analysis.empty:
        return base
    key = "candidate_id"
    analysis = candidate_identity_columns(analysis)
    columns = [key] + [col for col in analysis.columns if col.startswith(prefix)]
    columns = list(dict.fromkeys(columns))
    merged = base.merge(analysis[columns], on=key, how="left", suffixes=("", "_analysis"))
    for col in [c for c in merged.columns if c.endswith("_analysis")]:
        original = col.removesuffix("_analysis")
        if original in merged.columns:
            merged[original] = merged[original].where(merged[original].notna() & merged[original].astype(str).str.strip().ne(""), merged[col])
            merged = merged.drop(columns=[col])
    return merged


def _fixability_bucket(fixability: str, subtype: str) -> str:
    fix = str(fixability or "").upper()
    subtype_text = str(subtype or "").upper()
    if fix == "TRUE_UNSTABLE_STATE":
        return "true_unstable"
    if fix == "FIXABLE_WITH_EXISTING_LOGS":
        return "fixable_with_existing_logs"
    if fix == "FIXABLE_WITH_NEW_PIPELINE_DATA":
        return "fixable_with_new_pipeline_data"
    if fix == "NEEDS_MORE_SAMPLE":
        return "needs_more_sample"
    if fix == "FEATURE_MISSING" or "PIPELINE_MISSING" in subtype_text:
        return "feature_missing"
    return "unknown"


def _recommended_action(fixability: str, severity: str, near_core: bool) -> str:
    fix = str(fixability or "").upper()
    sev = str(severity or "").upper()
    if fix == "TRUE_UNSTABLE_STATE" or sev == "CRITICAL":
        return "KEEP_UNSAFE_TRUE_VOLATILITY"
    if near_core:
        return "WATCH_NEAR_CORE"
    if fix == "FIXABLE_WITH_EXISTING_LOGS":
        return "BACKFILL_EXISTING_LOG_FEATURES"
    if fix == "FIXABLE_WITH_NEW_PIPELINE_DATA":
        return "ADD_NEW_PIPELINE_DATA"
    if fix == "NEEDS_MORE_SAMPLE":
        return "NEEDS_MORE_SAMPLE"
    if fix == "FEATURE_MISSING":
        return "ADD_NEW_PIPELINE_DATA"
    return "SAFE_STATE_RECHECK_AFTER_BACKFILL"


def build_forecastability_blocker_resolution_report(
    *,
    output_dir: Path,
    annotated_candidates_csv: Path,
    candidate_blockers_csv: Path | None = None,
    minutes_gap_csv: Path | None = None,
    usage_gap_csv: Path | None = None,
    data_proc_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = candidate_identity_columns(_read_csv(annotated_candidates_csv))
    blockers = _read_csv(candidate_blockers_csv)
    blockers = candidate_identity_columns(blockers) if not blockers.empty else pd.DataFrame()

    minutes_analysis = _read_csv(minutes_gap_csv)
    if minutes_analysis.empty:
        minutes_analysis = annotate_minutes_gap_decomposition(annotated, data_proc_dir=data_proc_dir)
    usage_analysis = _read_csv(usage_gap_csv)
    if usage_analysis.empty:
        usage_analysis = annotate_usage_gap_decomposition(annotated)

    working = annotated.copy()
    if not blockers.empty:
        blocker_cols = [
            "candidate_id",
            "primary_blocker",
            "secondary_blockers",
            "missing_features",
            "evidence_gap_type",
            "safe_state_near_core_flag",
            "near_core_blocker_fixability",
        ]
        working = working.merge(blockers[[col for col in blocker_cols if col in blockers.columns]], on="candidate_id", how="left", suffixes=("", "_blockers"))
    working = _merge_analysis(working, minutes_analysis, "minutes_gap_")
    working = _merge_analysis(working, usage_analysis, "usage_gap_")

    edge_defendable = working.get("edge_defendability_tier", pd.Series("", index=working.index)).fillna("").astype(str).str.upper().eq("EDGE_DEFENDABLE")
    rows: list[dict[str, Any]] = []
    for _, row in working.loc[edge_defendable].iterrows():
        gap_text = _gap_text(row)
        if MINUTES_GAP in gap_text:
            subtype = _text(row, "minutes_gap_subtype") or "MINUTES_GAP_UNDECOMPOSED"
            severity = _text(row, "minutes_gap_severity") or "MEDIUM"
            fixability = _text(row, "minutes_gap_fixability") or _text(row, "forecastability_gap_fixability") or "UNKNOWN"
            reason = _text(row, "minutes_gap_reason")
            recommended_fix = _text(row, "minutes_gap_recommended_fix")
            gap_family = MINUTES_GAP
        elif USAGE_GAP in gap_text:
            subtype = _text(row, "usage_gap_subtype") or "USAGE_GAP_UNDECOMPOSED"
            severity = _text(row, "usage_gap_severity") or "MEDIUM"
            fixability = _text(row, "usage_gap_fixability") or _text(row, "forecastability_gap_fixability") or "UNKNOWN"
            reason = _text(row, "usage_gap_reason")
            recommended_fix = _text(row, "usage_gap_recommended_fix")
            gap_family = USAGE_GAP
        else:
            subtype = ""
            severity = _text(row, "forecastability_gap_severity") or "NONE"
            fixability = _text(row, "forecastability_gap_fixability") or "UNKNOWN"
            reason = _text(row, "forecastability_gap_reasons")
            recommended_fix = ""
            gap_family = _text(row, "forecastability_gap_primary")

        fix_upper = str(fixability).upper()
        severity_upper = str(severity).upper()
        one_major_blocker = str(row.get("secondary_blockers", "") or "").strip() in {"", "nan", "None"}
        near_core_after_decomposition = (
            gap_family in {MINUTES_GAP, USAGE_GAP}
            and fix_upper in FIXABLE_LABELS
            and severity_upper not in {"HIGH", "CRITICAL"}
            and one_major_blocker
        )
        permanently_unsafe = fix_upper == "TRUE_UNSTABLE_STATE" or severity_upper == "CRITICAL"

        rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "game_id": row.get("game_id", ""),
                "market_date": row.get("market_date", row.get("game_date", "")),
                "player": row.get("player", row.get("player_name", "")),
                "target": row.get("target", ""),
                "market_type": row.get("market_type", ""),
                "side": row.get("side", row.get("direction", "")),
                "line": row.get("line", row.get("market_line", np.nan)),
                "edge_defendability_tier": row.get("edge_defendability_tier", ""),
                "lcb_edge": _num(row, "lcb_edge"),
                "stress_edge": _num(row, "stress_edge"),
                "forecastability_gap_primary": row.get("forecastability_gap_primary", ""),
                "gap_family": gap_family,
                "gap_subtype": subtype,
                "gap_fixability": fixability,
                "gap_severity": severity,
                "gap_reason": reason,
                "gap_recommended_fix": recommended_fix,
                "near_core_after_decomposition": bool(near_core_after_decomposition),
                "permanently_unsafe": bool(permanently_unsafe),
                "candidate_could_become_near_core_after_data_fix": bool(not permanently_unsafe and fix_upper in FIXABLE_LABELS),
                "recommended_next_action": _recommended_action(fixability, severity, near_core_after_decomposition),
            }
        )

    rows_df = pd.DataFrame.from_records(rows)
    rows_path = output_dir / "forecastability_blocker_resolution_rows.csv"
    rows_df.to_csv(rows_path, index=False)

    minutes_gap_count = int(rows_df["gap_family"].eq(MINUTES_GAP).sum()) if not rows_df.empty else 0
    usage_gap_count = int(rows_df["gap_family"].eq(USAGE_GAP).sum()) if not rows_df.empty else 0
    bucket_counts = rows_df.apply(lambda row: _fixability_bucket(row.get("gap_fixability", ""), row.get("gap_subtype", "")), axis=1).value_counts().to_dict() if not rows_df.empty else {}
    report = {
        "input_paths": {
            "annotated_candidates_csv": str(annotated_candidates_csv),
            "candidate_blockers_csv": str(candidate_blockers_csv) if candidate_blockers_csv else "",
            "minutes_gap_csv": str(minutes_gap_csv) if minutes_gap_csv else "",
            "usage_gap_csv": str(usage_gap_csv) if usage_gap_csv else "",
        },
        "output_paths": {
            "json": str(output_dir / "forecastability_blocker_resolution_report.json"),
            "markdown": str(output_dir / "forecastability_blocker_resolution_report.md"),
            "rows_csv": str(rows_path),
        },
        "total_edge_defendable_candidates": int(edge_defendable.sum()),
        "minutes_gap_count": minutes_gap_count,
        "usage_gap_count": usage_gap_count,
        "true_unstable_count": int(bucket_counts.get("true_unstable", 0)),
        "fixable_with_existing_logs_count": int(bucket_counts.get("fixable_with_existing_logs", 0)),
        "fixable_with_new_pipeline_data_count": int(bucket_counts.get("fixable_with_new_pipeline_data", 0)),
        "needs_more_sample_count": int(bucket_counts.get("needs_more_sample", 0)),
        "feature_missing_count": int(bucket_counts.get("feature_missing", 0)),
        "near_core_candidates_after_decomposition": int(rows_df.get("near_core_after_decomposition", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows_df.empty else 0,
        "candidates_permanently_unsafe": int(rows_df.get("permanently_unsafe", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows_df.empty else 0,
        "candidates_that_could_become_near_core_after_data_fix": int(rows_df.get("candidate_could_become_near_core_after_data_fix", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows_df.empty else 0,
        "gap_subtype_counts": rows_df.get("gap_subtype", pd.Series(dtype=str)).fillna("").replace("", "NONE").value_counts().to_dict() if not rows_df.empty else {},
        "recommended_action_counts": rows_df.get("recommended_next_action", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts().to_dict() if not rows_df.empty else {},
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    (output_dir / "forecastability_blocker_resolution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "forecastability_blocker_resolution_report.md", report, rows_df)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Forecastability Blocker Resolution Report",
        "",
        "## Executive Summary",
        f"- EDGE_DEFENDABLE candidates: {report['total_edge_defendable_candidates']}",
        f"- Minutes-gap candidates: {report['minutes_gap_count']}",
        f"- Usage-gap candidates: {report['usage_gap_count']}",
        f"- True unstable: {report['true_unstable_count']}",
        f"- Fixable with existing logs: {report['fixable_with_existing_logs_count']}",
        f"- Fixable with new pipeline data: {report['fixable_with_new_pipeline_data_count']}",
        f"- Needs more sample: {report['needs_more_sample_count']}",
        f"- Feature missing: {report['feature_missing_count']}",
        f"- Near-core after decomposition: {report['near_core_candidates_after_decomposition']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Gap Subtypes",
    ]
    for subtype, count in report["gap_subtype_counts"].items():
        lines.append(f"- {subtype}: {count}")
    lines.extend(["", "## Candidate Actions"])
    if rows.empty:
        lines.append("- No EDGE_DEFENDABLE forecastability blockers found.")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"- {row.get('player', '')} {row.get('market_type', '')} {row.get('side', '')} {row.get('line', '')}: "
                f"{row.get('gap_subtype', '')} / {row.get('gap_fixability', '')} -> {row.get('recommended_next_action', '')}"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Shadow-only diagnostic.",
            "- No thresholds were relaxed.",
            "- No selector sidecar was materialized.",
            "- No promotion claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve minutes/usage forecastability blockers for EDGE_DEFENDABLE rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--candidate-blockers-csv", type=Path)
    parser.add_argument("--minutes-gap-csv", type=Path)
    parser.add_argument("--usage-gap-csv", type=Path)
    parser.add_argument("--data-proc-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_forecastability_blocker_resolution_report(
        output_dir=args.output_dir,
        annotated_candidates_csv=args.annotated_candidates_csv,
        candidate_blockers_csv=args.candidate_blockers_csv,
        minutes_gap_csv=args.minutes_gap_csv,
        usage_gap_csv=args.usage_gap_csv,
        data_proc_dir=args.data_proc_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

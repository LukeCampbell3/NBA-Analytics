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
from research.safe_state.backfill_minutes_state import (
    _candidate_date,
    _candidate_player,
    _candidate_player_id,
    _load_player_logs,
    _minutes_series,
)


ROOT_CAUSE_TO_REPAIR = {
    "REAL_MINUTES_VOLATILITY": "KEEP_UNSAFE_TRUE_VOLATILITY",
    "REAL_USAGE_VOLATILITY": "KEEP_UNSAFE_TRUE_VOLATILITY",
    "MISSING_MINUTES_FEATURES": "FIX_EXISTING_LOG_PARSE",
    "MISSING_USAGE_FEATURES": "ADD_USAGE_PROXY_COLUMNS",
    "STALE_PLAYER_LOGS": "REFRESH_PLAYER_DATA",
    "BAD_PLAYER_JOIN": "FIX_EXISTING_LOG_PARSE",
    "BAD_DATE_FILTER": "FIX_EXISTING_LOG_PARSE",
    "INSUFFICIENT_COMPARABLE_STATES": "NEEDS_MORE_SAMPLE",
    "TOO_STRICT_COMPARABLE_FILTER": "WIDEN_SIMILAR_STATE_FALLBACK",
    "MISSING_TEAMMATE_CONTEXT": "ADD_TEAMMATE_AVAILABILITY_PIPELINE",
    "MISSING_OPPONENT_CONTEXT": "ADD_OPPONENT_CONTEXT_PIPELINE",
    "MISSING_LINE_DISTRIBUTION": "FIX_EXISTING_LOG_PARSE",
    "THRESHOLD_REVIEW_NEEDED": "MANUAL_REVIEW",
    "UNKNOWN": "MANUAL_REVIEW",
}


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


def _has_any(row: pd.Series, columns: list[str]) -> bool:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return True
    return False


def _merge_resolution(annotated: pd.DataFrame, resolution: pd.DataFrame) -> pd.DataFrame:
    if resolution.empty:
        return annotated
    resolution = candidate_identity_columns(resolution)
    columns = [
        "candidate_id",
        "gap_family",
        "gap_subtype",
        "gap_fixability",
        "gap_severity",
        "gap_reason",
        "gap_recommended_fix",
        "recommended_next_action",
    ]
    return annotated.merge(resolution[[col for col in columns if col in resolution.columns]], on="candidate_id", how="left", suffixes=("", "_resolution"))


def _log_evidence(row: pd.Series, data_proc_dir: Path | None) -> dict[str, Any]:
    player = _candidate_player(row)
    player_id = _candidate_player_id(row)
    market_date = _candidate_date(row)
    logs = _load_player_logs(data_proc_dir, player, player_id=player_id) if data_proc_dir else pd.DataFrame()
    if logs.empty:
        return {
            "player_processed_csv_exists": False,
            "pre_event_log_count": 0,
            "minutes_fields_present_in_logs": False,
            "usage_fields_present_in_logs": False,
            "latest_pre_event_log_date": "",
            "history_staleness_days": np.nan,
            "log_warning": "player_logs_missing_or_unmapped",
        }
    prior = logs.loc[logs["Date"] < market_date].copy() if pd.notna(market_date) else logs.copy()
    latest = prior["Date"].max() if not prior.empty else pd.NaT
    staleness = float((market_date - latest).days) if pd.notna(market_date) and pd.notna(latest) else np.nan
    usage_columns = ["FGA", "FTA", "AST", "TRB", "USG%", "Touches", "Potential_AST", "Potential_Assists"]
    return {
        "player_processed_csv_exists": True,
        "pre_event_log_count": int(len(prior)),
        "minutes_fields_present_in_logs": bool(_minutes_series(prior).notna().any()),
        "usage_fields_present_in_logs": any(col in prior.columns and pd.to_numeric(prior[col], errors="coerce").notna().any() for col in usage_columns),
        "latest_pre_event_log_date": "" if pd.isna(latest) else latest.strftime("%Y-%m-%d"),
        "history_staleness_days": staleness,
        "log_warning": "" if len(prior) else "no_logs_before_market_date",
    }


def _root_cause(row: pd.Series, log: dict[str, Any]) -> tuple[str, str, list[str], list[str], bool, bool, bool, bool, bool]:
    subtype = _text(row, "gap_subtype").upper()
    fixability = _text(row, "gap_fixability").upper()
    missing_features = _text(row, "forecastability_gap_missing_features", "missing_features")
    evidence_found: list[str] = []
    evidence_missing: list[str] = []
    is_real_volatility = False
    is_pipeline_gap = False
    is_join_or_mapping_bug = False
    is_sample_gap = False
    is_external_context_gap = False

    if log.get("player_processed_csv_exists"):
        evidence_found.append("player_processed_csv")
    else:
        evidence_missing.append("player_processed_csv")

    if log.get("minutes_fields_present_in_logs"):
        evidence_found.append("minutes_logs")
    else:
        evidence_missing.append("minutes_logs")

    if log.get("usage_fields_present_in_logs"):
        evidence_found.append("usage_boxscore_proxy_logs")
    else:
        evidence_missing.append("usage_boxscore_proxy_logs")

    if "LOW_FLOOR" in subtype or "WIDE_BAND" in subtype or "HIGH_VOLATILITY" in subtype or "ROLE_UNSTABLE" in subtype:
        primary = "REAL_MINUTES_VOLATILITY"
        secondary = ""
        is_real_volatility = True
    elif "FGA_VOLATILE" in subtype or "AST_OPPORTUNITY_VOLATILE" in subtype or "REBOUND_CHANCE_VOLATILE" in subtype or subtype == "USAGE_ROLE_SHIFT":
        primary = "REAL_USAGE_VOLATILITY"
        secondary = ""
        is_real_volatility = True
    elif subtype == "MINUTES_PIPELINE_MISSING":
        if not log.get("player_processed_csv_exists"):
            primary = "BAD_PLAYER_JOIN"
            secondary = "MISSING_MINUTES_FEATURES"
            is_join_or_mapping_bug = True
        elif not log.get("minutes_fields_present_in_logs"):
            primary = "MISSING_MINUTES_FEATURES"
            secondary = ""
        else:
            primary = "MISSING_MINUTES_FEATURES"
            secondary = "BAD_DATE_FILTER" if log.get("pre_event_log_count", 0) == 0 else ""
        is_pipeline_gap = True
    elif subtype == "USAGE_PIPELINE_MISSING":
        primary = "MISSING_USAGE_FEATURES"
        secondary = ""
        is_pipeline_gap = True
    elif "SAMPLE_INSUFFICIENT" in subtype or fixability == "NEEDS_MORE_SAMPLE":
        primary = "INSUFFICIENT_COMPARABLE_STATES" if "similar" in missing_features.lower() else "INSUFFICIENT_COMPARABLE_STATES"
        secondary = ""
        is_sample_gap = True
    elif subtype == "USAGE_TEAMMATE_DEPENDENT":
        primary = "MISSING_TEAMMATE_CONTEXT"
        secondary = ""
        is_external_context_gap = True
    elif subtype == "USAGE_MATCHUP_DEPENDENT" or "BLOWOUT" in subtype:
        primary = "MISSING_OPPONENT_CONTEXT"
        secondary = ""
        is_external_context_gap = True
    elif "DISTRIBUTION" in missing_features.upper():
        primary = "MISSING_LINE_DISTRIBUTION"
        secondary = ""
        is_pipeline_gap = True
    else:
        primary = "UNKNOWN"
        secondary = ""

    if pd.notna(log.get("history_staleness_days")) and float(log.get("history_staleness_days")) > 7:
        secondary = ";".join([part for part in [secondary, "STALE_PLAYER_LOGS"] if part])

    return (
        primary,
        secondary,
        evidence_found,
        evidence_missing,
        is_real_volatility,
        is_pipeline_gap,
        is_join_or_mapping_bug,
        is_sample_gap,
        is_external_context_gap,
    )


def run_forecastability_root_cause_search(
    *,
    output_dir: Path,
    annotated_candidates_csv: Path,
    resolution_rows_csv: Path,
    data_proc_dir: Path | None = None,
    candidate_pool_csv: Path | None = None,
    market_snapshot: Path | None = None,
    slate_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = candidate_identity_columns(_read_csv(annotated_candidates_csv))
    resolution = _read_csv(resolution_rows_csv)
    working = _merge_resolution(annotated, resolution)
    edge_defendable = working.get("edge_defendability_tier", pd.Series("", index=working.index)).fillna("").astype(str).str.upper().eq("EDGE_DEFENDABLE")
    blocked = working.get("gap_family", pd.Series("", index=working.index)).fillna("").astype(str).str.contains(
        "FORECASTABILITY_GAP_MINUTES_STATE|FORECASTABILITY_GAP_USAGE_STATE", regex=True
    )
    target_rows = working.loc[edge_defendable & blocked].copy()

    records: list[dict[str, Any]] = []
    for _, row in target_rows.iterrows():
        log = _log_evidence(row, data_proc_dir)
        (
            primary,
            secondary,
            evidence_found,
            evidence_missing,
            is_real_volatility,
            is_pipeline_gap,
            is_join_or_mapping_bug,
            is_sample_gap,
            is_external_context_gap,
        ) = _root_cause(row, log)
        recommended_repair = ROOT_CAUSE_TO_REPAIR.get(primary, "MANUAL_REVIEW")
        if primary in {"REAL_MINUTES_VOLATILITY", "REAL_USAGE_VOLATILITY"}:
            priority = "HIGH"
            impact = "candidate_remains_unsafe"
        elif primary in {"MISSING_MINUTES_FEATURES", "MISSING_USAGE_FEATURES", "BAD_PLAYER_JOIN", "BAD_DATE_FILTER"}:
            priority = "HIGH"
            impact = "could_unlock_forecastability_evidence"
        elif primary in {"MISSING_TEAMMATE_CONTEXT", "MISSING_OPPONENT_CONTEXT", "INSUFFICIENT_COMPARABLE_STATES"}:
            priority = "MEDIUM"
            impact = "could_move_candidate_to_near_core_after_recheck"
        else:
            priority = "LOW"
            impact = "manual_review_required"
        records.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "game_id": row.get("game_id", ""),
                "market_date": row.get("market_date", row.get("game_date", "")),
                "player": row.get("player", row.get("player_name", "")),
                "target": row.get("target", ""),
                "market_type": row.get("market_type", ""),
                "side": row.get("side", row.get("direction", "")),
                "line": row.get("line", row.get("market_line", np.nan)),
                "gap_family": row.get("gap_family", ""),
                "gap_subtype": row.get("gap_subtype", ""),
                "gap_fixability": row.get("gap_fixability", ""),
                "root_cause_primary": primary,
                "root_cause_secondary": secondary,
                "evidence_found": ";".join(evidence_found),
                "evidence_missing": ";".join(evidence_missing),
                "is_real_volatility": bool(is_real_volatility),
                "is_pipeline_gap": bool(is_pipeline_gap),
                "is_join_or_mapping_bug": bool(is_join_or_mapping_bug),
                "is_sample_size_gap": bool(is_sample_gap),
                "is_external_context_gap": bool(is_external_context_gap),
                "recommended_repair": recommended_repair,
                "repair_priority": priority,
                "expected_safe_state_impact": impact,
                **log,
            }
        )

    rows = pd.DataFrame.from_records(records)
    rows_path = output_dir / "forecastability_root_cause_rows.csv"
    rows.to_csv(rows_path, index=False)

    report = {
        "input_paths": {
            "annotated_candidates_csv": str(annotated_candidates_csv),
            "resolution_rows_csv": str(resolution_rows_csv),
            "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv else "",
            "market_snapshot": str(market_snapshot) if market_snapshot else "",
            "slate_csv": str(slate_csv) if slate_csv else "",
            "data_proc_dir": str(data_proc_dir) if data_proc_dir else "",
        },
        "output_paths": {
            "json": str(output_dir / "forecastability_root_cause_report.json"),
            "markdown": str(output_dir / "forecastability_root_cause_report.md"),
            "rows_csv": str(rows_path),
        },
        "blocked_edge_defendable_rows": int(len(rows)),
        "root_cause_counts": rows.get("root_cause_primary", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts().to_dict() if not rows.empty else {},
        "recommended_repair_counts": rows.get("recommended_repair", pd.Series(dtype=str)).fillna("MANUAL_REVIEW").value_counts().to_dict() if not rows.empty else {},
        "real_volatility_count": int(rows.get("is_real_volatility", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows.empty else 0,
        "pipeline_gap_count": int(rows.get("is_pipeline_gap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows.empty else 0,
        "join_or_mapping_bug_count": int(rows.get("is_join_or_mapping_bug", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows.empty else 0,
        "sample_size_gap_count": int(rows.get("is_sample_size_gap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows.empty else 0,
        "external_context_gap_count": int(rows.get("is_external_context_gap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not rows.empty else 0,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    (output_dir / "forecastability_root_cause_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "forecastability_root_cause_report.md", report, rows)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Forecastability Root-Cause Search",
        "",
        "## Executive Summary",
        f"- Blocked EDGE_DEFENDABLE rows inspected: {report['blocked_edge_defendable_rows']}",
        f"- Real volatility rows: {report['real_volatility_count']}",
        f"- Pipeline-gap rows: {report['pipeline_gap_count']}",
        f"- Join/mapping bug rows: {report['join_or_mapping_bug_count']}",
        f"- Sample-size gap rows: {report['sample_size_gap_count']}",
        f"- External-context gap rows: {report['external_context_gap_count']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Root Causes",
    ]
    for cause, count in report["root_cause_counts"].items():
        lines.append(f"- {cause}: {count}")
    lines.extend(["", "## Candidate Repairs"])
    if rows.empty:
        lines.append("- No minutes/usage blocked EDGE_DEFENDABLE candidates found.")
    else:
        for _, row in rows.iterrows():
            lines.append(
                f"- {row.get('player', '')} {row.get('market_type', '')} {row.get('side', '')} {row.get('line', '')}: "
                f"{row.get('root_cause_primary', '')} -> {row.get('recommended_repair', '')}"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Root-cause evidence is diagnostic only.",
            "- No threshold relaxation or selector action is made.",
            "- Missing evidence never becomes safety.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect root causes behind minutes/usage forecastability blockers.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--resolution-rows-csv", type=Path, required=True)
    parser.add_argument("--data-proc-dir", type=Path)
    parser.add_argument("--candidate-pool-csv", type=Path)
    parser.add_argument("--market-snapshot", type=Path)
    parser.add_argument("--slate-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_forecastability_root_cause_search(
        output_dir=args.output_dir,
        annotated_candidates_csv=args.annotated_candidates_csv,
        resolution_rows_csv=args.resolution_rows_csv,
        data_proc_dir=args.data_proc_dir,
        candidate_pool_csv=args.candidate_pool_csv,
        market_snapshot=args.market_snapshot,
        slate_csv=args.slate_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

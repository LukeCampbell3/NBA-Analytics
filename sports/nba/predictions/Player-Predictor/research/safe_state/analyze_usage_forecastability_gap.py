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


USAGE_GAP = "FORECASTABILITY_GAP_USAGE_STATE"
TRUE_UNSTABLE = "TRUE_UNSTABLE_STATE"
FIX_NEW_PIPELINE = "FIXABLE_WITH_NEW_PIPELINE_DATA"
NEEDS_MORE_SAMPLE = "NEEDS_MORE_SAMPLE"
FEATURE_MISSING = "FEATURE_MISSING"

USAGE_GAP_COLUMNS = [
    "usage_gap_subtype",
    "usage_gap_primary_driver",
    "usage_gap_severity",
    "usage_gap_fixability",
    "usage_gap_reason",
    "usage_gap_recommended_fix",
    "usage_gap_blocks_safe_state_flag",
]


def _num(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def _text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _target(row: pd.Series) -> str:
    value = _text(row, "target")
    if value:
        return value.upper()
    market_type = _text(row, "market_type", "market_id").upper()
    return market_type.split("_", 1)[0] if "_" in market_type else market_type


def _has_usage_gap(row: pd.Series) -> bool:
    values = [
        _text(row, "forecastability_gap_primary"),
        _text(row, "forecastability_gap_secondary"),
        _text(row, "safe_state_blockers"),
        _text(row, "primary_blocker"),
    ]
    return any(USAGE_GAP in value.upper() for value in values)


def _has_any_usage_field(row: pd.Series) -> bool:
    columns = [
        "recent_fga_mean",
        "recent_fga_std",
        "recent_fga_cv",
        "fga_recent_mean",
        "fga_recent_std",
        "fga_recent_cv",
        "FGA",
        "FGA_mean",
        "FGA_std",
        "FGA_volatility",
        "fga_volatility",
        "recent_ast_cv",
        "assist_opportunity_volatility",
        "potential_assist_volatility",
        "recent_usage_proxy",
        "usage_proxy",
        "usage_volatility",
        "USG%",
        "touches_volatility",
        "rebound_chance_volatility",
        "player_rebound_share_std",
        "teammate_return_risk",
        "teammate_availability_uncertainty",
        "role_shift_risk",
        "opponent_scheme_disruption_score",
    ]
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return True
    return False


def _severity(kind: str, value: float) -> str:
    if kind in {"fga_cv", "ast_cv", "reb_cv", "usage_cv"}:
        if pd.notna(value) and value > 0.55:
            return "CRITICAL"
        return "HIGH"
    if kind == "std":
        if pd.notna(value) and value > 8.0:
            return "CRITICAL"
        return "HIGH"
    if kind in {"teammate", "matchup", "role"}:
        if pd.notna(value) and value > 0.80:
            return "HIGH"
        return "MEDIUM"
    return "MEDIUM"


def _candidate_driver(row: pd.Series) -> dict[str, Any]:
    if not _has_any_usage_field(row):
        return {
            "subtype": "USAGE_PIPELINE_MISSING",
            "driver": "usage_proxy_fields_missing",
            "severity": "MEDIUM",
            "fixability": FEATURE_MISSING,
            "reason": "FGA/touches/potential-assist/rebound-chance usage proxies unavailable",
            "recommended_fix": "ADD_USAGE_PROXY_COLUMNS",
            "blocks": True,
        }

    sample_count = _num(row, "usage_state_sample_count", "usage_sample_count", "recent_games_count")
    if pd.notna(sample_count) and sample_count < 3:
        return {
            "subtype": "USAGE_SAMPLE_INSUFFICIENT",
            "driver": "usage_sample_count",
            "severity": "MEDIUM",
            "fixability": NEEDS_MORE_SAMPLE,
            "reason": f"usage_sample_count={int(sample_count)}",
            "recommended_fix": "NEEDS_MORE_SAMPLE",
            "blocks": True,
        }

    target = _target(row)
    fga_cv = _num(row, "recent_fga_cv", "fga_recent_cv", "FGA_cv", "FGA_volatility", "fga_volatility")
    fga_std = _num(row, "recent_fga_std", "fga_recent_std", "FGA_std")
    ast_cv = _num(row, "recent_ast_cv", "ast_recent_cv", "AST_cv", "assist_opportunity_volatility", "potential_assist_volatility")
    reb_cv = _num(row, "recent_rebound_chance_cv", "rebound_chance_volatility", "player_rebound_share_std")
    usage_cv = _num(row, "recent_usage_cv", "usage_volatility", "usage_proxy_volatility", "touches_volatility")
    role_shift = _num(row, "role_shift_risk", "rotation_volatility_score", "starter_status_change_count")
    teammate = _num(row, "teammate_return_risk", "teammate_availability_uncertainty", "same_team_usage_competition")
    matchup = _num(row, "usage_matchup_dependency", "opponent_scheme_disruption_score", "opponent_context_mismatch_score")

    candidates: list[dict[str, Any]] = []
    if (pd.notna(fga_cv) and fga_cv > 0.30) or (pd.notna(fga_std) and fga_std > 5.0):
        value = fga_cv if pd.notna(fga_cv) else fga_std
        driver = "recent_fga_cv" if pd.notna(fga_cv) else "recent_fga_std"
        candidates.append(
            {
                "subtype": "USAGE_FGA_VOLATILE",
                "driver": driver,
                "severity": _severity("fga_cv" if pd.notna(fga_cv) else "std", value),
                "fixability": TRUE_UNSTABLE,
                "reason": f"{driver}={value:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(ast_cv) and ast_cv > 0.35:
        candidates.append(
            {
                "subtype": "USAGE_AST_OPPORTUNITY_VOLATILE",
                "driver": "assist_opportunity_volatility",
                "severity": _severity("ast_cv", ast_cv),
                "fixability": TRUE_UNSTABLE,
                "reason": f"assist_opportunity_volatility={ast_cv:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(reb_cv) and reb_cv > 0.35:
        candidates.append(
            {
                "subtype": "USAGE_REBOUND_CHANCE_VOLATILE",
                "driver": "rebound_chance_volatility",
                "severity": _severity("reb_cv", reb_cv),
                "fixability": TRUE_UNSTABLE,
                "reason": f"rebound_chance_volatility={reb_cv:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(usage_cv) and usage_cv > 0.35:
        candidates.append(
            {
                "subtype": "USAGE_ROLE_SHIFT",
                "driver": "usage_volatility",
                "severity": _severity("usage_cv", usage_cv),
                "fixability": TRUE_UNSTABLE,
                "reason": f"usage_volatility={usage_cv:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(role_shift) and role_shift > 0.55:
        candidates.append(
            {
                "subtype": "USAGE_ROLE_SHIFT",
                "driver": "role_shift_risk",
                "severity": _severity("role", role_shift),
                "fixability": TRUE_UNSTABLE if role_shift > 0.80 else FIX_NEW_PIPELINE,
                "reason": f"role_shift_risk={role_shift:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY" if role_shift > 0.80 else "ADD_TEAMMATE_AVAILABILITY_PIPELINE",
            }
        )
    if pd.notna(teammate) and teammate > 0.55:
        candidates.append(
            {
                "subtype": "USAGE_TEAMMATE_DEPENDENT",
                "driver": "teammate_availability_context",
                "severity": _severity("teammate", teammate),
                "fixability": FIX_NEW_PIPELINE,
                "reason": f"teammate_dependency_score={teammate:.3f}",
                "recommended_fix": "ADD_TEAMMATE_AVAILABILITY_PIPELINE",
            }
        )
    if pd.notna(matchup) and matchup > 0.55:
        candidates.append(
            {
                "subtype": "USAGE_MATCHUP_DEPENDENT",
                "driver": "opponent_context",
                "severity": _severity("matchup", matchup),
                "fixability": FIX_NEW_PIPELINE,
                "reason": f"matchup_dependency_score={matchup:.3f}",
                "recommended_fix": "ADD_OPPONENT_CONTEXT_PIPELINE",
            }
        )

    if not candidates:
        return {
            "subtype": "USAGE_SAMPLE_INSUFFICIENT",
            "driver": "usage_gap_without_specific_driver",
            "severity": "MEDIUM",
            "fixability": NEEDS_MORE_SAMPLE,
            "reason": "usage_gap_present_but_specific_driver_missing",
            "recommended_fix": "NEEDS_MORE_SAMPLE",
            "blocks": True,
        }

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    target_priority = {
        "PTS": ["USAGE_FGA_VOLATILE", "USAGE_ROLE_SHIFT", "USAGE_TEAMMATE_DEPENDENT", "USAGE_MATCHUP_DEPENDENT"],
        "AST": ["USAGE_AST_OPPORTUNITY_VOLATILE", "USAGE_TEAMMATE_DEPENDENT", "USAGE_ROLE_SHIFT", "USAGE_MATCHUP_DEPENDENT"],
        "TRB": ["USAGE_REBOUND_CHANCE_VOLATILE", "USAGE_TEAMMATE_DEPENDENT", "USAGE_ROLE_SHIFT", "USAGE_MATCHUP_DEPENDENT"],
    }.get(target, [])

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        subtype = str(item["subtype"])
        priority = target_priority.index(subtype) if subtype in target_priority else 99
        return severity_order.get(str(item["severity"]), 9), priority

    winner = sorted(candidates, key=sort_key)[0]
    winner["blocks"] = True
    return winner


def annotate_usage_gap_decomposition(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        if not _has_usage_gap(row):
            rows.append(
                {
                    "usage_gap_subtype": "",
                    "usage_gap_primary_driver": "",
                    "usage_gap_severity": "NONE",
                    "usage_gap_fixability": "",
                    "usage_gap_reason": "",
                    "usage_gap_recommended_fix": "",
                    "usage_gap_blocks_safe_state_flag": False,
                }
            )
            continue
        driver = _candidate_driver(row)
        rows.append(
            {
                "usage_gap_subtype": driver["subtype"],
                "usage_gap_primary_driver": driver["driver"],
                "usage_gap_severity": driver["severity"],
                "usage_gap_fixability": driver["fixability"],
                "usage_gap_reason": driver["reason"],
                "usage_gap_recommended_fix": driver["recommended_fix"],
                "usage_gap_blocks_safe_state_flag": bool(driver["blocks"]),
            }
        )
    frame = pd.DataFrame(rows, index=out.index)
    for column in frame.columns:
        out[column] = frame[column]
    return out


def _write_summary(path: Path, out: pd.DataFrame) -> None:
    payload = {
        "rows": int(len(out)),
        "usage_gap_rows": int(out["usage_gap_subtype"].fillna("").astype(str).str.strip().ne("").sum()),
        "usage_gap_subtype_counts": out["usage_gap_subtype"].fillna("").astype(str).replace("", "NONE").value_counts().to_dict(),
        "usage_gap_fixability_counts": out["usage_gap_fixability"].fillna("").astype(str).replace("", "NONE").value_counts().to_dict(),
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose usage-state forecastability blockers.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = annotate_usage_gap_decomposition(candidates)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.summary_json:
        _write_summary(args.summary_json, out)


if __name__ == "__main__":
    main()

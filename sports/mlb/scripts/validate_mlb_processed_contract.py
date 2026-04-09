#!/usr/bin/env python3
"""
Validate MLB processed player files against the isolated MLB schema contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_REQUIRED = [
    "Date",
    "Player",
    "Player_Type",
    "Team",
    "Opponent",
    "Season",
    "Game_ID",
    "Game_Index",
    "Team_ID",
    "Opponent_ID",
    "Is_Home",
    "Did_Not_Play",
    "Rest_Days",
    "Month_sin",
    "Month_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Market_Fetched_At_UTC",
]

HITTER_REQUIRED = [
    "H",
    "HR",
    "RBI",
    "PA",
    "AB",
    "BB",
    "SO",
    "Batting_Order",
    "Team_PA_share",
    "wOBA",
    "xwOBA",
    "ISO",
    "Barrel%",
    "HardHit%",
    "Opp_Pitcher_ERA_3",
    "Opp_Pitcher_K9_3",
    "Opp_Bullpen_ERA_7",
    "Park_Factor",
    "Wind_Out_MPH",
    "Temp_F",
    "H_rolling_avg",
    "HR_rolling_avg",
    "RBI_rolling_avg",
    "H_lag1",
    "HR_lag1",
    "RBI_lag1",
    "Market_H",
    "Market_HR",
    "Market_RBI",
    "Synthetic_Market_H",
    "Synthetic_Market_HR",
    "Synthetic_Market_RBI",
    "Market_Source_H",
    "Market_Source_HR",
    "Market_Source_RBI",
    "Market_H_books",
    "Market_HR_books",
    "Market_RBI_books",
    "Market_H_over_price",
    "Market_HR_over_price",
    "Market_RBI_over_price",
    "Market_H_under_price",
    "Market_HR_under_price",
    "Market_RBI_under_price",
    "Market_H_line_std",
    "Market_HR_line_std",
    "Market_RBI_line_std",
    "H_market_gap",
    "HR_market_gap",
    "RBI_market_gap",
]

PITCHER_REQUIRED = [
    "K",
    "ER",
    "ERA",
    "IP",
    "BF",
    "Pitches",
    "BB_allowed",
    "H_allowed",
    "HR_allowed",
    "FIP",
    "xFIP",
    "CSW%",
    "Whiff%",
    "Opp_Lineup_wOBA_3",
    "Opp_Lineup_K_rate_3",
    "Park_Factor",
    "Wind_Out_MPH",
    "Temp_F",
    "K_rolling_avg",
    "ER_rolling_avg",
    "ERA_rolling_avg",
    "K_lag1",
    "ER_lag1",
    "ERA_lag1",
    "Market_K",
    "Market_ER",
    "Market_ERA",
    "Synthetic_Market_K",
    "Synthetic_Market_ER",
    "Synthetic_Market_ERA",
    "Market_Source_K",
    "Market_Source_ER",
    "Market_Source_ERA",
    "Market_K_books",
    "Market_ER_books",
    "Market_ERA_books",
    "Market_K_over_price",
    "Market_ER_over_price",
    "Market_ERA_over_price",
    "Market_K_under_price",
    "Market_ER_under_price",
    "Market_ERA_under_price",
    "Market_K_line_std",
    "Market_ER_line_std",
    "Market_ERA_line_std",
    "K_market_gap",
    "ER_market_gap",
    "ERA_market_gap",
]

COMMON_NUMERIC = [
    "Season",
    "Game_Index",
    "Team_ID",
    "Opponent_ID",
    "Is_Home",
    "Did_Not_Play",
    "Rest_Days",
    "Month_sin",
    "Month_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
]

HITTER_NUMERIC = [
    "H",
    "HR",
    "RBI",
    "PA",
    "AB",
    "BB",
    "SO",
    "Batting_Order",
    "Team_PA_share",
    "wOBA",
    "xwOBA",
    "ISO",
    "Barrel%",
    "HardHit%",
    "Opp_Pitcher_ERA_3",
    "Opp_Pitcher_K9_3",
    "Opp_Bullpen_ERA_7",
    "Park_Factor",
    "Wind_Out_MPH",
    "Temp_F",
    "H_rolling_avg",
    "HR_rolling_avg",
    "RBI_rolling_avg",
    "H_lag1",
    "HR_lag1",
    "RBI_lag1",
    "Market_H",
    "Market_HR",
    "Market_RBI",
    "Synthetic_Market_H",
    "Synthetic_Market_HR",
    "Synthetic_Market_RBI",
    "Market_H_books",
    "Market_HR_books",
    "Market_RBI_books",
    "Market_H_over_price",
    "Market_HR_over_price",
    "Market_RBI_over_price",
    "Market_H_under_price",
    "Market_HR_under_price",
    "Market_RBI_under_price",
    "Market_H_line_std",
    "Market_HR_line_std",
    "Market_RBI_line_std",
    "H_market_gap",
    "HR_market_gap",
    "RBI_market_gap",
]

PITCHER_NUMERIC = [
    "K",
    "ER",
    "ERA",
    "IP",
    "BF",
    "Pitches",
    "BB_allowed",
    "H_allowed",
    "HR_allowed",
    "FIP",
    "xFIP",
    "CSW%",
    "Whiff%",
    "Opp_Lineup_wOBA_3",
    "Opp_Lineup_K_rate_3",
    "Park_Factor",
    "Wind_Out_MPH",
    "Temp_F",
    "K_rolling_avg",
    "ER_rolling_avg",
    "ERA_rolling_avg",
    "K_lag1",
    "ER_lag1",
    "ERA_lag1",
    "Market_K",
    "Market_ER",
    "Market_ERA",
    "Synthetic_Market_K",
    "Synthetic_Market_ER",
    "Synthetic_Market_ERA",
    "Market_K_books",
    "Market_ER_books",
    "Market_ERA_books",
    "Market_K_over_price",
    "Market_ER_over_price",
    "Market_ERA_over_price",
    "Market_K_under_price",
    "Market_ER_under_price",
    "Market_ERA_under_price",
    "Market_K_line_std",
    "Market_ER_line_std",
    "Market_ERA_line_std",
    "K_market_gap",
    "ER_market_gap",
    "ERA_market_gap",
]

ALLOWED_SOURCES = {"real", "synthetic", "baseline_fallback", "missing"}


def _find_processed_files(data_dir: Path) -> list[Path]:
    out: list[Path] = []
    for player_dir in sorted(data_dir.iterdir()):
        if not player_dir.is_dir():
            continue
        candidates = sorted(player_dir.glob("*_processed_processed.csv"))
        if not candidates:
            candidates = sorted(player_dir.glob("*_processed.csv"))
        out.extend(candidates)
    return out


def _detect_player_type(df: pd.DataFrame) -> str:
    if "Player_Type" in df.columns:
        values = (
            df["Player_Type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan})
            .dropna()
            .unique()
            .tolist()
        )
        if len(values) == 1 and values[0] in {"hitter", "pitcher"}:
            return values[0]
    hitter_hits = sum(1 for col in HITTER_REQUIRED if col in df.columns)
    pitcher_hits = sum(1 for col in PITCHER_REQUIRED if col in df.columns)
    return "hitter" if hitter_hits >= pitcher_hits else "pitcher"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _check_numeric(df: pd.DataFrame, columns: list[str], issues: list[str]) -> None:
    for col in columns:
        numeric = _to_numeric(df[col])
        if numeric.isna().any():
            issues.append(f"non_numeric_or_nan:{col}")
            continue
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            issues.append(f"non_finite_values:{col}")


def _check_file(path: Path, min_rows: int) -> tuple[bool, dict]:
    issues: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, {"file": str(path), "rows": 0, "issues": [f"failed_to_read:{exc}"]}

    if df.empty:
        issues.append("file_is_empty")
        return False, {"file": str(path), "rows": 0, "issues": issues}

    missing_common = [col for col in COMMON_REQUIRED if col not in df.columns]
    if missing_common:
        issues.append(f"missing_common_columns={missing_common}")
        return False, {"file": str(path), "rows": int(len(df)), "issues": issues}

    if len(df) < int(min_rows):
        issues.append(f"insufficient_rows={len(df)}<min_rows={min_rows}")

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    if parsed_dates.isna().any():
        issues.append("invalid_date_values")

    game_idx = _to_numeric(df["Game_Index"])
    if game_idx.isna().any():
        issues.append("invalid_game_index_values")
    elif (game_idx.diff().fillna(1) <= 0).any():
        issues.append("game_index_not_strictly_increasing")

    dnp = _to_numeric(df["Did_Not_Play"])
    if dnp.isna().any() or (~dnp.isin([0, 1])).any():
        issues.append("did_not_play_must_be_0_or_1")

    player_type = _detect_player_type(df)
    role_required = HITTER_REQUIRED if player_type == "hitter" else PITCHER_REQUIRED
    role_numeric = HITTER_NUMERIC if player_type == "hitter" else PITCHER_NUMERIC
    missing_role = [col for col in role_required if col not in df.columns]
    if missing_role:
        issues.append(f"missing_{player_type}_columns={missing_role}")
        return False, {"file": str(path), "rows": int(len(df)), "player_type": player_type, "issues": issues}

    _check_numeric(df, COMMON_NUMERIC, issues)
    _check_numeric(df, role_numeric, issues)

    if player_type == "hitter":
        source_cols = ["Market_Source_H", "Market_Source_HR", "Market_Source_RBI"]
        std_cols = ["Market_H_line_std", "Market_HR_line_std", "Market_RBI_line_std"]
    else:
        source_cols = ["Market_Source_K", "Market_Source_ER", "Market_Source_ERA"]
        std_cols = ["Market_K_line_std", "Market_ER_line_std", "Market_ERA_line_std"]

    for col in source_cols:
        bad = ~df[col].astype(str).str.lower().isin(ALLOWED_SOURCES)
        if bad.any():
            issues.append(f"invalid_market_source_values:{col}")

    for col in std_cols:
        if (_to_numeric(df[col]) < 0).any():
            issues.append(f"negative_market_line_std:{col}")

    return len(issues) == 0, {
        "file": str(path),
        "rows": int(len(df)),
        "player_type": player_type,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MLB processed files for isolated MLB schema contract.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed",
        help="Directory containing per-player processed CSV files.",
    )
    parser.add_argument("--min-rows", type=int, default=11, help="Minimum required rows per player file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}")
        return 2

    files = _find_processed_files(data_dir)
    if not files:
        print(f"ERROR: no processed files found in {data_dir}")
        return 2

    reports = []
    ok_count = 0
    total_rows = 0
    for path in files:
        passed, report = _check_file(path, min_rows=int(args.min_rows))
        reports.append(report)
        total_rows += int(report.get("rows", 0))
        if passed:
            ok_count += 1

    failed = [r for r in reports if r["issues"]]
    summary = {
        "data_dir": str(data_dir),
        "files_checked": len(files),
        "files_passed": ok_count,
        "files_failed": len(failed),
        "total_rows": total_rows,
    }

    if args.json:
        print(json.dumps({"summary": summary, "reports": reports}, indent=2))
    else:
        print("MLB processed contract validation")
        print(f"- data_dir: {summary['data_dir']}")
        print(f"- files_checked: {summary['files_checked']}")
        print(f"- files_passed: {summary['files_passed']}")
        print(f"- files_failed: {summary['files_failed']}")
        print(f"- total_rows: {summary['total_rows']}")
        if failed:
            print("\nFailed files:")
            for report in failed:
                print(f"- {report['file']} [{report.get('player_type', 'unknown')}]")
                for issue in report["issues"]:
                    print(f"  - {issue}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

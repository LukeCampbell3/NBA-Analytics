#!/usr/bin/env python3
"""
Replay dated market snapshots and grade the resulting final boards against actual outcomes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = REPO_ROOT / "model" / "analysis"
DAILY_RUNS_ROOT = ANALYSIS_ROOT / "daily_runs"
DEFAULT_HISTORY_CSV = ANALYSIS_ROOT / "refreshed_market_comparison_strict_rows.csv"
VALIDATION_ROOT = ANALYSIS_ROOT / "historical_validation"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
TARGETS = ["PTS", "TRB", "AST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate historical daily market runs against actual outcomes.")
    parser.add_argument("--season", type=int, default=2026, help="Season end year.")
    parser.add_argument("--dates", nargs="+", required=True, help="Run dates in YYYY-MM-DD format.")
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV, help="Historical calibration CSV to use when replaying the slate.")
    parser.add_argument("--policy-profile", type=str, default="production_calibrated", help="Pipeline policy profile to validate.")
    parser.add_argument("--output-tag", type=str, default=None, help="Optional folder/tag name for this validation run.")
    parser.add_argument("--history-wide-path", type=Path, default=MARKET_ROOT / "history_player_props_wide.parquet", help="Wide market history used to reconstruct dated snapshots.")
    parser.add_argument("--reconstruct-snapshot-from-history", action="store_true", help="Rebuild each dated source snapshot from history_player_props_wide instead of trusting the frozen daily parquet.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    return parser.parse_args()


def _stamp(run_date: str) -> str:
    return pd.Timestamp(run_date).strftime("%Y%m%d")


def _load_player_frame(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if path not in cache:
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError(f"Player CSV is missing Date: {path}")
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        cache[path] = df
    return cache[path]


def _extract_matchup_teams(value: str | None) -> set[str]:
    if not value:
        return set()
    tokens = [item for item in str(value).replace("vs.", "@").replace("vs", "@").split("@") if item]
    teams: set[str] = set()
    for token in tokens:
        team = token.strip().split()[0].upper()
        if 2 <= len(team) <= 3:
            teams.add(team)
    return teams


def _load_snapshot_metadata(snapshot_path: Path | None) -> dict[tuple[str, str, str], dict]:
    if snapshot_path is None or not snapshot_path.exists():
        return {}
    if snapshot_path.suffix.lower() == ".parquet":
        snapshot_df = pd.read_parquet(snapshot_path)
    else:
        snapshot_df = pd.read_csv(snapshot_path)
    metadata: dict[tuple[str, str, str], dict] = {}
    for _, row in snapshot_df.iterrows():
        player = str(row.get("Player", ""))
        market_date = str(row.get("Market_Date", ""))
        for target in TARGETS:
            market_line = pd.to_numeric(pd.Series([row.get(f"Market_{target}")]), errors="coerce").iloc[0]
            if pd.isna(market_line):
                continue
            metadata[(player, market_date, target)] = {
                "market_player_raw": row.get("Market_Player_Raw"),
                "market_event_id": row.get("Market_Event_ID"),
                "market_commence_time_utc": row.get("Market_Commence_Time_UTC"),
                "market_home_team": row.get("Market_Home_Team"),
                "market_away_team": row.get("Market_Away_Team"),
            }
    return metadata


def _effective_event_date(row: pd.Series) -> pd.Timestamp:
    commence = pd.to_datetime(row.get("market_commence_time_utc"), errors="coerce", utc=True)
    if pd.notna(commence):
        return commence.tz_convert("America/New_York").normalize()
    return pd.to_datetime(row.get("market_date"), errors="coerce").normalize()


def _resolve_actual_row(player_df: pd.DataFrame, event_date: pd.Timestamp, market_teams: set[str]) -> tuple[pd.Series | None, str]:
    exact = player_df.loc[player_df["Date"] == event_date].copy()
    if not exact.empty:
        if market_teams and "MATCHUP" in exact.columns:
            exact["matchup_teams"] = exact["MATCHUP"].map(_extract_matchup_teams)
            exact = exact.loc[exact["matchup_teams"].map(lambda item: market_teams.issubset(item) or market_teams == item)]
        if not exact.empty:
            return exact.iloc[-1], "exact_event_date"
    if market_teams and "MATCHUP" in player_df.columns and pd.notna(event_date):
        nearby = player_df.loc[player_df["Date"].between(event_date - pd.Timedelta(days=1), event_date + pd.Timedelta(days=1))].copy()
        if not nearby.empty:
            nearby["matchup_teams"] = nearby["MATCHUP"].map(_extract_matchup_teams)
            nearby = nearby.loc[nearby["matchup_teams"].map(lambda item: market_teams.issubset(item) or market_teams == item)]
            if not nearby.empty:
                nearby["day_delta"] = (nearby["Date"] - event_date).abs()
                nearby = nearby.sort_values(["day_delta", "Date"])
                return nearby.iloc[0], "nearby_team_match"
    return None, "missing"


def grade_board(board_path: Path, snapshot_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    board = pd.read_csv(board_path)
    if board.empty:
        return board, {"rows": 0, "wins": 0, "losses": 0, "pushes": 0, "graded_rows": 0, "win_rate": None}

    player_cache: dict[Path, pd.DataFrame] = {}
    snapshot_metadata = _load_snapshot_metadata(snapshot_path)
    graded_rows: list[dict] = []
    for _, row in board.iterrows():
        csv_path = Path(str(row.get("csv", "")))
        target = str(row.get("target"))
        direction = str(row.get("direction")).upper()
        market_line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
        actual_value = None
        result = "missing"
        match_mode = "missing"

        payload = dict(row)
        metadata_key = (str(payload.get("player", "")), str(payload.get("market_date", "")), target)
        for key, value in snapshot_metadata.get(metadata_key, {}).items():
            if key not in payload or pd.isna(payload.get(key)):
                payload[key] = value

        event_date = _effective_event_date(pd.Series(payload))
        market_teams = {str(item) for item in [payload.get("market_home_team"), payload.get("market_away_team")] if pd.notna(item) and str(item)}

        if csv_path.exists() and pd.notna(event_date) and target:
            player_df = _load_player_frame(csv_path, player_cache)
            match_row, match_mode = _resolve_actual_row(player_df, event_date, market_teams)
            if match_row is not None and target in player_df.columns:
                actual_value = pd.to_numeric(match_row[target], errors="coerce")
                if pd.notna(actual_value) and pd.notna(market_line):
                    if float(actual_value) == float(market_line):
                        result = "push"
                    elif direction == "OVER":
                        result = "win" if float(actual_value) > float(market_line) else "loss"
                    elif direction == "UNDER":
                        result = "win" if float(actual_value) < float(market_line) else "loss"
                    else:
                        result = "invalid_direction"

        payload["graded_event_date"] = None if pd.isna(event_date) else str(event_date.date())
        payload["grade_match_mode"] = match_mode
        payload["actual_value"] = None if actual_value is None or pd.isna(actual_value) else float(actual_value)
        payload["result"] = result
        graded_rows.append(payload)

    graded = pd.DataFrame.from_records(graded_rows)
    wins = int((graded["result"] == "win").sum())
    losses = int((graded["result"] == "loss").sum())
    pushes = int((graded["result"] == "push").sum())
    graded_rows_count = wins + losses + pushes
    decisions = wins + losses
    return graded, {
        "rows": int(len(graded)),
        "graded_rows": int(graded_rows_count),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": (wins / decisions) if decisions else None,
    }


def reconstruct_snapshot_from_history(manifest_path: Path, history_wide_path: Path, out_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not history_wide_path.exists():
        raise FileNotFoundError(f"History wide file not found: {history_wide_path}")
    history_df = pd.read_parquet(history_wide_path) if history_wide_path.suffix.lower() == ".parquet" else pd.read_csv(history_wide_path)
    if history_df.empty:
        raise RuntimeError(f"History wide file is empty: {history_wide_path}")
    history_df = history_df.copy()
    history_df["Market_Date"] = pd.to_datetime(history_df["Market_Date"], errors="coerce")
    history_df["Market_Fetched_At_UTC"] = pd.to_datetime(history_df["Market_Fetched_At_UTC"], errors="coerce", utc=True)
    snapshot_meta = manifest.get("current_market_snapshot_meta", {})
    start_date = pd.to_datetime(snapshot_meta.get("requested_start_date"), errors="coerce")
    end_date = pd.to_datetime(snapshot_meta.get("requested_end_date"), errors="coerce")
    cutoff = pd.to_datetime(manifest.get("updated_at_utc"), errors="coerce", utc=True)
    rebuilt = history_df.loc[history_df["Market_Date"].between(start_date, end_date)].copy()
    rebuilt = rebuilt.loc[rebuilt["Market_Fetched_At_UTC"].notna() & (rebuilt["Market_Fetched_At_UTC"] <= cutoff)].copy()
    rebuilt = rebuilt.sort_values(["Market_Date", "Player", "Market_Fetched_At_UTC"]).drop_duplicates(subset=["Market_Date", "Player"], keep="last")
    if rebuilt.empty:
        raise RuntimeError(f"Reconstructed snapshot is empty for manifest {manifest_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        rebuilt.to_parquet(out_path, index=False)
    else:
        rebuilt.to_csv(out_path, index=False)
    return out_path


def replay_date(
    run_date: str,
    season: int,
    history_csv: Path,
    python_exe: str,
    policy_profile: str,
    output_tag: str | None,
    reconstruct_snapshot: bool,
    history_wide_path: Path,
) -> dict:
    stamp = _stamp(run_date)
    daily_dir = DAILY_RUNS_ROOT / stamp
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily run directory not found: {daily_dir}")

    source_snapshot = daily_dir / f"current_market_snapshot_{stamp}.parquet"
    old_final = daily_dir / f"final_market_plays_{stamp}.csv"
    manifest_path = daily_dir / f"daily_market_pipeline_manifest_{stamp}.json"
    if not source_snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {source_snapshot}")
    if not old_final.exists():
        raise FileNotFoundError(f"Existing final board not found: {old_final}")
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")
    if reconstruct_snapshot and not manifest_path.exists():
        raise FileNotFoundError(f"Daily manifest not found: {manifest_path}")

    tag = output_tag or policy_profile
    out_dir = VALIDATION_ROOT / tag / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_snapshot = source_snapshot
    if reconstruct_snapshot:
        replay_snapshot = reconstruct_snapshot_from_history(
            manifest_path,
            history_wide_path,
            out_dir / f"{tag}_reconstructed_market_snapshot_{stamp}.parquet",
        )
    new_slate = out_dir / f"{tag}_upcoming_market_slate_{stamp}.csv"
    new_selector = out_dir / f"{tag}_upcoming_market_play_selector_{stamp}.csv"
    new_final = out_dir / f"{tag}_final_market_plays_{stamp}.csv"
    new_json = out_dir / f"{tag}_final_market_plays_{stamp}.json"

    subprocess.run(
        [
            python_exe,
            "scripts/run_market_pipeline.py",
            "--season",
            str(season),
            "--latest",
            "--policy-profile",
            str(policy_profile),
            "--history-csv",
            str(history_csv),
            "--market-wide-path",
            str(replay_snapshot),
            "--slate-csv-out",
            str(new_slate),
            "--selector-csv-out",
            str(new_selector),
            "--final-csv-out",
            str(new_final),
            "--final-json-out",
            str(new_json),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    old_graded, old_summary = grade_board(old_final, snapshot_path=replay_snapshot)
    new_graded, new_summary = grade_board(new_final, snapshot_path=replay_snapshot)
    old_graded_path = out_dir / f"graded_original_final_market_plays_{stamp}.csv"
    new_graded_path = out_dir / f"graded_{tag}_final_market_plays_{stamp}.csv"
    old_graded.to_csv(old_graded_path, index=False)
    new_graded.to_csv(new_graded_path, index=False)

    return {
        "run_date": run_date,
        "stamp": stamp,
        "snapshot": str(replay_snapshot),
        "snapshot_mode": "reconstructed_from_history" if reconstruct_snapshot else "daily_saved_snapshot",
        "history_csv": str(history_csv),
        "policy_profile": policy_profile,
        "original_final_csv": str(old_final),
        "latest_final_csv": str(new_final),
        "graded_original_csv": str(old_graded_path),
        "graded_latest_csv": str(new_graded_path),
        "original": old_summary,
        "latest": new_summary,
    }


def main() -> None:
    args = parse_args()
    summaries = [
        replay_date(
            run_date,
            args.season,
            args.history_csv.resolve(),
            args.python,
            args.policy_profile,
            args.output_tag,
            args.reconstruct_snapshot_from_history,
            args.history_wide_path.resolve(),
        )
        for run_date in args.dates
    ]

    tag = args.output_tag or args.policy_profile
    summary_path = VALIDATION_ROOT / tag / "historical_daily_validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"dates": summaries}, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("HISTORICAL DAILY VALIDATION")
    print("=" * 90)
    for item in summaries:
        print(f"{item['run_date']}:")
        print(
            "  original -> "
            f"{item['original']['wins']}/{item['original']['wins'] + item['original']['losses']}"
            f" (pushes={item['original']['pushes']})"
        )
        print(
            "  latest   -> "
            f"{item['latest']['wins']}/{item['latest']['wins'] + item['latest']['losses']}"
            f" (pushes={item['latest']['pushes']})"
        )
        print(f"  latest csv: {item['latest_final_csv']}")
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.player_simulation.discover_pre_cutoff_player_logs import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DISCOVERY_DIR,
    discover_pre_cutoff_player_logs,
)
from research.player_simulation.simulate_next_season_player_states import _normalize_logs


DEFAULT_BACKTEST_DIR = PLAYER_PREDICTOR_ROOT.parents[1] / "validation" / "production_shadow" / "player_simulation" / "backtests" / "2025_preseason"
DEFAULT_FROZEN_DIR = DEFAULT_BACKTEST_DIR / "frozen_sample"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_source(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    if frame.empty:
        return frame
    return _normalize_logs(frame, fallback_player=path.parent.name.replace("_", " "))


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = _num(frame[column]).dropna()
    return None if values.empty else float(values.mean())


def _std(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = _num(frame[column]).dropna()
    return None if values.size < 2 else float(values.std(ddof=0))


def _last_text(frame: pd.DataFrame, *columns: str) -> str:
    for column in columns:
        if column in frame.columns and frame[column].notna().any():
            return str(frame[column].dropna().astype(str).iloc[-1])
    return ""


def _actual_players(actual_outcomes: pd.DataFrame) -> pd.DataFrame:
    if actual_outcomes.empty:
        return pd.DataFrame(columns=["player", "player_id"])
    cols = [col for col in ["player", "Player", "player_id", "Player_ID"] if col in actual_outcomes.columns]
    players = actual_outcomes[cols].copy()
    if "player" not in players.columns and "Player" in players.columns:
        players["player"] = players["Player"]
    if "player_id" not in players.columns and "Player_ID" in players.columns:
        players["player_id"] = players["Player_ID"]
    return players[["player", "player_id"]].drop_duplicates()


def _state_row(player: str, group: pd.DataFrame, *, cutoff: pd.Timestamp, min_games: int) -> dict[str, Any]:
    group = group.sort_values("Date").copy()
    recent = group.tail(min(25, len(group)))
    minutes_mean = _mean(recent, "MIN")
    minutes_std = _std(recent, "MIN")
    pts_mean = _mean(recent, "PTS")
    pts_std = _std(recent, "PTS")
    reb_mean = _mean(recent, "REB")
    reb_std = _std(recent, "REB")
    ast_mean = _mean(recent, "AST")
    ast_std = _std(recent, "AST")
    warnings: list[str] = []
    for column in ["MIN", "PTS", "REB", "AST"]:
        if column not in group.columns or _num(group[column]).notna().sum() < min_games:
            warnings.append(f"insufficient_{column.lower()}_history")
    if "FGA" not in group.columns:
        warnings.append("missing_fga")
    if "FTA" not in group.columns:
        warnings.append("missing_fta")
    minutes_cv = None if not minutes_mean else float((minutes_std or 0.0) / max(minutes_mean, 1.0))
    primary_vols = []
    for mean, std in [(pts_mean, pts_std), (reb_mean, reb_std), (ast_mean, ast_std)]:
        if mean is not None:
            primary_vols.append(float((std or 0.0) / max(mean, 1.0)))
    stat_vol = float(np.nanmean(primary_vols)) if primary_vols else 1.0
    volatility_score = float(np.clip(0.45 * (minutes_cv if minutes_cv is not None else 1.0) + 0.55 * stat_vol, 0.0, 1.0))
    role_stability_score = float(np.clip(1.0 - (minutes_cv if minutes_cv is not None else 1.0), 0.0, 1.0))
    forecastability_score = float(np.clip(0.55 * role_stability_score + 0.45 * (1.0 - volatility_score), 0.0, 1.0))
    eligible = bool(len(group) >= min_games and minutes_mean is not None and pts_mean is not None and reb_mean is not None and ast_mean is not None)
    return {
        "player_id": _last_text(group, "Player_ID", "player_id"),
        "player": player,
        "team_prior": _last_text(group, "Team", "team"),
        "position": _last_text(group, "Pos", "position"),
        "data_cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "max_input_game_date": group["Date"].max().strftime("%Y-%m-%d") if group["Date"].notna().any() else "",
        "max_source_date": group["Date"].max().strftime("%Y-%m-%d") if group["Date"].notna().any() else "",
        "seasons_available_before_cutoff": int(group["Date"].dt.year.nunique()),
        "games_available_before_cutoff": int(len(group)),
        "prior_season_games": int(len(group)),
        "prior_season_mpg": _mean(group, "MIN"),
        "prior_season_minutes": _mean(group, "MIN"),
        "prior_season_pts": _mean(group, "PTS"),
        "prior_season_reb": _mean(group, "REB"),
        "prior_season_ast": _mean(group, "AST"),
        "prior_season_pra": _mean(group, "PRA"),
        "prior_season_fga": _mean(group, "FGA"),
        "prior_season_fta": _mean(group, "FTA"),
        "recent_minutes_mean": minutes_mean,
        "recent_minutes_std": minutes_std,
        "recent_pts_mean": pts_mean,
        "recent_pts_std": pts_std,
        "recent_reb_mean": reb_mean,
        "recent_reb_std": reb_std,
        "recent_ast_mean": ast_mean,
        "recent_ast_std": ast_std,
        "minutes_mean": minutes_mean,
        "minutes_std": minutes_std,
        "pts_mean": pts_mean,
        "pts_std": pts_std,
        "reb_mean": reb_mean,
        "reb_std": reb_std,
        "ast_mean": ast_mean,
        "ast_std": ast_std,
        "stl_mean": _mean(recent, "STL"),
        "stl_std": _std(recent, "STL"),
        "blk_mean": _mean(recent, "BLK"),
        "blk_std": _std(recent, "BLK"),
        "threepm_mean": _mean(recent, "3PM"),
        "threepm_std": _std(recent, "3PM"),
        "usage_fga_mean": _mean(recent, "FGA"),
        "usage_fga_std": _std(recent, "FGA"),
        "volatility_score": volatility_score,
        "role_stability_score": role_stability_score,
        "forecastability_score": forecastability_score,
        "missing_feature_warnings": "|".join(sorted(set(warnings))),
        "simulation_eligible": eligible,
        "ineligible_reason": "" if eligible else "insufficient_pre_cutoff_history",
    }


def backfill_pre_cutoff_player_state_logs(
    *,
    output_dir: Path = DEFAULT_FROZEN_DIR,
    discovery_csv: Path | None = None,
    actual_outcomes: Path | None = None,
    cutoff_date: str = "2025-10-01",
    min_games: int = 10,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    discovery_csv = discovery_csv or (DEFAULT_DISCOVERY_DIR / "pre_cutoff_available_sources.csv")
    if not discovery_csv.exists():
        discover_pre_cutoff_player_logs(output_dir=DEFAULT_DISCOVERY_DIR, cutoff_date=cutoff_date)
    sources = _read_table(discovery_csv)
    usable = sources.loc[sources.get("usable_pre_cutoff_rows", pd.Series(dtype=int)).fillna(0).astype(int).gt(0)].copy() if not sources.empty else pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path_str in usable.get("path", pd.Series(dtype=str)).dropna().astype(str).tolist():
        frame = _read_source(Path(path_str))
        if frame.empty or "Date" not in frame.columns:
            continue
        frame = frame.loc[pd.to_datetime(frame["Date"], errors="coerce").lt(cutoff)].copy()
        if not frame.empty:
            frames.append(frame)
    logs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not logs.empty:
        logs["Date"] = pd.to_datetime(logs["Date"], errors="coerce")
        key_cols = [col for col in ["Player_ID", "Player", "Date", "Team", "Opponent"] if col in logs.columns]
        if key_cols:
            logs = logs.drop_duplicates(key_cols)

    actual_path = actual_outcomes or (output_dir / "frozen_preseason_actual_outcomes.csv")
    actual_df = _read_table(actual_path)
    actual_players = _actual_players(actual_df)
    actual_names = set(actual_players.get("player", pd.Series(dtype=str)).fillna("").astype(str))

    rows = []
    if not logs.empty and "Player" in logs.columns:
        for player, group in logs.groupby(logs["Player"].fillna("Unknown Player").astype(str), dropna=False):
            if actual_names and player not in actual_names:
                continue
            rows.append(_state_row(player, group, cutoff=cutoff, min_games=int(min_games)))
    state_df = pd.DataFrame(rows)
    state_csv = output_dir / "frozen_preseason_player_state_rows.csv"
    state_parquet = output_dir / "frozen_preseason_player_state_rows.parquet"
    ineligible_csv = output_dir / "frozen_preseason_ineligible_players.csv"
    state_df.to_csv(state_csv, index=False)
    try:
        state_df.to_parquet(state_parquet, index=False)
    except Exception:
        state_parquet.write_text("", encoding="utf-8")

    eligible_names = set(state_df.loc[state_df.get("simulation_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool), "player"].astype(str)) if not state_df.empty else set()
    all_state_names = set(state_df.get("player", pd.Series(dtype=str)).astype(str)) if not state_df.empty else set()
    ineligible_rows = []
    for _, row in actual_players.iterrows():
        player = str(row.get("player", ""))
        if not player:
            continue
        if player not in eligible_names:
            reason = "insufficient_pre_cutoff_history" if player in all_state_names else "no_pre_cutoff_history_found"
            ineligible_rows.append({"player": player, "player_id": row.get("player_id", ""), "ineligible_reason": reason})
    pd.DataFrame(ineligible_rows).to_csv(ineligible_csv, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "min_games": int(min_games),
        "discovery_csv": str(discovery_csv),
        "actual_outcomes": str(actual_path),
        "discovered_usable_source_count": int(len(usable)),
        "frozen_input_rows": int(len(state_df)),
        "eligible_frozen_input_rows": int(state_df.get("simulation_eligible", pd.Series(dtype=bool)).fillna(False).sum()) if not state_df.empty else 0,
        "ineligible_players": int(len(ineligible_rows)),
        "max_input_game_date": state_df.get("max_input_game_date", pd.Series(dtype=str)).max() if not state_df.empty else "",
        "production_behavior_changed": False,
        "promotion_ready": False,
        "output_paths": {
            "state_csv": str(state_csv),
            "state_parquet": str(state_parquet),
            "ineligible_csv": str(ineligible_csv),
        },
    }
    (output_dir / "frozen_preseason_backfill_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill frozen pre-cutoff player-state rows from discovered local logs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--discovery-csv", type=Path)
    parser.add_argument("--actual-outcomes", type=Path)
    parser.add_argument("--cutoff-date", default="2025-10-01")
    parser.add_argument("--min-games", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = backfill_pre_cutoff_player_state_logs(
        output_dir=args.output_dir,
        discovery_csv=args.discovery_csv,
        actual_outcomes=args.actual_outcomes,
        cutoff_date=str(args.cutoff_date),
        min_games=int(args.min_games),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

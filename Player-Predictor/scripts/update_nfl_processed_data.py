#!/usr/bin/env python3
"""
Refresh NFL processed player files from nflverse enrichment tables.

Outputs are written to:
    Data-Proc-NFL/<Player>/<season>_processed_processed.csv

This mirrors the NBA update flow conceptually:
- source refresh + cache-aware season tables
- optional market line merge
- rolling/lag/context feature engineering
- per-player processed files plus a manifest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_nfl_enrichment import NFLEnrichmentScraper, safe_write_json, utc_now_iso  # noqa: E402


RAW_ROOT = REPO_ROOT / "data copy" / "raw" / "nfl_enrichment"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nfl"
PROC_ROOT = REPO_ROOT / "Data-Proc-NFL"
TARGETS = ["PASS_YDS", "RUSH_YDS", "REC_YDS"]
MARKET_CONTRACT_COLUMNS = [
    "Market_PASS_YDS",
    "Market_RUSH_YDS",
    "Market_REC_YDS",
    "Synthetic_Market_PASS_YDS",
    "Synthetic_Market_RUSH_YDS",
    "Synthetic_Market_REC_YDS",
    "Market_Source_PASS_YDS",
    "Market_Source_RUSH_YDS",
    "Market_Source_REC_YDS",
    "Market_PASS_YDS_books",
    "Market_RUSH_YDS_books",
    "Market_REC_YDS_books",
    "Market_PASS_YDS_over_price",
    "Market_RUSH_YDS_over_price",
    "Market_REC_YDS_over_price",
    "Market_PASS_YDS_under_price",
    "Market_RUSH_YDS_under_price",
    "Market_REC_YDS_under_price",
    "Market_PASS_YDS_line_std",
    "Market_RUSH_YDS_line_std",
    "Market_REC_YDS_line_std",
    "Market_Fetched_At_UTC",
]


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Data-Proc-NFL season files from nflverse weekly stats.")
    parser.add_argument("--season", type=int, required=True, help="Season year. Example: 2024.")
    parser.add_argument("--season-type", type=str, default="REG", choices=["REG", "POST", "ALL"], help="Season type filter.")
    parser.add_argument("--through-week", type=int, default=None, help="Optional inclusive week cutoff.")
    parser.add_argument("--through-date", type=str, default=None, help="Optional inclusive date cutoff YYYY-MM-DD.")
    parser.add_argument("--refresh-source", action="store_true", help="Overwrite cached season raw tables before processing.")
    parser.add_argument("--sleep-seconds", type=float, default=0.25, help="Remote fetch delay when refreshing source.")
    parser.add_argument("--retries", type=int, default=3, help="Remote fetch retry count.")
    parser.add_argument("--player-limit", type=int, default=None, help="Optional limit on number of players written.")
    parser.add_argument("--merge-market-props", action="store_true", help="Merge latest normalized NFL market props snapshot.")
    parser.add_argument("--market-wide-path", type=Path, default=None, help="Optional explicit NFL market wide path.")
    parser.add_argument("--include-rosters", action="store_true", help="Fetch roster table into enrichment output.")
    parser.add_argument("--include-depth-charts", action="store_true", help="Fetch depth chart table into enrichment output.")
    return parser.parse_args()


def _to_num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _resolve_player_name(df: pd.DataFrame) -> pd.Series:
    for col in ["player_display_name", "player_name", "player_short_name"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_season_tables(season: int) -> dict[str, pd.DataFrame]:
    season_dir = RAW_ROOT / f"season={season}"
    return {
        "player_weekly": _read_optional_parquet(season_dir / "player_weekly.parquet"),
        "games": _read_optional_parquet(season_dir / "games.parquet"),
        "ngs_passing": _read_optional_parquet(season_dir / "ngs_passing.parquet"),
        "ngs_rushing": _read_optional_parquet(season_dir / "ngs_rushing.parquet"),
        "ngs_receiving": _read_optional_parquet(season_dir / "ngs_receiving.parquet"),
        "rosters": _read_optional_parquet(season_dir / "rosters.parquet"),
        "depth_charts": _read_optional_parquet(season_dir / "depth_charts.parquet"),
    }


def _game_type_to_season_type(token: str) -> str:
    value = str(token or "").strip().upper()
    return "REG" if value == "REG" else "POST"


def build_games_long(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Season_Type",
                "Team",
                "Opponent",
                "Game_ID",
                "Date",
                "Is_Home",
                "Spread_Line",
                "Total_Line",
                "Team_Rest_Days",
                "Opp_Rest_Days",
            ]
        )

    games = games_df.copy()
    games["Season"] = pd.to_numeric(games.get("season"), errors="coerce").fillna(0).astype(int)
    games["Week"] = pd.to_numeric(games.get("week"), errors="coerce").fillna(0).astype(int)
    games["Season_Type"] = games.get("game_type", "REG").astype(str).map(_game_type_to_season_type)
    games["Date"] = pd.to_datetime(games.get("gameday"), errors="coerce")

    common = [
        "Season",
        "Week",
        "Season_Type",
        "game_id",
        "Date",
        "spread_line",
        "total_line",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
    ]
    common = [col for col in common if col in games.columns]
    base = games[common].copy()

    home = base.copy()
    home["Team"] = home.get("home_team", "")
    home["Opponent"] = home.get("away_team", "")
    home["Is_Home"] = 1
    home["Team_Rest_Days"] = pd.to_numeric(home.get("home_rest"), errors="coerce")
    home["Opp_Rest_Days"] = pd.to_numeric(home.get("away_rest"), errors="coerce")

    away = base.copy()
    away["Team"] = away.get("away_team", "")
    away["Opponent"] = away.get("home_team", "")
    away["Is_Home"] = 0
    away["Team_Rest_Days"] = pd.to_numeric(away.get("away_rest"), errors="coerce")
    away["Opp_Rest_Days"] = pd.to_numeric(away.get("home_rest"), errors="coerce")

    out = pd.concat([home, away], ignore_index=True)
    out = out.rename(
        columns={
            "game_id": "Game_ID",
            "spread_line": "Spread_Line",
            "total_line": "Total_Line",
        }
    )
    keep_cols = [
        "Season",
        "Week",
        "Season_Type",
        "Team",
        "Opponent",
        "Game_ID",
        "Date",
        "Is_Home",
        "Spread_Line",
        "Total_Line",
        "Team_Rest_Days",
        "Opp_Rest_Days",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep_cols]
    out["Team"] = out["Team"].fillna("").astype(str)
    out["Opponent"] = out["Opponent"].fillna("").astype(str)
    out = out.drop_duplicates(subset=["Season", "Week", "Season_Type", "Team", "Opponent"], keep="last").reset_index(drop=True)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month = out["Date"].dt.month.astype(float)
    dow = out["Date"].dt.dayofweek.astype(float)
    week = pd.to_numeric(out["Week"], errors="coerce").fillna(0.0)
    out["Month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    out["Month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    out["DayOfWeek_sin"] = np.sin(2.0 * math.pi * dow / 7.0)
    out["DayOfWeek_cos"] = np.cos(2.0 * math.pi * dow / 7.0)
    out["Week_sin"] = np.sin(2.0 * math.pi * week / 18.0)
    out["Week_cos"] = np.cos(2.0 * math.pi * week / 18.0)
    return out


def load_market_props_wide(explicit_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            MARKET_ROOT / "latest_player_props_wide.parquet",
            MARKET_ROOT / "latest_player_props_wide.csv",
        ]
    )
    selected = next((path for path in candidates if path and path.exists()), None)
    if selected is None:
        return pd.DataFrame(), {"available": False, "path": None, "rows": 0, "matched_rows": 0}

    if selected.suffix.lower() == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)

    if df.empty:
        return df, {"available": True, "path": str(selected), "rows": 0, "matched_rows": 0}

    rename_map = {}
    if "player_name_norm" in df.columns and "Player" not in df.columns:
        rename_map["player_name_norm"] = "Player"
    if "event_date_et" in df.columns and "Market_Date" not in df.columns:
        rename_map["event_date_et"] = "Market_Date"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Player" not in df.columns or "Market_Date" not in df.columns:
        raise ValueError(f"Market props snapshot missing required columns in {selected}")

    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce").dt.date.astype(str)

    numeric_cols = [
        "Market_PASS_YDS",
        "Market_RUSH_YDS",
        "Market_REC_YDS",
        "Market_PASS_YDS_books",
        "Market_RUSH_YDS_books",
        "Market_REC_YDS_books",
        "Market_PASS_YDS_over_price",
        "Market_RUSH_YDS_over_price",
        "Market_REC_YDS_over_price",
        "Market_PASS_YDS_under_price",
        "Market_RUSH_YDS_under_price",
        "Market_REC_YDS_under_price",
        "Market_PASS_YDS_line_std",
        "Market_RUSH_YDS_line_std",
        "Market_REC_YDS_line_std",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["Player", "Market_Date"] + [col for col in MARKET_CONTRACT_COLUMNS if col in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["Player", "Market_Date"], keep="last").copy()
    return df, {"available": True, "path": str(selected), "rows": int(len(df)), "matched_rows": 0}


def apply_market_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    summary = {"rows_filled": 0, "targets": {}}
    for target in TARGETS:
        market_col = f"Market_{target}"
        synth_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        baseline_col = f"{target}_rolling_avg"

        if market_col not in out.columns:
            out[market_col] = np.nan
        if synth_col not in out.columns:
            out[synth_col] = np.nan
        if source_col not in out.columns:
            out[source_col] = pd.Series([""] * len(out), index=out.index, dtype="object")
        else:
            out[source_col] = out[source_col].fillna("").astype("object")

        baseline = pd.to_numeric(out.get(baseline_col), errors="coerce").fillna(0.0)
        real_mask = out[market_col].notna()
        fill_mask = ~real_mask

        out.loc[fill_mask, synth_col] = baseline[fill_mask]
        out.loc[fill_mask, market_col] = baseline[fill_mask]
        out.loc[real_mask, source_col] = "real"
        out.loc[fill_mask, source_col] = "synthetic_baseline"

        rows_filled = int(fill_mask.sum())
        summary["targets"][target] = {"rows_filled": rows_filled}
        summary["rows_filled"] += rows_filled
    return out, summary


def _prepare_nextgen_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["Season"] = pd.to_numeric(out.get("season"), errors="coerce").fillna(0).astype(int)
    out["Week"] = pd.to_numeric(out.get("week"), errors="coerce").fillna(0).astype(int)
    out["Season_Type"] = out.get("season_type", "REG").astype(str).str.upper()
    out["Player_ID"] = out.get("player_gsis_id", "").astype(str)

    keep_base = ["Season", "Week", "Season_Type", "Player_ID"]
    rename_map = {}
    if kind == "passing":
        for col in [
            "avg_time_to_throw",
            "avg_intended_air_yards",
            "aggressiveness",
            "completion_percentage_above_expectation",
            "passer_rating",
        ]:
            if col in out.columns:
                rename_map[col] = f"NGS_PASS_{col}"
    elif kind == "rushing":
        for col in [
            "efficiency",
            "avg_time_to_los",
            "rush_yards_over_expected",
            "rush_yards_over_expected_per_att",
            "rush_pct_over_expected",
        ]:
            if col in out.columns:
                rename_map[col] = f"NGS_RUSH_{col}"
    elif kind == "receiving":
        for col in [
            "avg_cushion",
            "avg_separation",
            "percent_share_of_intended_air_yards",
            "avg_yac",
            "avg_yac_above_expectation",
        ]:
            if col in out.columns:
                rename_map[col] = f"NGS_REC_{col}"

    keep_cols = keep_base + list(rename_map.keys())
    out = out[[col for col in keep_cols if col in out.columns]].copy()
    out = out.rename(columns=rename_map)
    out = out.drop_duplicates(subset=keep_base, keep="last").reset_index(drop=True)
    return out


def _build_fallback_date(df: pd.DataFrame) -> pd.Series:
    dates = []
    for season, week in zip(df["Season"], df["Week"]):
        try:
            start_year = int(season)
            base = pd.Timestamp(f"{start_year}-09-01")
            dates.append(base + pd.to_timedelta(max(int(week) - 1, 0) * 7, unit="D"))
        except Exception:
            dates.append(pd.NaT)
    return pd.Series(dates, index=df.index)


def build_processed_season(
    weekly_df: pd.DataFrame,
    games_df: pd.DataFrame,
    ngs_tables: dict[str, pd.DataFrame],
    *,
    through_date: str | None = None,
    market_props_wide: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if weekly_df.empty:
        return {}, {"players": 0, "rows": 0, "min_date": None, "max_date": None, "market_props_rows_matched": 0, "market_rows_filled": 0}

    df = weekly_df.copy()
    df["Player"] = _resolve_player_name(df).map(normalize_name)
    df["Player_ID"] = df.get("player_id", "").fillna("").astype(str)
    df["Team"] = df.get("recent_team", "").fillna("").astype(str)
    df["Opponent"] = df.get("opponent_team", "").fillna("").astype(str)
    df["Season"] = pd.to_numeric(df.get("season"), errors="coerce").fillna(0).astype(int)
    df["Week"] = pd.to_numeric(df.get("week"), errors="coerce").fillna(0).astype(int)
    df["Season_Type"] = df.get("season_type", "REG").astype(str).str.upper()

    df["PASS_YDS"] = _to_num(df, "passing_yards")
    df["RUSH_YDS"] = _to_num(df, "rushing_yards")
    df["REC_YDS"] = _to_num(df, "receiving_yards")
    df["PASS_TD"] = _to_num(df, "passing_tds")
    df["RUSH_TD"] = _to_num(df, "rushing_tds")
    df["REC_TD"] = _to_num(df, "receiving_tds")
    df["INTERCEPTIONS"] = _to_num(df, "interceptions")
    df["SACKS"] = _to_num(df, "sacks")
    df["COMPLETIONS"] = _to_num(df, "completions")
    df["ATTEMPTS"] = _to_num(df, "attempts")
    df["CARRIES"] = _to_num(df, "carries")
    df["RECEPTIONS"] = _to_num(df, "receptions")
    df["TARGETS"] = _to_num(df, "targets")
    df["FANTASY_POINTS"] = _to_num(df, "fantasy_points")
    df["FANTASY_POINTS_PPR"] = _to_num(df, "fantasy_points_ppr")
    df["PASS_EPA"] = _to_num(df, "passing_epa")
    df["RUSH_EPA"] = _to_num(df, "rushing_epa")
    df["REC_EPA"] = _to_num(df, "receiving_epa")
    df["TARGET_SHARE"] = _to_num(df, "target_share")
    df["AIR_YARDS_SHARE"] = _to_num(df, "air_yards_share")
    df["WOPR"] = _to_num(df, "wopr")

    df["CMP_PCT"] = np.where(df["ATTEMPTS"] > 0, df["COMPLETIONS"] / df["ATTEMPTS"], 0.0)
    df["YPA_PASS"] = np.where(df["ATTEMPTS"] > 0, df["PASS_YDS"] / df["ATTEMPTS"], 0.0)
    df["YPA_RUSH"] = np.where(df["CARRIES"] > 0, df["RUSH_YDS"] / df["CARRIES"], 0.0)
    df["YPR_REC"] = np.where(df["RECEPTIONS"] > 0, df["REC_YDS"] / df["RECEPTIONS"], 0.0)

    games_long = build_games_long(games_df)
    if not games_long.empty:
        df = df.merge(
            games_long,
            how="left",
            on=["Season", "Week", "Season_Type", "Team", "Opponent"],
        )
    else:
        df["Game_ID"] = ""
        df["Date"] = pd.NaT
        df["Is_Home"] = np.nan
        df["Spread_Line"] = np.nan
        df["Total_Line"] = np.nan
        df["Team_Rest_Days"] = np.nan
        df["Opp_Rest_Days"] = np.nan

    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    missing_date_mask = df["Date"].isna()
    if missing_date_mask.any():
        df.loc[missing_date_mask, "Date"] = _build_fallback_date(df.loc[missing_date_mask, ["Season", "Week"]])

    if through_date:
        cutoff = pd.Timestamp(through_date)
        df = df.loc[df["Date"] <= cutoff].copy()

    for kind in ("passing", "rushing", "receiving"):
        ngs = _prepare_nextgen_table(ngs_tables.get(f"ngs_{kind}", pd.DataFrame()), kind)
        if not ngs.empty:
            df = df.merge(
                ngs,
                how="left",
                on=["Season", "Week", "Season_Type", "Player_ID"],
            )

    df = df.sort_values(["Player", "Date", "Week"]).reset_index(drop=True)

    activity = df["ATTEMPTS"] + df["CARRIES"] + df["TARGETS"] + df["RECEPTIONS"] + df["PASS_YDS"] + df["RUSH_YDS"] + df["REC_YDS"]
    df["Did_Not_Play"] = (activity <= 0.0).astype(int)
    df["Rest_Days"] = df.groupby("Player")["Date"].diff().dt.days.fillna(7).clip(lower=0).astype(float)
    df["Game_Index"] = df.groupby("Player").cumcount().astype(int)
    df = add_time_features(df)

    opp_allowed = (
        df.groupby(["Opponent", "Date"], as_index=False)[["PASS_YDS", "RUSH_YDS", "REC_YDS"]]
        .sum()
        .rename(
            columns={
                "PASS_YDS": "Opp_PASS_YDS_Allowed",
                "RUSH_YDS": "Opp_RUSH_YDS_Allowed",
                "REC_YDS": "Opp_REC_YDS_Allowed",
            }
        )
        .sort_values(["Opponent", "Date"])
    )
    for col in ["Opp_PASS_YDS_Allowed", "Opp_RUSH_YDS_Allowed", "Opp_REC_YDS_Allowed"]:
        opp_allowed[f"{col}_3"] = (
            opp_allowed.groupby("Opponent")[col]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        )
    df = df.merge(
        opp_allowed[
            [
                "Opponent",
                "Date",
                "Opp_PASS_YDS_Allowed_3",
                "Opp_RUSH_YDS_Allowed_3",
                "Opp_REC_YDS_Allowed_3",
            ]
        ],
        how="left",
        on=["Opponent", "Date"],
    )
    for col in ["Opp_PASS_YDS_Allowed_3", "Opp_RUSH_YDS_Allowed_3", "Opp_REC_YDS_Allowed_3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rolling_cols = [
        "PASS_YDS",
        "RUSH_YDS",
        "REC_YDS",
        "PASS_TD",
        "RUSH_TD",
        "REC_TD",
        "INTERCEPTIONS",
        "SACKS",
        "COMPLETIONS",
        "ATTEMPTS",
        "CARRIES",
        "RECEPTIONS",
        "TARGETS",
        "FANTASY_POINTS_PPR",
        "PASS_EPA",
        "RUSH_EPA",
        "REC_EPA",
        "CMP_PCT",
        "YPA_PASS",
        "YPA_RUSH",
        "YPR_REC",
    ]
    for col in rolling_cols:
        rolling = df.groupby("Player")[col].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        rolling_std = df.groupby("Player")[col].transform(lambda s: s.shift(1).rolling(window=5, min_periods=2).std())
        df[f"{col}_rolling_avg"] = rolling.fillna(df[col])
        df[f"{col}_rolling_std"] = pd.to_numeric(rolling_std, errors="coerce").fillna(0.0)
        df[f"{col}_lag1"] = df.groupby("Player")[col].shift(1).fillna(df[col])

    market_match_count = 0
    if market_props_wide is not None and not market_props_wide.empty:
        merge_df = market_props_wide.copy()
        df["Market_Date"] = df["Date"].dt.date.astype(str)
        df = df.merge(merge_df, how="left", on=["Player", "Market_Date"])
        market_match_count = int(df["Market_PASS_YDS"].notna().sum() if "Market_PASS_YDS" in df.columns else 0)
        df = df.drop(columns=["Market_Date"])
    else:
        for col in MARKET_CONTRACT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

    for col in MARKET_CONTRACT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df, market_fill_summary = apply_market_fallback(df)

    for target in TARGETS:
        market_col = f"Market_{target}"
        baseline_col = f"{target}_rolling_avg"
        df[f"{target}_market_gap"] = pd.to_numeric(df[market_col], errors="coerce") - pd.to_numeric(df[baseline_col], errors="coerce")

    teams = sorted({team for team in df["Team"].dropna().astype(str) if team} | {team for team in df["Opponent"].dropna().astype(str) if team})
    team_id_map = {team: idx + 1 for idx, team in enumerate(teams)}
    df["Team_ID"] = df["Team"].map(team_id_map).fillna(0).astype(int)
    df["Opponent_ID"] = df["Opponent"].map(team_id_map).fillna(0).astype(int)

    ngs_cols = [col for col in df.columns if col.startswith("NGS_PASS_") or col.startswith("NGS_RUSH_") or col.startswith("NGS_REC_")]
    keep_cols = [
        "Date",
        "Player",
        "PASS_YDS",
        "RUSH_YDS",
        "REC_YDS",
        "PASS_TD",
        "RUSH_TD",
        "REC_TD",
        "INTERCEPTIONS",
        "SACKS",
        "COMPLETIONS",
        "ATTEMPTS",
        "CARRIES",
        "RECEPTIONS",
        "TARGETS",
        "FANTASY_POINTS",
        "FANTASY_POINTS_PPR",
        "TARGET_SHARE",
        "AIR_YARDS_SHARE",
        "WOPR",
        "CMP_PCT",
        "YPA_PASS",
        "YPA_RUSH",
        "YPR_REC",
        "PASS_EPA",
        "RUSH_EPA",
        "REC_EPA",
        "Did_Not_Play",
        "Rest_Days",
        "Game_Index",
        "Player_ID",
        "Team",
        "Team_ID",
        "Opponent",
        "Opponent_ID",
        "Game_ID",
        "Season",
        "Week",
        "Season_Type",
        "Is_Home",
        "Spread_Line",
        "Total_Line",
        "Team_Rest_Days",
        "Opp_Rest_Days",
        "Month_sin",
        "Month_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Week_sin",
        "Week_cos",
        "Opp_PASS_YDS_Allowed_3",
        "Opp_RUSH_YDS_Allowed_3",
        "Opp_REC_YDS_Allowed_3",
    ] + ngs_cols + MARKET_CONTRACT_COLUMNS + [f"{target}_market_gap" for target in TARGETS]
    keep_cols += [f"{col}_rolling_avg" for col in rolling_cols]
    keep_cols += [f"{col}_rolling_std" for col in rolling_cols]
    keep_cols += [f"{col}_lag1" for col in rolling_cols]
    keep_cols = [col for col in keep_cols if col in df.columns]

    df = df[keep_cols].copy()
    df = df.sort_values(["Player", "Date", "Week"]).reset_index(drop=True)

    player_frames = {}
    for player_name, player_df in df.groupby("Player", sort=True):
        player_frames[player_name] = player_df.reset_index(drop=True)

    summary = {
        "players": int(len(player_frames)),
        "rows": int(len(df)),
        "min_date": str(df["Date"].min().date()) if len(df) and pd.notna(df["Date"].min()) else None,
        "max_date": str(df["Date"].max().date()) if len(df) and pd.notna(df["Date"].max()) else None,
        "market_props_rows_matched": market_match_count,
        "market_rows_filled": int(market_fill_summary.get("rows_filled", 0)),
    }
    return player_frames, summary


def write_processed_files(player_frames: dict[str, pd.DataFrame], season: int, player_limit: int | None) -> dict:
    written = {}
    items = list(sorted(player_frames.items()))
    if player_limit is not None:
        items = items[:player_limit]

    for player_name, player_df in items:
        player_dir = PROC_ROOT / player_name
        player_dir.mkdir(parents=True, exist_ok=True)
        out_path = player_dir / f"{season}_processed_processed.csv"
        player_df.to_csv(out_path, index=False)
        written[player_name] = {
            "rows": int(len(player_df)),
            "path": str(out_path),
            "max_date": str(pd.to_datetime(player_df["Date"]).max().date()) if len(player_df) else None,
        }
    return written


def main() -> None:
    args = parse_args()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    season_manifest_path = RAW_ROOT / f"season={args.season}" / "manifest.json"
    refresh_source = bool(args.refresh_source)
    if season_manifest_path.exists() and not refresh_source:
        try:
            previous_manifest = json.loads(season_manifest_path.read_text(encoding="utf-8"))
            previous_type = str(previous_manifest.get("season_type", "REG")).upper()
            previous_max_weeks = previous_manifest.get("max_weeks")
            if previous_type != args.season_type:
                refresh_source = True
            elif args.through_week is None and previous_max_weeks is not None:
                refresh_source = True
            elif args.through_week is not None:
                if previous_max_weeks is None or int(previous_max_weeks) < int(args.through_week):
                    refresh_source = True
        except Exception:
            refresh_source = True

    scraper = NFLEnrichmentScraper(
        outdir=RAW_ROOT,
        season_type=args.season_type,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        overwrite=refresh_source,
        include_nextgen=True,
        include_rosters=args.include_rosters,
        include_depth_charts=args.include_depth_charts,
    )
    scraper.scrape_season(args.season, max_weeks=args.through_week)

    tables = _load_season_tables(args.season)
    weekly_df = tables["player_weekly"]
    if weekly_df.empty:
        raise RuntimeError(f"No player_weekly data available for season {args.season}. Run fetch_nfl_enrichment first.")
    if args.through_week is not None and "week" in weekly_df.columns:
        weekly_df = weekly_df.loc[pd.to_numeric(weekly_df["week"], errors="coerce") <= int(args.through_week)].copy()

    games_df = tables["games"]
    if args.through_week is not None and not games_df.empty and "week" in games_df.columns:
        games_df = games_df.loc[pd.to_numeric(games_df["week"], errors="coerce") <= int(args.through_week)].copy()

    market_props_wide = pd.DataFrame()
    market_merge_summary = {"available": False, "path": None, "rows": 0, "matched_rows": 0}
    if args.merge_market_props:
        market_props_wide, market_merge_summary = load_market_props_wide(args.market_wide_path)
        if market_merge_summary["available"]:
            print(
                "Market props snapshot loaded: "
                f"{market_merge_summary['rows']} rows from {market_merge_summary['path']}"
            )
        else:
            print("Market props snapshot not found; continuing without market merge.")

    player_frames, summary = build_processed_season(
        weekly_df,
        games_df,
        {
            "ngs_passing": tables["ngs_passing"],
            "ngs_rushing": tables["ngs_rushing"],
            "ngs_receiving": tables["ngs_receiving"],
        },
        through_date=args.through_date,
        market_props_wide=market_props_wide,
    )
    written = write_processed_files(player_frames, args.season, args.player_limit)
    market_merge_summary["matched_rows"] = int(summary.get("market_props_rows_matched", 0))

    manifest = {
        "season": args.season,
        "season_type": args.season_type,
        "through_week_requested": args.through_week,
        "through_date_requested": args.through_date,
        "source_refresh": bool(args.refresh_source),
        "weekly_rows": int(len(weekly_df)),
        "games_rows": int(len(games_df)),
        "ngs_passing_rows": int(len(tables["ngs_passing"])),
        "ngs_rushing_rows": int(len(tables["ngs_rushing"])),
        "ngs_receiving_rows": int(len(tables["ngs_receiving"])),
        "processed_summary": summary,
        "market_props_merge": market_merge_summary,
        "players_written": len(written),
        "written": written,
        "updated_at_utc": utc_now_iso(),
    }
    safe_write_json(PROC_ROOT / f"update_manifest_{args.season}.json", manifest)

    print("\n" + "=" * 80)
    print("NFL DATA UPDATE COMPLETE")
    print("=" * 80)
    print(f"Season: {args.season}")
    print(f"Players written: {len(written)}")
    print(f"Processed max date: {summary['max_date']}")
    if args.merge_market_props:
        print(
            "Market props merge: "
            f"{market_merge_summary['matched_rows']} matched rows from {market_merge_summary['rows']} market rows"
        )
    print(f"Manifest: {PROC_ROOT / f'update_manifest_{args.season}.json'}")


if __name__ == "__main__":
    main()

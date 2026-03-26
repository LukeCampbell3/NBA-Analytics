#!/usr/bin/env python3
"""
Refresh official NBA season data and rebuild Data-Proc season files.

This script is intended to be rerun after new NBA games complete.
It:
- refreshes official player game logs for a season
- optionally backfills missing advanced boxscore cache
- rebuilds Data-Proc/<Player>/<season>_processed_processed.csv

By default it is incremental and cache-aware.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_nba_enrichment import NBAEnrichmentScraper, safe_write_json, utc_now_iso  # noqa: E402


RAW_ROOT = REPO_ROOT / "data copy" / "raw" / "nba_enrichment"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
PROC_ROOT = REPO_ROOT / "Data-Proc"
MARKET_ANCHOR_DIR = REPO_ROOT / "model" / "market_anchor"
LEGACY_CONTRACT_DEFAULTS = {
    "FTA": 0.0,
    "PLUS_MINUS": 0.0,
}
MARKET_CONTRACT_COLUMNS = [
    "Market_PTS",
    "Market_TRB",
    "Market_AST",
    "Synthetic_Market_PTS",
    "Synthetic_Market_TRB",
    "Synthetic_Market_AST",
    "Market_Source_PTS",
    "Market_Source_TRB",
    "Market_Source_AST",
    "Market_PTS_books",
    "Market_TRB_books",
    "Market_AST_books",
    "Market_PTS_over_price",
    "Market_TRB_over_price",
    "Market_AST_over_price",
    "Market_PTS_under_price",
    "Market_TRB_under_price",
    "Market_AST_under_price",
    "Market_PTS_line_std",
    "Market_TRB_line_std",
    "Market_AST_line_std",
    "Market_Fetched_At_UTC",
]


def normalize_name(value: str) -> str:
    out = value
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


def resolve_player_dir(player_name: str) -> Path:
    normalized = normalize_name(player_name)
    existing = {normalize_name(path.name): path for path in PROC_ROOT.iterdir() if path.is_dir()} if PROC_ROOT.exists() else {}
    if normalized in existing:
        return existing[normalized]
    return PROC_ROOT / normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Data-Proc season files from official NBA logs.")
    parser.add_argument("--season", type=int, required=True, help="Season end year. Example: 2026 for 2025-26.")
    parser.add_argument("--through-date", type=str, default=None, help="Optional inclusive cutoff date YYYY-MM-DD.")
    parser.add_argument("--refresh-advanced", action="store_true", help="Fetch missing advanced boxscore cache for all season games up to the cutoff.")
    parser.add_argument("--sleep-seconds", type=float, default=0.35, help="API cooldown between calls when refreshing advanced data.")
    parser.add_argument("--timeout", type=int, default=30, help="nba_api timeout.")
    parser.add_argument("--max-games", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--player-limit", type=int, default=None, help="Optional limit on number of players written.")
    parser.add_argument(
        "--skip-contract-repair",
        action="store_true",
        help="Skip legacy processed-file contract repair after writing the refreshed season.",
    )
    parser.add_argument(
        "--merge-market-props",
        action="store_true",
        help="Merge latest normalized market props snapshot into processed files when available.",
    )
    parser.add_argument(
        "--market-wide-path",
        type=Path,
        default=None,
        help="Optional explicit path to a normalized wide market props parquet/csv snapshot.",
    )
    parser.add_argument(
        "--apply-market-anchor",
        action="store_true",
        help="Backfill missing Market_* values from a trained synthetic market-anchor model when available.",
    )
    parser.add_argument(
        "--market-anchor-path",
        type=Path,
        default=None,
        help="Optional explicit path to a trained market anchor bundle (.pkl).",
    )
    return parser.parse_args()


def _reorder_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = [
        "Date", "Player", "PTS", "TRB", "AST", "STL", "TOV", "FG%", "3P%", "FT%", "TS%", "USG%",
        "BPM", "ORTG", "DRTG", "GmSc", "Did_Not_Play", "Rest_Days", "Game_Index", "Player_ID",
        "Team_ID", "Opponent", "Opponent_ID", "MATCHUP", "MP", "FGA", "FTA", "PLUS_MINUS",
        "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos", "oppDfRtg_3",
        "Market_PTS", "Market_TRB", "Market_AST",
        "Synthetic_Market_PTS", "Synthetic_Market_TRB", "Synthetic_Market_AST",
        "Market_Source_PTS", "Market_Source_TRB", "Market_Source_AST",
        "Market_PTS_books", "Market_TRB_books", "Market_AST_books",
        "Market_PTS_over_price", "Market_TRB_over_price", "Market_AST_over_price",
        "Market_PTS_under_price", "Market_TRB_under_price", "Market_AST_under_price",
        "Market_PTS_line_std", "Market_TRB_line_std", "Market_AST_line_std",
        "PTS_market_gap", "TRB_market_gap", "AST_market_gap",
    ]
    ordered = [col for col in preferred_order if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]


def repair_processed_file_contract(csv_path: Path) -> dict | None:
    header = pd.read_csv(csv_path, nrows=0)
    missing = [col for col in LEGACY_CONTRACT_DEFAULTS if col not in header.columns]
    if not missing:
        return None

    df = pd.read_csv(csv_path)
    for col, default in LEGACY_CONTRACT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)

    df = _reorder_contract_columns(df)
    df.to_csv(csv_path, index=False)
    return {
        "path": str(csv_path),
        "missing_columns_added": missing,
        "rows": int(len(df)),
    }


def repair_legacy_contracts_for_players(player_names: list[str]) -> dict:
    repaired = []
    scanned = 0
    seen_paths: set[Path] = set()
    for player_name in player_names:
        player_dir = resolve_player_dir(player_name)
        if not player_dir.exists():
            continue
        for csv_path in sorted(player_dir.glob("*_processed_processed.csv")):
            if csv_path in seen_paths:
                continue
            seen_paths.add(csv_path)
            scanned += 1
            payload = repair_processed_file_contract(csv_path)
            if payload is not None:
                repaired.append(payload)
    return {
        "files_scanned": scanned,
        "files_repaired": len(repaired),
        "repaired": repaired,
    }


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
        "Market_PTS", "Market_TRB", "Market_AST",
        "Market_PTS_books", "Market_TRB_books", "Market_AST_books",
        "Market_PTS_over_price", "Market_TRB_over_price", "Market_AST_over_price",
        "Market_PTS_under_price", "Market_TRB_under_price", "Market_AST_under_price",
        "Market_PTS_line_std", "Market_TRB_line_std", "Market_AST_line_std",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["Player", "Market_Date"] + [col for col in MARKET_CONTRACT_COLUMNS if col in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["Player", "Market_Date"], keep="last").copy()
    return df, {"available": True, "path": str(selected), "rows": int(len(df)), "matched_rows": 0}


def load_market_anchor_bundle(explicit_path: Path | None = None):
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(MARKET_ANCHOR_DIR / "latest_market_anchor.pkl")
    selected = next((path for path in candidates if path and path.exists()), None)
    if selected is None:
        return None, {"available": False, "path": None, "targets": []}
    bundle = joblib.load(selected)
    return bundle, {
        "available": True,
        "path": str(selected),
        "targets": sorted(bundle.get("targets", {}).keys()),
    }


def apply_market_anchor(df: pd.DataFrame, market_anchor_bundle) -> tuple[pd.DataFrame, dict]:
    if market_anchor_bundle is None:
        return df, {"rows_filled": 0, "targets": {}}
    out = df.copy()
    feature_cols = market_anchor_bundle.get("feature_columns", [])
    categorical_cols = set(market_anchor_bundle.get("categorical_features", []))
    for col in feature_cols:
        if col not in out.columns:
            out[col] = "UNK" if col in categorical_cols else 0.0
    for col in feature_cols:
        if col in categorical_cols:
            out[col] = out[col].fillna("UNK").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    summary = {"rows_filled": 0, "targets": {}}
    X = out[feature_cols].copy()
    for target, payload in market_anchor_bundle.get("targets", {}).items():
        model = payload["model"]
        real_col = f"Market_{target}"
        synth_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        preds = model.predict(X)
        out[synth_col] = preds
        real_mask = out[real_col].notna() if real_col in out.columns else pd.Series([False] * len(out), index=out.index)
        out[real_col] = out[real_col] if real_col in out.columns else np.nan
        fill_mask = ~real_mask
        out.loc[fill_mask, real_col] = out.loc[fill_mask, synth_col]
        out[source_col] = np.where(real_mask, "real", "synthetic")
        rows_filled = int(fill_mask.sum())
        summary["targets"][target] = {"rows_filled": rows_filled}
        summary["rows_filled"] += rows_filled
    return out, summary


def game_score(row: pd.Series) -> float:
    return float(
        row.get("PTS", 0.0)
        + 0.4 * row.get("FGM", 0.0)
        - 0.7 * row.get("FGA", 0.0)
        - 0.4 * (row.get("FTA", 0.0) - row.get("FTM", 0.0))
        + 0.7 * row.get("OREB", 0.0)
        + 0.3 * row.get("DREB", 0.0)
        + row.get("STL", 0.0)
        + 0.7 * row.get("AST", 0.0)
        + 0.7 * row.get("BLK", 0.0)
        - 0.4 * row.get("PF", 0.0)
        - row.get("TOV", 0.0)
    )


def true_shooting_from_logs(df: pd.DataFrame) -> pd.Series:
    denom = 2.0 * (df["FGA"].astype(float) + 0.44 * df["FTA"].astype(float))
    return np.where(denom > 0, df["PTS"].astype(float) / denom, 0.0)


def usage_proxy(df: pd.DataFrame) -> pd.Series:
    possessions_used = df["FGA"].astype(float) + 0.44 * df["FTA"].astype(float) + df["TOV"].astype(float)
    minutes = pd.to_numeric(df["MP"], errors="coerce").fillna(0.0).astype(float)
    return np.where(minutes > 0, np.clip((possessions_used / minutes) * 20.0, 0.0, 60.0), 0.0)


def bpm_proxy(df: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(df["MP"], errors="coerce").fillna(0.0).astype(float)
    plus_minus = pd.to_numeric(df["PLUS_MINUS"], errors="coerce").fillna(0.0).astype(float)
    return np.where(minutes > 0, (plus_minus / minutes) * 36.0, 0.0)


def parse_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        try:
            mins, secs = text.split(":", 1)
            return float(mins) + float(secs) / 60.0
        except Exception:
            return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def fetch_player_logs(scraper: NBAEnrichmentScraper, season: int, through_date: str | None, max_games: int | None) -> pd.DataFrame:
    season_dir = RAW_ROOT / f"season={season}"
    season_dir.mkdir(parents=True, exist_ok=True)
    logs = scraper.fetch_player_game_logs(season)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    if through_date:
        cutoff = pd.Timestamp(through_date)
        logs = logs.loc[logs["GAME_DATE"] <= cutoff].copy()
    if max_games is not None:
        keep_game_ids = sorted(logs["GAME_ID"].astype(str).unique().tolist())[:max_games]
        logs = logs.loc[logs["GAME_ID"].astype(str).isin(keep_game_ids)].copy()
    logs.to_parquet(season_dir / "player_game_logs.parquet", index=False)
    return logs


def refresh_advanced_cache(scraper: NBAEnrichmentScraper, season: int, game_ids: list[str]) -> None:
    spec = next(spec for spec in scraper.endpoint_specs if spec.name == "boxscore_advanced")
    print(f"\nRefreshing advanced cache for {len(game_ids)} games...")
    for idx, game_id in enumerate(game_ids, start=1):
        cached_player = scraper._load_cached_frame(season, "boxscore_advanced_player", game_id)
        cached_team = scraper._load_cached_frame(season, "boxscore_advanced_team", game_id)
        if cached_player is not None and cached_team is not None:
            continue
        print(f"  [{idx}/{len(game_ids)}] game {game_id}")
        frames = scraper._fetch_boxscore_endpoint(spec, game_id)
        for split_name, frame in zip(spec.split_names, frames):
            scraper._store_cached_frame(season, f"{spec.name}_{split_name}", game_id, frame)


def load_advanced_tables(scraper: NBAEnrichmentScraper, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = scraper._collect_cached_tables(season, ["boxscore_advanced_player", "boxscore_advanced_team"])
    player_adv = tables.get("boxscore_advanced_player", pd.DataFrame()).copy()
    team_adv = tables.get("boxscore_advanced_team", pd.DataFrame()).copy()
    season_dir = RAW_ROOT / f"season={season}"
    if not player_adv.empty:
        player_adv.to_parquet(season_dir / "boxscore_advanced_player.parquet", index=False)
    if not team_adv.empty:
        team_adv.to_parquet(season_dir / "boxscore_advanced_team.parquet", index=False)
    return player_adv, team_adv


def derive_opponent_abbrev(matchup: str) -> str:
    text = str(matchup).strip()
    if not text:
        return ""
    parts = text.split()
    return parts[-1] if parts else ""


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    month = df["Date"].dt.month.astype(float)
    dow = df["Date"].dt.dayofweek.astype(float)
    df["Month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    df["Month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    df["DayOfWeek_sin"] = np.sin(2.0 * math.pi * dow / 7.0)
    df["DayOfWeek_cos"] = np.cos(2.0 * math.pi * dow / 7.0)
    return df


def build_processed_season(
    logs: pd.DataFrame,
    player_adv: pd.DataFrame,
    team_adv: pd.DataFrame,
    market_props_wide: pd.DataFrame | None = None,
    market_anchor_bundle=None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    df = logs.copy()
    df["Date"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["PLAYER_NAME", "Date", "GAME_ID"]).reset_index(drop=True)
    df["Player"] = df["PLAYER_NAME"].astype(str).map(normalize_name)
    df["Player_ID"] = pd.to_numeric(df.get("PLAYER_ID"), errors="coerce").fillna(0).astype(int)
    df["Team_ID"] = pd.to_numeric(df.get("TEAM_ID"), errors="coerce").fillna(0).astype(int)
    df["MP"] = df["MIN"].apply(parse_minutes).astype(float)
    df["TRB"] = df["REB"].astype(float)
    df["FG%"] = pd.to_numeric(df["FG_PCT"], errors="coerce").fillna(0.0).astype(float)
    df["3P%"] = pd.to_numeric(df["FG3_PCT"], errors="coerce").fillna(0.0).astype(float)
    df["FT%"] = pd.to_numeric(df["FT_PCT"], errors="coerce").fillna(0.0).astype(float)
    df["Opponent"] = df["MATCHUP"].map(derive_opponent_abbrev)

    team_map = (
        df[["TEAM_ABBREVIATION", "TEAM_ID"]]
        .dropna()
        .drop_duplicates()
        .assign(TEAM_ID=lambda x: x["TEAM_ID"].astype(int))
    )
    team_id_by_abbrev = dict(zip(team_map["TEAM_ABBREVIATION"], team_map["TEAM_ID"]))
    df["Opponent_ID"] = df["Opponent"].map(team_id_by_abbrev).fillna(0).astype(int)

    if not player_adv.empty:
        padv = player_adv.rename(columns={"gameId": "GAME_ID", "personId": "PLAYER_ID", "teamId": "TEAM_ID"})
        keep_cols = [
            "GAME_ID", "PLAYER_ID", "TEAM_ID",
            "trueShootingPercentage", "usagePercentage", "offensiveRating", "defensiveRating",
            "estimatedNetRating", "PIE", "pace",
        ]
        padv = padv[[col for col in keep_cols if col in padv.columns]].copy()
        padv["GAME_ID"] = padv["GAME_ID"].astype(str)
        padv["PLAYER_ID"] = pd.to_numeric(padv["PLAYER_ID"], errors="coerce").fillna(0).astype(int)
        padv["TEAM_ID"] = pd.to_numeric(padv["TEAM_ID"], errors="coerce").fillna(0).astype(int)
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["Player_ID"] = pd.to_numeric(df["Player_ID"], errors="coerce").fillna(0).astype(int)
        df["Team_ID"] = pd.to_numeric(df["Team_ID"], errors="coerce").fillna(0).astype(int)
        df = df.merge(
            padv,
            how="left",
            left_on=["GAME_ID", "Player_ID", "Team_ID"],
            right_on=["GAME_ID", "PLAYER_ID", "TEAM_ID"],
        )
        for redundant in ["PLAYER_ID", "TEAM_ID"]:
            if redundant in df.columns:
                df = df.drop(columns=redundant)

    if not team_adv.empty:
        tadv = team_adv.rename(columns={"gameId": "GAME_ID", "teamId": "TEAM_ID"})
        keep_cols = ["GAME_ID", "TEAM_ID", "defensiveRating"]
        tadv = tadv[[col for col in keep_cols if col in tadv.columns]].copy()
        tadv["GAME_ID"] = tadv["GAME_ID"].astype(str)
        tadv["TEAM_ID"] = pd.to_numeric(tadv["TEAM_ID"], errors="coerce").fillna(0).astype(int)
        tadv = tadv.rename(columns={"TEAM_ID": "Opponent_ID", "defensiveRating": "opp_def_rating_game"})
        df = df.merge(tadv, how="left", on=["GAME_ID", "Opponent_ID"])

    df["TS%"] = pd.to_numeric(df.get("trueShootingPercentage"), errors="coerce")
    ts_fallback = true_shooting_from_logs(df)
    df["TS%"] = df["TS%"].fillna(pd.Series(ts_fallback, index=df.index)).astype(float)

    df["USG%"] = pd.to_numeric(df.get("usagePercentage"), errors="coerce")
    usg_fallback = usage_proxy(df)
    df["USG%"] = df["USG%"].fillna(pd.Series(usg_fallback, index=df.index)).astype(float)

    ortg_fallback = np.where(
        (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) > 0,
        100.0 * df["PTS"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]),
        100.0,
    )
    df["ORTG"] = pd.to_numeric(df.get("offensiveRating"), errors="coerce").fillna(pd.Series(ortg_fallback, index=df.index))
    drtg_fallback = pd.Series(110.0, index=df.index)
    df["DRTG"] = pd.to_numeric(df.get("defensiveRating"), errors="coerce").fillna(drtg_fallback)
    df["BPM"] = bpm_proxy(df)
    df["GmSc"] = df.apply(game_score, axis=1)
    df["Did_Not_Play"] = (
        (pd.to_numeric(df.get("AVAILABLE_FLAG"), errors="coerce").fillna(1.0) <= 0.0)
        | (df["MP"] <= 0.0)
    ).astype(int)

    df["Rest_Days"] = (
        df.groupby("Player")["Date"].diff().dt.days.fillna(2).clip(lower=0).astype(float)
    )
    df["Game_Index"] = df.groupby("Player").cumcount().astype(int)
    df = add_time_features(df)

    market_match_count = 0
    if market_props_wide is not None and not market_props_wide.empty:
        merge_df = market_props_wide.copy()
        df["Market_Date"] = df["Date"].dt.date.astype(str)
        df = df.merge(merge_df, how="left", on=["Player", "Market_Date"])
        market_match_count = int(df["Market_PTS"].notna().sum() if "Market_PTS" in df.columns else 0)
        df = df.drop(columns=["Market_Date"])
    else:
        for col in MARKET_CONTRACT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

    if "opp_def_rating_game" in df.columns:
        team_game = (
            df[["Opponent_ID", "Date", "opp_def_rating_game"]]
            .dropna()
            .sort_values(["Opponent_ID", "Date"])
            .drop_duplicates(subset=["Opponent_ID", "Date"], keep="last")
        )
        team_game["oppDfRtg_3"] = (
            team_game.groupby("Opponent_ID")["opp_def_rating_game"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            / 100.0
        )
        df = df.merge(team_game[["Opponent_ID", "Date", "oppDfRtg_3"]], how="left", on=["Opponent_ID", "Date"])
    else:
        df["oppDfRtg_3"] = pd.to_numeric(df["DRTG"], errors="coerce").fillna(110.0) / 100.0

    rolling_cols = ["PTS", "TRB", "AST", "STL", "TOV", "FG%", "3P%", "FT%", "TS%", "USG%", "BPM", "ORTG", "DRTG", "GmSc"]
    for col in rolling_cols:
        rolling = df.groupby("Player")[col].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
        df[f"{col}_rolling_avg"] = rolling.fillna(df[col])
        df[f"{col}_lag1"] = df.groupby("Player")[col].shift(1).fillna(df[col])

    for target in ["PTS", "TRB", "AST"]:
        market_col = f"Market_{target}"
        baseline_col = f"{target}_rolling_avg"
        if market_col in df.columns and baseline_col in df.columns:
            df[f"{target}_market_gap"] = pd.to_numeric(df[market_col], errors="coerce") - pd.to_numeric(df[baseline_col], errors="coerce")
        else:
            df[f"{target}_market_gap"] = np.nan

    market_anchor_summary = {"rows_filled": 0, "targets": {}}
    if market_anchor_bundle is not None:
        df, market_anchor_summary = apply_market_anchor(df, market_anchor_bundle)
        for target in ["PTS", "TRB", "AST"]:
            market_col = f"Market_{target}"
            baseline_col = f"{target}_rolling_avg"
            if market_col in df.columns and baseline_col in df.columns:
                df[f"{target}_market_gap"] = pd.to_numeric(df[market_col], errors="coerce") - pd.to_numeric(df[baseline_col], errors="coerce")

    keep_cols = [
        "Date", "Player", "PTS", "TRB", "AST", "STL", "TOV", "FG%", "3P%", "FT%", "TS%", "USG%",
        "BPM", "ORTG", "DRTG", "GmSc", "Did_Not_Play", "Rest_Days", "Game_Index", "Player_ID",
        "Team_ID", "Opponent", "Opponent_ID", "MATCHUP", "MP", "FGA", "FTA", "PLUS_MINUS",
        "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos", "oppDfRtg_3",
        "Market_PTS", "Market_TRB", "Market_AST",
        "Synthetic_Market_PTS", "Synthetic_Market_TRB", "Synthetic_Market_AST",
        "Market_Source_PTS", "Market_Source_TRB", "Market_Source_AST",
        "Market_PTS_books", "Market_TRB_books", "Market_AST_books",
        "Market_PTS_over_price", "Market_TRB_over_price", "Market_AST_over_price",
        "Market_PTS_under_price", "Market_TRB_under_price", "Market_AST_under_price",
        "Market_PTS_line_std", "Market_TRB_line_std", "Market_AST_line_std",
        "PTS_market_gap", "TRB_market_gap", "AST_market_gap",
    ] + [f"{col}_rolling_avg" for col in rolling_cols] + [f"{col}_lag1" for col in rolling_cols]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    player_frames = {}
    for player_name, player_df in df.groupby("Player", sort=True):
        player_frames[player_name] = player_df.sort_values("Date").reset_index(drop=True)

    summary = {
        "players": int(len(player_frames)),
        "rows": int(len(df)),
        "min_date": str(df["Date"].min().date()) if len(df) else None,
        "max_date": str(df["Date"].max().date()) if len(df) else None,
        "market_props_rows_matched": market_match_count,
        "market_anchor_rows_filled": int(market_anchor_summary.get("rows_filled", 0)),
    }
    return player_frames, summary


def write_processed_files(player_frames: dict[str, pd.DataFrame], season: int, player_limit: int | None) -> dict:
    written = {}
    items = list(sorted(player_frames.items()))
    if player_limit is not None:
        items = items[:player_limit]
    for player_name, player_df in items:
        player_dir = resolve_player_dir(player_name)
        player_dir.mkdir(parents=True, exist_ok=True)
        out_path = player_dir / f"{season}_processed_processed.csv"
        player_df.to_csv(out_path, index=False)
        written[player_name] = {
            "rows": int(len(player_df)),
            "path": str(out_path),
            "max_date": str(pd.to_datetime(player_df["Date"]).max().date()) if "Date" in player_df.columns and len(player_df) else None,
        }
    return written


def main() -> None:
    args = parse_args()
    season_dir = RAW_ROOT / f"season={args.season}"
    season_dir.mkdir(parents=True, exist_ok=True)

    scraper = NBAEnrichmentScraper(
        outdir=RAW_ROOT,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=3,
        overwrite=False,
        enabled_endpoints={"boxscore_advanced"},
        include_playbyplay=False,
    )

    logs = fetch_player_logs(scraper, args.season, args.through_date, args.max_games)
    max_api_date = logs["GAME_DATE"].max()
    print(f"Fetched {len(logs):,} player log rows across {logs['GAME_ID'].nunique():,} games.")
    print(f"Latest game date available from API: {max_api_date.date() if pd.notna(max_api_date) else 'unknown'}")

    game_ids = sorted(logs["GAME_ID"].astype(str).unique().tolist())
    if args.refresh_advanced:
        refresh_advanced_cache(scraper, args.season, game_ids)

    player_adv, team_adv = load_advanced_tables(scraper, args.season)
    print(f"Advanced player rows available: {len(player_adv):,}")
    print(f"Advanced team rows available: {len(team_adv):,}")

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

    market_anchor_bundle = None
    market_anchor_summary = {"available": False, "path": None, "targets": [], "matched_rows": 0}
    if args.apply_market_anchor:
        market_anchor_bundle, market_anchor_summary = load_market_anchor_bundle(args.market_anchor_path)
        if market_anchor_summary["available"]:
            print(
                "Market anchor bundle loaded: "
                f"{market_anchor_summary['targets']} from {market_anchor_summary['path']}"
            )
        else:
            print("Market anchor bundle not found; continuing without synthetic market fill.")

    player_frames, summary = build_processed_season(
        logs,
        player_adv,
        team_adv,
        market_props_wide=market_props_wide,
        market_anchor_bundle=market_anchor_bundle,
    )
    market_merge_summary["matched_rows"] = int(summary.get("market_props_rows_matched", 0))
    market_anchor_summary["matched_rows"] = int(summary.get("market_anchor_rows_filled", 0))
    written = write_processed_files(player_frames, args.season, args.player_limit)
    contract_repair = {"files_scanned": 0, "files_repaired": 0, "repaired": []}
    if not args.skip_contract_repair:
        contract_repair = repair_legacy_contracts_for_players(list(written.keys()))

    manifest = {
        "season": args.season,
        "through_date_requested": args.through_date,
        "api_latest_game_date": str(max_api_date.date()) if pd.notna(max_api_date) else None,
        "refresh_advanced": bool(args.refresh_advanced),
        "advanced_player_rows": int(len(player_adv)),
        "advanced_team_rows": int(len(team_adv)),
        "processed_summary": summary,
        "market_props_merge": market_merge_summary,
        "market_anchor_fill": market_anchor_summary,
        "players_written": len(written),
        "written": written,
        "legacy_contract_repair": contract_repair,
        "updated_at_utc": utc_now_iso(),
    }
    safe_write_json(PROC_ROOT / f"update_manifest_{args.season}.json", manifest)

    print("\n" + "=" * 80)
    print("DATA UPDATE COMPLETE")
    print("=" * 80)
    print(f"Season: {args.season}")
    print(f"Players written: {len(written)}")
    print(f"Processed max date: {summary['max_date']}")
    if args.merge_market_props:
        print(
            "Market props merge: "
            f"{market_merge_summary['matched_rows']} matched rows from {market_merge_summary['rows']} market rows"
        )
    if args.apply_market_anchor:
        print(
            "Market anchor fill: "
            f"{market_anchor_summary['matched_rows']} filled rows using {market_anchor_summary.get('targets', [])}"
        )
    if not args.skip_contract_repair:
        print(
            "Legacy contract repair: "
            f"{contract_repair['files_repaired']} of {contract_repair['files_scanned']} files updated"
        )
    print(f"Manifest: {PROC_ROOT / f'update_manifest_{args.season}.json'}")


if __name__ == "__main__":
    main()

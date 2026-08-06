#!/usr/bin/env python3
"""
Refresh MLB processed player files from live MLB Stats API game data.

Outputs are written to:
    Data-Proc-MLB/<Player>/<season>_processed_processed.csv

The updater is cache-aware:
- completed game feeds are stored locally per game
- future runs reuse cached feeds unless --refresh-source is passed
- processed files are rebuilt from the cached season corpus each run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sports.mlb.decision_engine.matchup_network import (  # noqa: E402
    HITTER_TARGETS as NETWORK_HITTER_TARGETS,
    NETWORK_VERSION as MATCHUP_NETWORK_VERSION,
    build_matchup_network_signal,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data copy" / "raw" / "mlb_enrichment"
MARKET_ROOT = REPO_ROOT.parent / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io"
PROC_ROOT = REPO_ROOT / "Data-Proc-MLB"
TARGETS_HITTER = ["H", "TB", "R", "HR", "RBI"]
TARGETS_PITCHER = ["K", "ER", "ERA"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    out = str(value or "").strip()
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


def deduplicate_player_games(df: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["Player", "Player_Type", "Game_ID"] if column in df.columns]
    if len(keys) < 3:
        return df.copy()
    sort_columns = [column for column in ["Player", "Date", "Game_ID", "Player_Type"] if column in df.columns]
    return (
        df.sort_values(sort_columns)
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Data-Proc-MLB season files from MLB Stats API game feeds.")
    parser.add_argument("--season", type=int, default=None, help="Season year. Defaults from --through-date/current date.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date. Defaults to March 1 of season.")
    parser.add_argument("--through-date", type=str, default=None, help="Optional inclusive YYYY-MM-DD cutoff. Defaults to today.")
    parser.add_argument("--refresh-source", action="store_true", help="Overwrite cached schedule/game feeds before processing.")
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="Remote fetch delay between uncached game calls.")
    parser.add_argument("--retries", type=int, default=3, help="Remote fetch retry count.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout per request.")
    parser.add_argument("--player-limit", type=int, default=None, help="Optional limit on number of players written.")
    parser.add_argument("--min-rows", type=int, default=11, help="Minimum rows required for a player file to be written.")
    return parser.parse_args()


def infer_season(through_date: str | None) -> int:
    if through_date:
        return int(pd.Timestamp(through_date).year)
    return int(pd.Timestamp.now().year)


def default_start_date(season: int) -> str:
    return f"{int(season)}-03-01"


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_rate_text(value: object) -> float:
    text = str(value or "").strip()
    if text in {"", "-", "--", "---", "-.--", ".---"}:
        return 0.0
    text = text.replace("%", "")
    if text.startswith("."):
        text = f"0{text}"
    try:
        out = float(text)
    except ValueError:
        return 0.0
    if out <= 1.0:
        return out
    return out / 100.0


def parse_ip(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "." not in text:
        return coerce_float(text, default=0.0)
    whole, frac = text.split(".", 1)
    outs = 0
    if frac and frac[0].isdigit():
        outs = int(frac[0])
    return max(0.0, coerce_float(whole, default=0.0) + (outs / 3.0))


def round_half(value: float, *, min_value: float = 0.5) -> float:
    return max(float(min_value), round(float(value) * 2.0) / 2.0)


def round_book_half(value: float, *, min_value: float = 0.5) -> float:
    return max(float(min_value), math.ceil(float(value)) - 0.5)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-9:
        return float(default)
    return float(num) / den


def request_json(url: str, *, timeout_seconds: float, retries: int) -> object:
    last_exc: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network variability
            last_exc = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_exc}") from last_exc


def load_json_cache(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_cache(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_schedule(
    *,
    season: int,
    start_date: str,
    through_date: str,
    refresh_source: bool,
    timeout_seconds: float,
    retries: int,
) -> list[dict]:
    cache_path = RAW_ROOT / f"season={int(season)}" / f"schedule_{start_date}_{through_date}.json"
    cached = None if refresh_source else load_json_cache(cache_path)
    if isinstance(cached, dict):
        games = cached.get("games")
        if isinstance(games, list):
            return games

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={start_date}&endDate={through_date}"
    )
    payload = request_json(url, timeout_seconds=timeout_seconds, retries=retries)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected MLB schedule payload.")
    games: list[dict] = []
    for date_bucket in payload.get("dates", []):
        for game in date_bucket.get("games", []):
            games.append(game)
    write_json_cache(cache_path, {"games": games, "updated_at_utc": utc_now_iso()})
    return games


def fetch_teams(*, season: int, refresh_source: bool, timeout_seconds: float, retries: int) -> dict[int, str]:
    cache_path = RAW_ROOT / f"season={int(season)}" / "teams.json"
    cached = None if refresh_source else load_json_cache(cache_path)
    if isinstance(cached, dict) and isinstance(cached.get("id_to_abbr"), dict):
        return {int(key): str(value) for key, value in cached["id_to_abbr"].items()}

    url = f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={int(season)}"
    payload = request_json(url, timeout_seconds=timeout_seconds, retries=retries)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected MLB teams payload.")
    id_to_abbr: dict[int, str] = {}
    for team in payload.get("teams", []):
        try:
            team_id = int(team.get("id"))
        except Exception:
            continue
        abbr = str(team.get("abbreviation") or team.get("teamCode") or "").strip().upper()
        if abbr:
            id_to_abbr[team_id] = abbr
    write_json_cache(
        cache_path,
        {"id_to_abbr": {str(key): value for key, value in id_to_abbr.items()}, "updated_at_utc": utc_now_iso()},
    )
    return id_to_abbr


def fetch_game_feed(
    *,
    season: int,
    game_pk: int,
    refresh_source: bool,
    timeout_seconds: float,
    retries: int,
) -> dict:
    cache_path = RAW_ROOT / f"season={int(season)}" / "games" / f"{int(game_pk)}.json"
    cached = None if refresh_source else load_json_cache(cache_path)
    if isinstance(cached, dict):
        return cached

    url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
    payload = request_json(url, timeout_seconds=timeout_seconds, retries=retries)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected MLB game feed payload for gamePk={game_pk}.")
    write_json_cache(cache_path, payload)
    return payload


def parse_temp_f(weather: dict) -> float:
    raw = weather.get("temp")
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?\d+", str(raw or ""))
    return float(match.group(0)) if match else 72.0


def parse_wind_out_mph(weather: dict) -> float:
    text = str(weather.get("wind") or "").strip()
    mph_match = re.search(r"(-?\d+)\s*mph", text, flags=re.IGNORECASE)
    mph = float(mph_match.group(1)) if mph_match else 0.0
    lower = text.lower()
    if "out" in lower:
        return mph
    if "in" in lower:
        return -mph
    return 0.0


def infer_park_factor(venue: dict) -> float:
    field = venue.get("fieldInfo", {}) if isinstance(venue, dict) else {}
    center = coerce_float(field.get("center"), default=400.0)
    left_line = coerce_float(field.get("leftLine"), default=330.0)
    right_line = coerce_float(field.get("rightLine"), default=330.0)
    mean_depth = np.mean([center, left_line, right_line])
    adjustment = (385.0 - mean_depth) / 300.0
    return float(np.clip(1.0 + adjustment, 0.92, 1.08))


def compute_woba(*, singles: float, doubles: float, triples: float, home_runs: float, walks: float, hbp: float, ab: float, sf: float) -> float:
    denom = ab + walks - 0.0 + hbp + sf
    if denom <= 0:
        return 0.0
    num = (0.69 * walks) + (0.72 * hbp) + (0.89 * singles) + (1.27 * doubles) + (1.62 * triples) + (2.10 * home_runs)
    return float(num / denom)


def build_game_rows(game_feed: dict, team_id_map: dict[int, str]) -> list[dict[str, object]]:
    game_pk = int(game_feed.get("gamePk"))
    game_data = game_feed.get("gameData", {})
    live_data = game_feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    box_teams = boxscore.get("teams", {})
    status = game_data.get("status", {})
    detailed_state = str(status.get("detailedState") or "")
    if detailed_state.lower() not in {"final", "game over", "completed early"}:
        return []

    game_date = str(game_data.get("datetime", {}).get("originalDate") or "")[:10]
    if not game_date:
        game_date = str(game_data.get("datetime", {}).get("dateTime") or "")[:10]
    weather = game_data.get("weather", {}) or {}
    venue = game_data.get("venue", {}) or {}
    park_factor = infer_park_factor(venue)
    temp_f = parse_temp_f(weather)
    wind_out_mph = parse_wind_out_mph(weather)

    game_teams = game_data.get("teams", {})
    probable_pitchers = game_data.get("probablePitchers", {}) or {}
    team_pitching_order = {
        side: [int(value) for value in (box_teams.get(side, {}).get("pitchers") or [])]
        for side in ("home", "away")
    }
    starter_ids: dict[str, int | None] = {}
    for side in ("home", "away"):
        probable = probable_pitchers.get(side, {}) if isinstance(probable_pitchers, dict) else {}
        probable_id = probable.get("id")
        if probable_id is not None:
            starter_ids[side] = int(probable_id)
        elif team_pitching_order.get(side):
            starter_ids[side] = int(team_pitching_order[side][0])
        else:
            starter_ids[side] = None

    rows: list[dict[str, object]] = []
    for side in ("home", "away"):
        team_box = box_teams.get(side, {}) or {}
        opp_side = "away" if side == "home" else "home"
        opp_box = box_teams.get(opp_side, {}) or {}
        team_meta = game_teams.get(side, {}) or {}
        opp_meta = game_teams.get(opp_side, {}) or {}
        team_id = int(team_meta.get("id") or 0)
        opp_id = int(opp_meta.get("id") or 0)
        team_abbr = str(team_meta.get("abbreviation") or team_id_map.get(team_id) or "").upper()
        opp_abbr = str(opp_meta.get("abbreviation") or team_id_map.get(opp_id) or "").upper()
        team_batting_totals = team_box.get("teamStats", {}).get("batting", {}) or {}
        team_pa = max(1.0, coerce_float(team_batting_totals.get("plateAppearances"), default=0.0))
        team_players = team_box.get("players", {}) or {}
        opp_starter_id = starter_ids.get(opp_side)
        own_starter_id = starter_ids.get(side)

        for player_payload in team_players.values():
            person = player_payload.get("person", {}) or {}
            player_id = int(person.get("id") or 0)
            player_name = normalize_name(person.get("fullName", ""))
            if not player_id or not player_name:
                continue

            stats = player_payload.get("stats", {}) or {}
            batting = stats.get("batting", {}) or {}
            pitching = stats.get("pitching", {}) or {}
            position = player_payload.get("position", {}) or {}
            batting_order_raw = str(player_payload.get("battingOrder") or "").strip()

            pa = coerce_float(batting.get("plateAppearances"), default=0.0)
            pitched_ip = parse_ip(pitching.get("inningsPitched"))
            pitches = coerce_float(pitching.get("numberOfPitches"), default=0.0)
            outs = coerce_float(pitching.get("outs"), default=pitched_ip * 3.0)
            is_pitcher_row = pitched_ip > 0.0 or pitches > 0.0 or outs > 0.0
            is_hitter_row = (pa > 0.0 or batting_order_raw) and not is_pitcher_row

            if is_hitter_row:
                hits = coerce_float(batting.get("hits"), default=0.0)
                home_runs = coerce_float(batting.get("homeRuns"), default=0.0)
                doubles = coerce_float(batting.get("doubles"), default=0.0)
                triples = coerce_float(batting.get("triples"), default=0.0)
                walks = coerce_float(batting.get("baseOnBalls"), default=0.0)
                strikeouts = coerce_float(batting.get("strikeOuts"), default=0.0)
                ab = coerce_float(batting.get("atBats"), default=0.0)
                rbi = coerce_float(batting.get("rbi"), default=0.0)
                total_bases = coerce_float(batting.get("totalBases"), default=0.0)
                runs = coerce_float(batting.get("runs"), default=0.0)
                stolen_bases = coerce_float(batting.get("stolenBases"), default=0.0)
                hbp = coerce_float(batting.get("hitByPitch"), default=0.0)
                sf = coerce_float(batting.get("sacFlies"), default=0.0)
                singles = max(0.0, hits - doubles - triples - home_runs)
                woba = compute_woba(
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    home_runs=home_runs,
                    walks=walks,
                    hbp=hbp,
                    ab=ab,
                    sf=sf,
                )
                iso = safe_div(total_bases - hits, ab, default=0.0)
                xwoba = woba
                barrel_pct = float(np.clip((iso * 85.0) + (home_runs * 2.5), 0.0, 30.0))
                hard_hit_pct = float(np.clip((woba * 120.0) + (iso * 45.0), 10.0, 65.0))
                batting_order = coerce_float(batting_order_raw[:1], default=9.0) if batting_order_raw else 9.0

                rows.append(
                    {
                        "Date": game_date,
                        "Player": player_name,
                        "Player_MLBAM_ID": player_id,
                        "Player_Type": "hitter",
                        "Team": team_abbr,
                        "Opponent": opp_abbr,
                        "Season": int(pd.Timestamp(game_date).year),
                        "Game_ID": str(game_pk),
                        "Team_ID": team_id,
                        "Opponent_ID": opp_id,
                        "Is_Home": 1 if side == "home" else 0,
                        "H": hits,
                        "HR": home_runs,
                        "RBI": rbi,
                        "TB": total_bases,
                        "R": runs,
                        "PA": pa,
                        "AB": ab,
                        "BB": walks,
                        "SO": strikeouts,
                        "SB": stolen_bases,
                        "Batting_Order": batting_order,
                        "Team_PA_share": safe_div(pa, team_pa, default=0.0),
                        "wOBA": woba,
                        "xwOBA": xwoba,
                        "ISO": iso,
                        "Barrel%": barrel_pct,
                        "HardHit%": hard_hit_pct,
                        "Opp_Starter_ID": int(opp_starter_id) if opp_starter_id else 0,
                        "Own_Starter_ID": int(own_starter_id) if own_starter_id else 0,
                        "Park_Factor": park_factor,
                        "Wind_Out_MPH": wind_out_mph,
                        "Temp_F": temp_f,
                        "Did_Not_Play": 0,
                    }
                )
                continue

            if is_pitcher_row:
                strikeouts = coerce_float(pitching.get("strikeOuts"), default=0.0)
                earned_runs = coerce_float(pitching.get("earnedRuns"), default=0.0)
                hits_allowed = coerce_float(pitching.get("hits"), default=0.0)
                walks_allowed = coerce_float(pitching.get("baseOnBalls"), default=0.0)
                home_runs_allowed = coerce_float(pitching.get("homeRuns"), default=0.0)
                bf = coerce_float(pitching.get("battersFaced"), default=0.0)
                strike_pct = parse_rate_text(pitching.get("strikePercentage"))
                era = safe_div(earned_runs * 9.0, pitched_ip, default=0.0)
                fip = safe_div((13.0 * home_runs_allowed) + (3.0 * walks_allowed) - (2.0 * strikeouts), pitched_ip, default=0.0) + 3.2
                xfip = fip
                csw_pct = float(np.clip(strike_pct * 100.0, 0.0, 100.0))
                whiff_pct = float(np.clip(safe_div(strikeouts * 3.0, max(pitches, 1.0), default=0.0) * 100.0, 0.0, 60.0))
                was_starter = 1 if own_starter_id and int(own_starter_id) == player_id else 0

                rows.append(
                    {
                        "Date": game_date,
                        "Player": player_name,
                        "Player_MLBAM_ID": player_id,
                        "Player_Type": "pitcher",
                        "Team": team_abbr,
                        "Opponent": opp_abbr,
                        "Season": int(pd.Timestamp(game_date).year),
                        "Game_ID": str(game_pk),
                        "Team_ID": team_id,
                        "Opponent_ID": opp_id,
                        "Is_Home": 1 if side == "home" else 0,
                        "K": strikeouts,
                        "ER": earned_runs,
                        "ERA": era,
                        "IP": pitched_ip,
                        "BF": bf,
                        "Pitches": pitches,
                        "BB_allowed": walks_allowed,
                        "H_allowed": hits_allowed,
                        "HR_allowed": home_runs_allowed,
                        "FIP": fip,
                        "xFIP": xfip,
                        "CSW%": csw_pct,
                        "Whiff%": whiff_pct,
                        "Was_Starter": was_starter,
                        "Park_Factor": park_factor,
                        "Wind_Out_MPH": wind_out_mph,
                        "Temp_F": temp_f,
                        "Did_Not_Play": 0,
                    }
                )

    return rows


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month = out["Date"].dt.month.astype(float)
    dow = out["Date"].dt.dayofweek.astype(float)
    out["Month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    out["Month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    out["DayOfWeek_sin"] = np.sin(2.0 * math.pi * dow / 7.0)
    out["DayOfWeek_cos"] = np.cos(2.0 * math.pi * dow / 7.0)
    return out


def attach_walk_forward_matchup_network(df: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly pregame batter/pitcher network features to hitter rows."""

    out = df.copy()
    hitter_mask = out["Player_Type"].eq("hitter")
    network_float_columns = [
        "Matchup_Network_Batter_Support",
        "Matchup_Network_Pitcher_Support",
        "Pitcher_Profile_Uncertainty",
        "Batter_Vs_Starter_Games",
        "Matchup_Network_Confidence",
    ]
    for target in NETWORK_HITTER_TARGETS:
        network_float_columns.extend(
            [
                f"Batter_Profile_{target}_Strength",
                f"Pitcher_Profile_{target}_Vulnerability",
                f"Batter_Vs_Starter_{target}_Lift",
                f"Matchup_Network_{target}_Score",
                f"Matchup_Network_{target}_Adjustment",
            ]
        )
    for column in network_float_columns:
        out[column] = 0.0
    out["Pitcher_Profile_Uncertainty"] = 1.0
    out["Matchup_Network_Version"] = ""

    pitcher_rows = out.loc[out["Player_Type"].eq("pitcher")].copy()
    pitcher_by_id: dict[int, pd.DataFrame] = {}
    if "Player_MLBAM_ID" in pitcher_rows.columns:
        pitcher_rows["_player_id"] = pd.to_numeric(
            pitcher_rows["Player_MLBAM_ID"], errors="coerce"
        ).fillna(0).astype(int)
        pitcher_by_id = {
            int(player_id): part.sort_values(["Date", "Game_Index"]).copy()
            for player_id, part in pitcher_rows.loc[pitcher_rows["_player_id"].gt(0)].groupby("_player_id")
        }

    batter_ids = (
        pd.to_numeric(out["Player_MLBAM_ID"], errors="coerce").fillna(0).astype(int)
        if "Player_MLBAM_ID" in out.columns
        else pd.Series(0, index=out.index, dtype=int)
    )
    out["_matchup_batter_key"] = [
        f"id:{player_id}" if player_id > 0 else f"name:{player_name}"
        for player_id, player_name in zip(batter_ids, out["Player"].astype(str))
    ]
    for _, batter_rows in out.loc[hitter_mask].groupby("_matchup_batter_key", sort=False):
        ordered = batter_rows.sort_values(["Date", "Game_Index"])
        for position, (row_index, row) in enumerate(ordered.iterrows()):
            batter_history = ordered.iloc[:position].copy()
            if batter_history.empty:
                continue
            pitcher_id = int(coerce_float(row.get("Opp_Starter_ID"), default=0.0))
            pitcher_history = pitcher_by_id.get(pitcher_id, pd.DataFrame())
            if not pitcher_history.empty:
                pitcher_history = pitcher_history.loc[pitcher_history["Date"] < row["Date"]].copy()
            signal = build_matchup_network_signal(
                batter_history,
                pitcher_history,
                opposing_pitcher_id=pitcher_id,
                opposing_pitcher_name=row.get("Opp_Starter_Player", ""),
            )
            out.at[row_index, "Matchup_Network_Version"] = MATCHUP_NETWORK_VERSION
            out.at[row_index, "Matchup_Network_Batter_Support"] = signal.batter_support
            out.at[row_index, "Matchup_Network_Pitcher_Support"] = signal.pitcher_support
            out.at[row_index, "Pitcher_Profile_Uncertainty"] = signal.pitcher_uncertainty
            out.at[row_index, "Batter_Vs_Starter_Games"] = signal.direct_matchup_games
            out.at[row_index, "Matchup_Network_Confidence"] = signal.confidence
            for target in NETWORK_HITTER_TARGETS:
                out.at[row_index, f"Batter_Profile_{target}_Strength"] = signal.batter_strength[target]
                out.at[row_index, f"Pitcher_Profile_{target}_Vulnerability"] = signal.pitcher_vulnerability[target]
                out.at[row_index, f"Batter_Vs_Starter_{target}_Lift"] = signal.direct_matchup_lift[target]
                out.at[row_index, f"Matchup_Network_{target}_Score"] = signal.network_score[target]
                out.at[row_index, f"Matchup_Network_{target}_Adjustment"] = signal.adjustment[target]
    return out.drop(columns=["_matchup_batter_key"])


def load_market_history() -> pd.DataFrame:
    candidates = [
        MARKET_ROOT / "history_player_props_wide.parquet",
        MARKET_ROOT / "history_player_props_wide.csv",
        MARKET_ROOT / "latest_player_props_wide.parquet",
        MARKET_ROOT / "latest_player_props_wide.csv",
    ]
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        return pd.DataFrame()
    if selected.suffix.lower() == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)
    if df.empty:
        return df
    if "Player" not in df.columns or "Market_Date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce").dt.date.astype(str)
    return df.drop_duplicates(subset=["Market_Date", "Player"], keep="last").reset_index(drop=True)


def build_processed_frames(
    raw_rows: list[dict[str, object]],
    fetched_at_utc: str,
    market_history: pd.DataFrame,
    *,
    min_rows: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    if not raw_rows:
        return {}, {"players": 0, "rows": 0, "min_date": None, "max_date": None, "market_props_rows_matched": 0, "market_rows_filled": 0}

    df = pd.DataFrame(raw_rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[df["Date"].notna()].copy()
    df = deduplicate_player_games(df)
    df["Rest_Days"] = df.groupby("Player")["Date"].diff().dt.days.fillna(2).clip(lower=0).astype(float)
    df["Game_Index"] = df.groupby("Player").cumcount().astype(int)
    df = add_time_features(df)

    hitter_mask = df["Player_Type"] == "hitter"
    pitcher_mask = df["Player_Type"] == "pitcher"

    if hitter_mask.any():
        team_daily = (
            df.loc[hitter_mask]
            .groupby(["Team", "Date"], as_index=False)
            .agg(
                team_woba=("wOBA", "mean"),
                team_k_rate=("SO", lambda s: safe_div(float(np.sum(s)), float(np.sum(df.loc[s.index, "PA"])), default=0.22)),
            )
            .sort_values(["Team", "Date"])
        )
        team_daily["Opp_Lineup_wOBA_3"] = team_daily.groupby("Team")["team_woba"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        team_daily["Opp_Lineup_K_rate_3"] = team_daily.groupby("Team")["team_k_rate"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    else:
        team_daily = pd.DataFrame(columns=["Team", "Date", "Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3"])

    if pitcher_mask.any():
        pitcher_rows = df.loc[pitcher_mask].copy()
        pitcher_rows["K9"] = pitcher_rows.apply(lambda row: safe_div(float(row["K"]) * 9.0, float(row["IP"]), default=0.0), axis=1)
        pitcher_rows["Opp_Pitcher_ERA_3"] = pitcher_rows.groupby("Player")["ERA"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        pitcher_rows["Opp_Pitcher_K9_3"] = pitcher_rows.groupby("Player")["K9"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        starter_context = pitcher_rows[["Player", "Date", "Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3"]].copy()
        starter_context["Player"] = starter_context["Player"].astype(str)
        starter_context = starter_context.rename(columns={"Player": "Opp_Starter_Player"})

        bullpen_daily = pitcher_rows.loc[pitcher_rows["Was_Starter"].fillna(0).astype(int) == 0].copy()
        if bullpen_daily.empty:
            bullpen_context = pd.DataFrame(columns=["Team", "Date", "Opp_Bullpen_ERA_7"])
        else:
            bullpen_daily = (
                bullpen_daily.groupby(["Team", "Date"], as_index=False)
                .agg(bullpen_er=("ER", "sum"), bullpen_ip=("IP", "sum"))
                .sort_values(["Team", "Date"])
            )
            bullpen_daily["bullpen_era"] = bullpen_daily.apply(
                lambda row: safe_div(float(row["bullpen_er"]) * 9.0, float(row["bullpen_ip"]), default=4.0),
                axis=1,
            )
            bullpen_daily["Opp_Bullpen_ERA_7"] = bullpen_daily.groupby("Team")["bullpen_era"].transform(
                lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
            )
            bullpen_context = bullpen_daily[["Team", "Date", "Opp_Bullpen_ERA_7"]].copy()
    else:
        pitcher_rows = pd.DataFrame()
        starter_context = pd.DataFrame(columns=["Opp_Starter_Player", "Date", "Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3"])
        bullpen_context = pd.DataFrame(columns=["Team", "Date", "Opp_Bullpen_ERA_7"])

    if hitter_mask.any():
        pitcher_name_by_id = {
            int(row["Player_MLBAM_ID"]): str(row["Player"])
            for _, row in df.loc[
                pitcher_mask & pd.to_numeric(df["Player_MLBAM_ID"], errors="coerce").fillna(0).gt(0),
                ["Player_MLBAM_ID", "Player"],
            ].drop_duplicates(subset=["Player_MLBAM_ID"], keep="last").iterrows()
        }
        df.loc[hitter_mask, "Opp_Starter_Player"] = df.loc[hitter_mask, "Opp_Starter_ID"].map(
            lambda value: pitcher_name_by_id.get(int(coerce_float(value, default=0.0)), "")
        )
        df = df.merge(starter_context, how="left", on=["Opp_Starter_Player", "Date"])
        df = df.merge(
            bullpen_context.rename(columns={"Team": "Opponent"}),
            how="left",
            on=["Opponent", "Date"],
        )
        df["Opp_Pitcher_ERA_3"] = pd.to_numeric(df.get("Opp_Pitcher_ERA_3"), errors="coerce").fillna(4.1)
        df["Opp_Pitcher_K9_3"] = pd.to_numeric(df.get("Opp_Pitcher_K9_3"), errors="coerce").fillna(8.2)
        df["Opp_Bullpen_ERA_7"] = pd.to_numeric(df.get("Opp_Bullpen_ERA_7"), errors="coerce").fillna(4.0)

    if pitcher_mask.any():
        df = df.merge(
            team_daily[["Team", "Date", "Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3"]].rename(columns={"Team": "Opponent"}),
            how="left",
            on=["Opponent", "Date"],
        )
        df["Opp_Lineup_wOBA_3"] = pd.to_numeric(df.get("Opp_Lineup_wOBA_3"), errors="coerce").fillna(0.315)
        df["Opp_Lineup_K_rate_3"] = pd.to_numeric(df.get("Opp_Lineup_K_rate_3"), errors="coerce").fillna(0.225)

    # Suspended games and context joins can duplicate a player-game row.
    df = deduplicate_player_games(df)
    df["Game_Index"] = df.groupby("Player").cumcount().astype(int)

    # Context merges can expand and reindex rows, so masks created before them are no longer safe.
    hitter_mask = df["Player_Type"].eq("hitter")
    pitcher_mask = df["Player_Type"].eq("pitcher")

    df = attach_walk_forward_matchup_network(df)

    rolling_specs = {
        "hitter": ["H", "TB", "R", "HR", "RBI"],
        "pitcher": ["K", "ER", "ERA"],
    }
    for role, targets in rolling_specs.items():
        role_mask = df["Player_Type"] == role
        if not role_mask.any():
            continue
        for target in targets:
            series = df.loc[role_mask].groupby("Player")[target]
            df.loc[role_mask, f"{target}_rolling_avg"] = series.transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
            df.loc[role_mask, f"{target}_lag1"] = series.transform(lambda s: s.shift(1))
            df.loc[role_mask, f"{target}_rolling_avg"] = pd.to_numeric(df.loc[role_mask, f"{target}_rolling_avg"], errors="coerce").fillna(df.loc[role_mask, target])
            df.loc[role_mask, f"{target}_lag1"] = pd.to_numeric(df.loc[role_mask, f"{target}_lag1"], errors="coerce").fillna(df.loc[role_mask, target])

    if hitter_mask.any():
        df.loc[hitter_mask, "H_proj"] = (
            (0.68 * pd.to_numeric(df.loc[hitter_mask, "H_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.14 * pd.to_numeric(df.loc[hitter_mask, "H_lag1"], errors="coerce").fillna(0.0))
            + (0.35 * pd.to_numeric(df.loc[hitter_mask, "Team_PA_share"], errors="coerce").fillna(0.0) * 4.2)
            + (0.12 * (pd.to_numeric(df.loc[hitter_mask, "Park_Factor"], errors="coerce").fillna(1.0) - 1.0) * 4.0)
            + (0.07 * ((pd.to_numeric(df.loc[hitter_mask, "Temp_F"], errors="coerce").fillna(70.0) - 65.0) / 15.0))
            - (0.05 * (pd.to_numeric(df.loc[hitter_mask, "Opp_Pitcher_K9_3"], errors="coerce").fillna(8.2) - 8.0))
            + (0.04 * (pd.to_numeric(df.loc[hitter_mask, "Opp_Bullpen_ERA_7"], errors="coerce").fillna(4.0) - 4.0))
        ).clip(lower=0.0)
        df.loc[hitter_mask, "TB_proj"] = (
            (0.62 * pd.to_numeric(df.loc[hitter_mask, "TB_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.14 * pd.to_numeric(df.loc[hitter_mask, "TB_lag1"], errors="coerce").fillna(0.0))
            + (0.26 * pd.to_numeric(df.loc[hitter_mask, "Team_PA_share"], errors="coerce").fillna(0.0) * 4.2)
            + (1.20 * pd.to_numeric(df.loc[hitter_mask, "ISO"], errors="coerce").fillna(0.0))
            + (0.45 * (pd.to_numeric(df.loc[hitter_mask, "wOBA"], errors="coerce").fillna(0.315) - 0.315))
            + (0.12 * (pd.to_numeric(df.loc[hitter_mask, "Park_Factor"], errors="coerce").fillna(1.0) - 1.0) * 4.0)
        ).clip(lower=0.0)
        df.loc[hitter_mask, "R_proj"] = (
            (0.64 * pd.to_numeric(df.loc[hitter_mask, "R_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.14 * pd.to_numeric(df.loc[hitter_mask, "R_lag1"], errors="coerce").fillna(0.0))
            + (0.30 * pd.to_numeric(df.loc[hitter_mask, "Team_PA_share"], errors="coerce").fillna(0.0) * 4.2)
            + (0.55 * (pd.to_numeric(df.loc[hitter_mask, "wOBA"], errors="coerce").fillna(0.315) - 0.315))
            + (0.08 * (1.0 - (pd.to_numeric(df.loc[hitter_mask, "Batting_Order"], errors="coerce").fillna(9.0) - 1.0) / 8.0))
            + (0.05 * (pd.to_numeric(df.loc[hitter_mask, "Opp_Bullpen_ERA_7"], errors="coerce").fillna(4.0) - 4.0))
        ).clip(lower=0.0)
        df.loc[hitter_mask, "HR_proj"] = (
            (0.70 * pd.to_numeric(df.loc[hitter_mask, "HR_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.10 * pd.to_numeric(df.loc[hitter_mask, "HR_lag1"], errors="coerce").fillna(0.0))
            + (0.25 * pd.to_numeric(df.loc[hitter_mask, "ISO"], errors="coerce").fillna(0.0))
            + (0.0025 * pd.to_numeric(df.loc[hitter_mask, "Barrel%"], errors="coerce").fillna(0.0))
            + (0.08 * (pd.to_numeric(df.loc[hitter_mask, "Park_Factor"], errors="coerce").fillna(1.0) - 1.0) * 4.0)
        ).clip(lower=0.0)
        df.loc[hitter_mask, "RBI_proj"] = (
            (0.68 * pd.to_numeric(df.loc[hitter_mask, "RBI_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.16 * pd.to_numeric(df.loc[hitter_mask, "RBI_lag1"], errors="coerce").fillna(0.0))
            + (0.28 * pd.to_numeric(df.loc[hitter_mask, "Team_PA_share"], errors="coerce").fillna(0.0) * 4.2)
            + (0.30 * (pd.to_numeric(df.loc[hitter_mask, "wOBA"], errors="coerce").fillna(0.0) - 0.31))
            + (0.05 * (pd.to_numeric(df.loc[hitter_mask, "Opp_Bullpen_ERA_7"], errors="coerce").fillna(4.0) - 4.0))
        ).clip(lower=0.0)
        for target in NETWORK_HITTER_TARGETS:
            adjustment_column = f"Matchup_Network_{target}_Adjustment"
            projection_column = f"{target}_proj"
            df.loc[hitter_mask, projection_column] = (
                pd.to_numeric(df.loc[hitter_mask, projection_column], errors="coerce").fillna(0.0)
                + pd.to_numeric(df.loc[hitter_mask, adjustment_column], errors="coerce").fillna(0.0)
            ).clip(lower=0.0)

    if pitcher_mask.any():
        df.loc[pitcher_mask, "K_proj"] = (
            (0.72 * pd.to_numeric(df.loc[pitcher_mask, "K_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.14 * pd.to_numeric(df.loc[pitcher_mask, "K_lag1"], errors="coerce").fillna(0.0))
            + (8.0 * (pd.to_numeric(df.loc[pitcher_mask, "Opp_Lineup_K_rate_3"], errors="coerce").fillna(0.225) - 0.20))
            + (0.10 * (pd.to_numeric(df.loc[pitcher_mask, "Park_Factor"], errors="coerce").fillna(1.0) - 1.0) * -4.0)
        ).clip(lower=0.0)
        df.loc[pitcher_mask, "ER_proj"] = (
            (0.72 * pd.to_numeric(df.loc[pitcher_mask, "ER_rolling_avg"], errors="coerce").fillna(0.0))
            + (0.16 * pd.to_numeric(df.loc[pitcher_mask, "ER_lag1"], errors="coerce").fillna(0.0))
            + (4.5 * (pd.to_numeric(df.loc[pitcher_mask, "Opp_Lineup_wOBA_3"], errors="coerce").fillna(0.315) - 0.300))
            + (0.18 * (pd.to_numeric(df.loc[pitcher_mask, "Park_Factor"], errors="coerce").fillna(1.0) - 1.0) * 4.0)
        ).clip(lower=0.0)
        projected_ip = pd.to_numeric(df.loc[pitcher_mask, "IP"], errors="coerce").fillna(5.5)
        projected_ip = projected_ip.where(projected_ip > 0.0, 5.5)
        df.loc[pitcher_mask, "ERA_proj"] = (pd.to_numeric(df.loc[pitcher_mask, "ER_proj"], errors="coerce").fillna(0.0) * 9.0 / projected_ip).clip(lower=0.0)

    if not market_history.empty:
        merge_cols = ["Market_Date", "Player"]
        df["Market_Date"] = df["Date"].dt.date.astype(str)
        market_merge_cols = [col for col in market_history.columns if col in merge_cols or col.startswith("Market_")]
        df = df.merge(market_history[market_merge_cols], how="left", on=merge_cols)
        market_match_count = int(df.get("Market_Fetched_At_UTC", pd.Series(dtype="object")).notna().sum())
    else:
        market_match_count = 0

    def _ensure_market_cols(prefixes: list[str]) -> None:
        for target in prefixes:
            for suffix in ["", "_books", "_over_price", "_under_price", "_line_std"]:
                col = f"Market_{target}{suffix}"
                if col not in df.columns:
                    df[col] = np.nan

    _ensure_market_cols(["H", "TB", "R", "HR", "RBI", "K", "ER", "ERA"])
    if "Market_Fetched_At_UTC" not in df.columns:
        df["Market_Fetched_At_UTC"] = fetched_at_utc

    market_rows_filled = 0
    synthetic_specs = {
        "H": ("H_proj", "H_rolling_avg", 0.5),
        "TB": ("TB_proj", "TB_rolling_avg", 0.5),
        "R": ("R_proj", "R_rolling_avg", 0.5),
        "HR": ("HR_proj", "HR_rolling_avg", 0.5),
        "RBI": ("RBI_proj", "RBI_rolling_avg", 0.5),
        "K": ("K_proj", "K_rolling_avg", 2.5),
        "ER": ("ER_proj", "ER_rolling_avg", 0.5),
        "ERA": ("ERA_proj", "ERA_rolling_avg", 1.5),
    }
    for target, (proj_col, baseline_col, min_line) in synthetic_specs.items():
        market_col = f"Market_{target}"
        synth_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        over_col = f"Market_{target}_over_price"
        under_col = f"Market_{target}_under_price"
        books_col = f"Market_{target}_books"
        std_col = f"Market_{target}_line_std"
        gap_col = f"{target}_market_gap"

        if source_col not in df.columns:
            df[source_col] = ""
        baseline = pd.to_numeric(df.get(baseline_col), errors="coerce").fillna(0.0)
        projection = pd.to_numeric(df.get(proj_col), errors="coerce").fillna(baseline)
        synthetic_line = baseline.map(lambda value: round_book_half(float(value), min_value=min_line))
        if target == "HR":
            synthetic_line = pd.Series([0.5] * len(df), index=df.index, dtype=float)
        elif target == "ERA":
            synthetic_line = np.maximum(1.5, np.round(baseline.astype(float), 1))
        df[synth_col] = synthetic_line.astype(float)

        real_mask = pd.to_numeric(df.get(market_col), errors="coerce").notna()
        fill_mask = ~real_mask
        df.loc[fill_mask, market_col] = df.loc[fill_mask, synth_col]
        df.loc[real_mask, source_col] = "real"
        df.loc[fill_mask, source_col] = "synthetic"
        df.loc[fill_mask, over_col] = df.loc[fill_mask, over_col].fillna(-110)
        df.loc[fill_mask, under_col] = df.loc[fill_mask, under_col].fillna(-110)
        df.loc[fill_mask, books_col] = df.loc[fill_mask, books_col].fillna(0)
        df.loc[fill_mask, std_col] = df.loc[fill_mask, std_col].fillna(0.0)
        df[gap_col] = projection - pd.to_numeric(df[market_col], errors="coerce").fillna(df[synth_col])
        market_rows_filled += int(fill_mask.sum())

    df["Market_Fetched_At_UTC"] = df["Market_Fetched_At_UTC"].fillna(fetched_at_utc).astype(str)

    for col in ["Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3", "Opp_Bullpen_ERA_7"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3"]:
        if col not in df.columns:
            df[col] = 0.0

    hitter_keep = [
        "Date", "Player", "Player_MLBAM_ID", "Player_Type", "Team", "Opponent", "Season", "Game_ID", "Game_Index", "Team_ID", "Opponent_ID",
        "Is_Home", "Opp_Starter_ID", "Opp_Starter_Player", "H", "TB", "R", "HR", "RBI", "PA", "AB", "BB", "SO", "Batting_Order", "Team_PA_share", "wOBA", "xwOBA", "ISO",
        "Barrel%", "HardHit%", "Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3", "Opp_Bullpen_ERA_7", "Park_Factor", "Wind_Out_MPH",
        "Temp_F", "Did_Not_Play", "Rest_Days", "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos", "Market_H",
        "Market_TB", "Market_R", "Market_HR", "Market_RBI", "Synthetic_Market_H", "Synthetic_Market_TB", "Synthetic_Market_R",
        "Synthetic_Market_HR", "Synthetic_Market_RBI", "Market_Source_H", "Market_Source_TB", "Market_Source_R",
        "Market_Source_HR", "Market_Source_RBI", "Market_H_books", "Market_TB_books", "Market_R_books", "Market_HR_books",
        "Market_RBI_books", "Market_H_over_price", "Market_TB_over_price", "Market_R_over_price", "Market_HR_over_price",
        "Market_RBI_over_price", "Market_H_under_price", "Market_TB_under_price", "Market_R_under_price", "Market_HR_under_price",
        "Market_RBI_under_price", "Market_H_line_std", "Market_TB_line_std", "Market_R_line_std", "Market_HR_line_std",
        "Market_RBI_line_std", "Market_Fetched_At_UTC", "H_market_gap", "TB_market_gap", "R_market_gap", "HR_market_gap",
        "RBI_market_gap", "H_rolling_avg", "TB_rolling_avg", "R_rolling_avg", "HR_rolling_avg", "RBI_rolling_avg",
        "H_lag1", "TB_lag1", "R_lag1", "HR_lag1", "RBI_lag1",
    ]
    hitter_keep.extend(
        [
            "Matchup_Network_Version",
            "Matchup_Network_Batter_Support",
            "Matchup_Network_Pitcher_Support",
            "Pitcher_Profile_Uncertainty",
            "Batter_Vs_Starter_Games",
            "Matchup_Network_Confidence",
        ]
    )
    for target in NETWORK_HITTER_TARGETS:
        hitter_keep.extend(
            [
                f"Batter_Profile_{target}_Strength",
                f"Pitcher_Profile_{target}_Vulnerability",
                f"Batter_Vs_Starter_{target}_Lift",
                f"Matchup_Network_{target}_Score",
                f"Matchup_Network_{target}_Adjustment",
            ]
        )
    pitcher_keep = [
        "Date", "Player", "Player_MLBAM_ID", "Player_Type", "Team", "Opponent", "Season", "Game_ID", "Game_Index", "Team_ID", "Opponent_ID",
        "Is_Home", "Was_Starter", "K", "ER", "ERA", "IP", "BF", "Pitches", "BB_allowed", "H_allowed", "HR_allowed", "FIP", "xFIP",
        "CSW%", "Whiff%", "Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3", "Park_Factor", "Wind_Out_MPH", "Temp_F", "Did_Not_Play",
        "Rest_Days", "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos", "Market_K", "Market_ER", "Market_ERA",
        "Synthetic_Market_K", "Synthetic_Market_ER", "Synthetic_Market_ERA", "Market_Source_K", "Market_Source_ER",
        "Market_Source_ERA", "Market_K_books", "Market_ER_books", "Market_ERA_books", "Market_K_over_price",
        "Market_ER_over_price", "Market_ERA_over_price", "Market_K_under_price", "Market_ER_under_price",
        "Market_ERA_under_price", "Market_K_line_std", "Market_ER_line_std", "Market_ERA_line_std", "Market_Fetched_At_UTC",
        "K_market_gap", "ER_market_gap", "ERA_market_gap", "K_rolling_avg", "ER_rolling_avg", "ERA_rolling_avg", "K_lag1",
        "ER_lag1", "ERA_lag1",
    ]

    player_frames: dict[str, pd.DataFrame] = {}
    skipped_short_history = 0
    for player_name, player_df in df.groupby("Player", sort=True):
        role = str(player_df["Player_Type"].astype(str).value_counts().idxmax())
        player_df = player_df.loc[player_df["Player_Type"].astype(str) == role].copy()
        keep_cols = hitter_keep if role == "hitter" else pitcher_keep
        out = player_df[[col for col in keep_cols if col in player_df.columns]].copy()
        numeric_cols = [
            col
            for col in out.columns
            if col
            not in {
                "Date",
                "Player",
                "Player_Type",
                "Team",
                "Opponent",
                "Game_ID",
                "Market_Source_H",
                "Market_Source_TB",
                "Market_Source_R",
                "Market_Source_HR",
                "Market_Source_RBI",
                "Market_Source_K",
                "Market_Source_ER",
                "Market_Source_ERA",
                "Market_Fetched_At_UTC",
                "Opp_Starter_Player",
                "Matchup_Network_Version",
            }
        ]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
        out = out.reset_index(drop=True)
        if len(out) < int(min_rows):
            skipped_short_history += 1
            continue
        player_frames[player_name] = out

    summary = {
        "players": int(len(player_frames)),
        "rows": int(len(df)),
        "min_date": str(df["Date"].min().date()) if len(df) and pd.notna(df["Date"].min()) else None,
        "max_date": str(df["Date"].max().date()) if len(df) and pd.notna(df["Date"].max()) else None,
        "market_props_rows_matched": int(market_match_count),
        "market_rows_filled": int(market_rows_filled),
        "players_skipped_short_history": int(skipped_short_history),
        "matchup_network_version": MATCHUP_NETWORK_VERSION,
        "matchup_network_rows": int(
            df.get("Matchup_Network_Version", pd.Series("", index=df.index)).astype(str).eq(MATCHUP_NETWORK_VERSION).sum()
        ),
        "matchup_network_direct_history_rows": int(
            pd.to_numeric(df.get("Batter_Vs_Starter_Games"), errors="coerce").fillna(0).gt(0).sum()
        ),
    }
    return player_frames, summary


def write_processed_files(player_frames: dict[str, pd.DataFrame], season: int, player_limit: int | None) -> dict:
    written: dict[str, dict[str, object]] = {}
    items = list(sorted(player_frames.items()))
    if player_limit is not None:
        items = items[: int(player_limit)]

    for player_name, player_df in items:
        player_dir = PROC_ROOT / player_name
        player_dir.mkdir(parents=True, exist_ok=True)
        out_path = player_dir / f"{int(season)}_processed_processed.csv"
        player_df.to_csv(out_path, index=False)
        written[player_name] = {
            "rows": int(len(player_df)),
            "path": str(out_path),
            "max_date": str(pd.to_datetime(player_df["Date"], errors="coerce").max().date()) if len(player_df) else None,
            "player_type": str(player_df["Player_Type"].iloc[0]) if len(player_df) else None,
        }
    return written


def prune_stale_player_dirs(active_players: set[str]) -> int:
    removed = 0
    for path in PROC_ROOT.iterdir():
        if not path.is_dir():
            continue
        if path.name in {"schema", "__pycache__"}:
            continue
        if path.name not in active_players:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    through_date = str((pd.Timestamp(args.through_date).date() if args.through_date else pd.Timestamp.now().date()))
    season = int(args.season or infer_season(through_date))
    start_date = str(args.start_date or default_start_date(season))

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    games = fetch_schedule(
        season=season,
        start_date=start_date,
        through_date=through_date,
        refresh_source=bool(args.refresh_source),
        timeout_seconds=float(args.timeout_seconds),
        retries=int(args.retries),
    )
    teams = fetch_teams(
        season=season,
        refresh_source=bool(args.refresh_source),
        timeout_seconds=float(args.timeout_seconds),
        retries=int(args.retries),
    )

    completed_games = []
    for game in games:
        detailed_state = str(game.get("status", {}).get("detailedState") or "").strip().lower()
        if detailed_state in {"final", "game over", "completed early"}:
            completed_games.append(game)

    completed_games = sorted(completed_games, key=lambda item: (str(item.get("officialDate") or ""), int(item.get("gamePk") or 0)))

    raw_rows: list[dict[str, object]] = []
    uncached_fetches = 0
    for idx, game in enumerate(completed_games, start=1):
        game_pk = int(game.get("gamePk"))
        cache_path = RAW_ROOT / f"season={int(season)}" / "games" / f"{int(game_pk)}.json"
        was_cached = cache_path.exists() and not bool(args.refresh_source)
        feed = fetch_game_feed(
            season=season,
            game_pk=game_pk,
            refresh_source=bool(args.refresh_source),
            timeout_seconds=float(args.timeout_seconds),
            retries=int(args.retries),
        )
        raw_rows.extend(build_game_rows(feed, teams))
        if not was_cached:
            uncached_fetches += 1
            if idx < len(completed_games) and float(args.sleep_seconds) > 0:
                time.sleep(float(args.sleep_seconds))

    fetched_at_utc = utc_now_iso()
    market_history = load_market_history()
    player_frames, summary = build_processed_frames(
        raw_rows,
        fetched_at_utc,
        market_history,
        min_rows=int(args.min_rows),
    )
    written = write_processed_files(player_frames, season, args.player_limit)
    removed_dirs = prune_stale_player_dirs(set(written.keys()))

    manifest = {
        "season": int(season),
        "start_date_requested": start_date,
        "through_date_requested": through_date,
        "source_refresh": bool(args.refresh_source),
        "schedule_games": int(len(games)),
        "completed_games": int(len(completed_games)),
        "uncached_game_fetches": int(uncached_fetches),
        "processed_summary": summary,
        "players_written": int(len(written)),
        "player_dirs_removed": int(removed_dirs),
        "written": written,
        "market_history_rows": int(len(market_history)),
        "updated_at_utc": fetched_at_utc,
    }
    manifest_path = PROC_ROOT / f"update_manifest_{int(season)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MLB DATA UPDATE COMPLETE")
    print("=" * 80)
    print(f"Season: {season}")
    print(f"Completed games processed: {len(completed_games)}")
    print(f"Players written:          {len(written)}")
    print(f"Processed max date:       {summary['max_date']}")
    print(f"Market history rows:      {len(market_history)}")
    print(f"Manifest:                 {manifest_path}")


if __name__ == "__main__":
    main()

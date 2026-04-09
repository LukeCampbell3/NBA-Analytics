from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io_utils import ensure_dir, write_csv, write_json


MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default=default)))


def _parse_batting_order(value: Any) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        return 9
    # MLB boxscore usually encodes order as "100", "200", ...
    if text.isdigit():
        as_int = int(text)
        if as_int >= 100:
            return max(1, min(9, as_int // 100))
        return max(1, min(9, as_int))
    match = re.search(r"\d+", text)
    if not match:
        return 9
    return max(1, min(9, int(match.group(0))))


def innings_pitched_to_float(value: Any) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        return 0.0
    if "." not in text:
        return _num(text, default=0.0)
    whole, frac = text.split(".", 1)
    outs = 0
    if frac == "1":
        outs = 1
    elif frac == "2":
        outs = 2
    return _num(whole, 0.0) + (outs / 3.0)


def _extract_weather(info_block: list[dict] | None) -> tuple[float, float]:
    temp_f = 0.0
    wind_out_mph = 0.0
    for item in info_block or []:
        label = str(item.get("label", "")).strip().lower()
        value = str(item.get("value", "")).strip()
        if label != "weather" or not value:
            continue
        temp_match = re.search(r"(-?\d+)\s*degrees", value, flags=re.IGNORECASE)
        if temp_match:
            temp_f = _num(temp_match.group(1), 0.0)
        # "Wind 8 mph, Out To CF"
        mph_match = re.search(r"(\d+)\s*mph", value, flags=re.IGNORECASE)
        direction_out = bool(re.search(r"\bout\b", value, flags=re.IGNORECASE))
        if mph_match and direction_out:
            wind_out_mph = _num(mph_match.group(1), 0.0)
    return temp_f, wind_out_mph


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
    max_retries: int = 4,
) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(2.0 * attempt, 8.0))


@dataclass
class CollectorConfig:
    start_date: str
    end_date: str
    game_type: str = "R"  # regular season


def collect_schedule(session: requests.Session, config: CollectorConfig) -> list[dict]:
    schedule_url = f"{MLB_STATS_API_BASE}/schedule"
    payload = _request_json(
        session,
        schedule_url,
        params={
            "sportId": 1,
            "startDate": config.start_date,
            "endDate": config.end_date,
            "gameType": config.game_type,
        },
    )
    out: list[dict] = []
    for day in payload.get("dates", []) or []:
        official_date = str(day.get("date", "")).strip()
        for game in day.get("games", []) or []:
            status = str(game.get("status", {}).get("codedGameState", "")).strip()
            # Final / game over states only.
            if status not in {"F", "O", "R"}:
                continue
            teams = game.get("teams", {}) or {}
            home_team = teams.get("home", {}).get("team", {}) or {}
            away_team = teams.get("away", {}).get("team", {}) or {}
            out.append(
                {
                    "Date": official_date,
                    "Season": _int(game.get("season"), default=0),
                    "Game_ID": f"{game.get('gamePk', '')}",
                    "GamePk": _int(game.get("gamePk"), default=0),
                    "Home_Team_ID": _int(home_team.get("id"), default=0),
                    "Away_Team_ID": _int(away_team.get("id"), default=0),
                    "Home_Team": str(home_team.get("abbreviation") or home_team.get("name") or ""),
                    "Away_Team": str(away_team.get("abbreviation") or away_team.get("name") or ""),
                    "Venue": str(game.get("venue", {}).get("name", "") or ""),
                }
            )
    return out


def _parse_game_players(boxscore: dict, game_meta: dict) -> tuple[list[dict], list[dict]]:
    teams = boxscore.get("teams", {}) or {}
    info_block = boxscore.get("info", []) or []
    temp_f, wind_out_mph = _extract_weather(info_block)
    hitter_rows: list[dict] = []
    pitcher_rows: list[dict] = []

    for side in ["home", "away"]:
        team_block = teams.get(side, {}) or {}
        opponent_side = "away" if side == "home" else "home"
        opponent_block = teams.get(opponent_side, {}) or {}

        team = team_block.get("team", {}) or {}
        opponent = opponent_block.get("team", {}) or {}

        team_id = _int(team.get("id"), default=0)
        team_name = str(team.get("abbreviation") or team.get("name") or "")
        opp_id = _int(opponent.get("id"), default=0)
        opp_name = str(opponent.get("abbreviation") or opponent.get("name") or "")
        is_home = 1 if side == "home" else 0

        players = team_block.get("players", {}) or {}
        for _, player_payload in players.items():
            person = player_payload.get("person", {}) or {}
            player_name = str(person.get("fullName", "")).strip()
            player_id = str(person.get("id", "")).strip()
            position = str((player_payload.get("position", {}) or {}).get("abbreviation", "")).strip()
            batting_order = _parse_batting_order(player_payload.get("battingOrder"))
            stats_block = player_payload.get("stats", {}) or {}
            batting = stats_block.get("batting", {}) or {}
            pitching = stats_block.get("pitching", {}) or {}

            common = {
                "Date": game_meta["Date"],
                "Season": game_meta["Season"],
                "Game_ID": game_meta["Game_ID"],
                "Player": player_name,
                "Player_ID": player_id,
                "Team": team_name,
                "Team_ID": team_id,
                "Opponent": opp_name,
                "Opponent_ID": opp_id,
                "Is_Home": is_home,
                "Position": position,
                "Temp_F": temp_f,
                "Wind_Out_MPH": wind_out_mph,
            }

            if batting:
                at_bats = _int(batting.get("atBats"), 0)
                hits = _int(batting.get("hits"), 0)
                doubles = _int(batting.get("doubles"), 0)
                triples = _int(batting.get("triples"), 0)
                home_runs = _int(batting.get("homeRuns"), 0)
                singles = max(0, hits - doubles - triples - home_runs)
                walks = _int(batting.get("baseOnBalls"), 0)
                ibb = _int(batting.get("intentionalWalks"), 0)
                hbp = _int(batting.get("hitByPitch"), 0)
                sac_flies = _int(batting.get("sacFlies"), 0)
                sac_bunts = _int(batting.get("sacBunts"), 0)
                strikeouts = _int(batting.get("strikeOuts"), 0)
                stolen_bases = _int(batting.get("stolenBases"), 0)
                plate_appearances = _int(
                    batting.get("plateAppearances"),
                    at_bats + walks + hbp + sac_flies + sac_bunts,
                )
                total_bases = _int(
                    batting.get("totalBases"),
                    singles + (2 * doubles) + (3 * triples) + (4 * home_runs),
                )
                if plate_appearances > 0 or at_bats > 0:
                    hitter_rows.append(
                        {
                            **common,
                            "Player_Type": "hitter",
                            "Batting_Order": batting_order,
                            "R": _int(batting.get("runs"), 0),
                            "H": hits,
                            "HR": home_runs,
                            "RBI": _int(batting.get("rbi"), 0),
                            "TB": total_bases,
                            "PA": plate_appearances,
                            "AB": at_bats,
                            "BB": walks,
                            "SO": strikeouts,
                            "SB": stolen_bases,
                            "2B": doubles,
                            "3B": triples,
                            "HBP": hbp,
                            "IBB": ibb,
                            "SF": sac_flies,
                            "SH": sac_bunts,
                        }
                    )

            if pitching:
                ip = innings_pitched_to_float(pitching.get("inningsPitched"))
                batters_faced = _int(pitching.get("battersFaced"), 0)
                if ip > 0 or batters_faced > 0:
                    pitcher_rows.append(
                        {
                            **common,
                            "Player_Type": "pitcher",
                            "K": _int(pitching.get("strikeOuts"), 0),
                            "ER": _int(pitching.get("earnedRuns"), 0),
                            "ERA": _num(pitching.get("era"), 0.0),
                            "IP": ip,
                            "BF": batters_faced,
                            "Pitches": _int(pitching.get("numberOfPitches"), 0),
                            "BB_allowed": _int(pitching.get("baseOnBalls"), 0),
                            "H_allowed": _int(pitching.get("hits"), 0),
                            "HR_allowed": _int(pitching.get("homeRuns"), 0),
                            "HBP_allowed": _int(pitching.get("hitByPitch"), 0),
                            "Is_Starter": 1 if _int(pitching.get("gamesStarted"), 0) > 0 else 0,
                        }
                    )
    return hitter_rows, pitcher_rows


def collect_raw_mlb_data(config: CollectorConfig, out_dir: Path) -> dict:
    out_dir = ensure_dir(out_dir)
    raw_games_path = out_dir / "games.csv"
    raw_hitters_path = out_dir / "hitter_game_logs.csv"
    raw_pitchers_path = out_dir / "pitcher_game_logs.csv"

    session = requests.Session()
    schedule_rows = collect_schedule(session, config=config)
    if not schedule_rows:
        raise RuntimeError(
            f"No completed games found for range {config.start_date} to {config.end_date} (gameType={config.game_type})."
        )
    games_df = pd.DataFrame.from_records(schedule_rows).drop_duplicates(subset=["GamePk"]).sort_values(["Date", "GamePk"])

    hitter_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    failures: list[dict] = []

    for game in games_df.itertuples(index=False):
        game_pk = int(getattr(game, "GamePk"))
        boxscore_url = f"{MLB_STATS_API_BASE}/game/{game_pk}/boxscore"
        try:
            box = _request_json(session, boxscore_url)
            h_rows, p_rows = _parse_game_players(box, game._asdict())
            hitter_rows.extend(h_rows)
            pitcher_rows.extend(p_rows)
        except Exception as exc:  # pragma: no cover - network/runtime variability
            failures.append({"GamePk": game_pk, "error": str(exc)})

    hitters_df = pd.DataFrame.from_records(hitter_rows)
    pitchers_df = pd.DataFrame.from_records(pitcher_rows)

    if hitters_df.empty and pitchers_df.empty:
        raise RuntimeError("Collector produced no player rows.")

    if not hitters_df.empty:
        hitters_df = hitters_df.sort_values(["Date", "Game_ID", "Team_ID", "Player"]).reset_index(drop=True)
    if not pitchers_df.empty:
        pitchers_df = pitchers_df.sort_values(["Date", "Game_ID", "Team_ID", "Player"]).reset_index(drop=True)

    write_csv(games_df.drop(columns=["GamePk"]), raw_games_path)
    write_csv(hitters_df, raw_hitters_path)
    write_csv(pitchers_df, raw_pitchers_path)

    manifest = {
        "sport": "mlb",
        "range": {"start_date": config.start_date, "end_date": config.end_date, "game_type": config.game_type},
        "games_collected": int(len(games_df)),
        "hitter_rows": int(len(hitters_df)),
        "pitcher_rows": int(len(pitchers_df)),
        "failures": failures,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "games": str(raw_games_path),
            "hitters": str(raw_hitters_path),
            "pitchers": str(raw_pitchers_path),
        },
    }
    write_json(out_dir / "collection_manifest.json", manifest)
    return manifest


def default_collection_window(days_back: int = 30) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=max(1, int(days_back)))
    return start.isoformat(), end.isoformat()


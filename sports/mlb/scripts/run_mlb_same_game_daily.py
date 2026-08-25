#!/usr/bin/env python3
"""Daily MLB same-game combo run: real today's schedule + real probable
starters -> real leakage-safe team/pitcher/bullpen history -> real joint
Monte Carlo game simulation -> real live team-market odds -> real
same-game combo selection -> a durable JSON payload, mirroring how PGA
golf's run_pga_daily_predictions.py runs and publishes its own board.

Deliberately writes to its OWN payload file (same_game_predictions.json)
rather than touching sports/mlb/web/data/daily_predictions.json -- the
existing live single-leg production board and its thresholds are
untouched by this entirely new, additive same-game pipeline, per the
explicit standing constraint on this session's MLB work.

WHAT THIS SCRIPT DOES NOT DO: fabricate a game, a starter, or a price
when the real data isn't there yet. No real games scheduled today, no
real probable pitcher posted yet, or no real market currently priced
each publish an honest per-game status -- never a guessed one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT / "sports" / "mlb" / "predictions",
    REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers",
    REPO_ROOT / "sports" / "mlb" / "parlay_v2",
    REPO_ROOT / "sports" / "mlb" / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import game_simulation_model as sim  # noqa: E402
import pitcher_bullpen_model as pitching  # noqa: E402
import pitching_enriched_win_model as enriched_model  # noqa: E402
import select_mlb_same_game_bets as select  # noqa: E402
import team_win_model as base_model  # noqa: E402
from backtest_pitching_enriched_win_model import flatten_starts_and_bullpen, load_pitcher_rows  # noqa: E402
from backtest_team_win_model import load_games  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from fanduel_public_mlb_team_market_provider import FanduelPublicMlbTeamMarketProvider  # noqa: E402
from fetch_mlb_pitcher_game_data import _to_espn_abbreviation  # noqa: E402
from the_odds_api_mlb_team_market_provider import TheOddsApiMlbTeamMarketProvider  # noqa: E402

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
REQUEST_TIMEOUT_SECONDS = 20.0
NUM_TRIALS = 20000
MIN_GAMES_PLAYED_FOR_PREDICTION = 10

DEFAULT_TEAM_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_team_game_history.csv"
DEFAULT_PITCHER_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_pitcher_game_data.csv"
DEFAULT_CALIBRATION_LEDGER = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "calibration" / "reports" / "same_game_calibration_ledger.jsonl"
DEFAULT_PAIR_LEDGER = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "calibration" / "reports" / "same_game_pair_observations.jsonl"
DEFAULT_WEB_DATA_ROOT = REPO_ROOT / "sports" / "mlb" / "web" / "data"

# Real, VERIFIED full-team-name -> ESPN-abbreviation mapping (confirmed
# against MLB StatsAPI's own /v1/teams endpoint, then normalized through
# fetch_mlb_pitcher_game_data.py's already-verified AZ/CWS fix) -- The
# Odds API's team-market rows carry real full team names ("Atlanta
# Braves"), not abbreviations, so this is the real join key needed to
# match a live odds row to a real scheduled game.
STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def fetch_todays_schedule(run_date: date, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": run_date.isoformat(), "hydrate": "team,probablePitcher"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def extract_scheduled_games(schedule_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Real, not-yet-played games only (StatsAPI's own real "Preview"
    abstract state) -- a same-game combo is only meaningful pregame.
    Each side's real probable-starter id/name is None (never guessed)
    when StatsAPI hasn't posted one yet for that real game."""
    games: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {})
            if status.get("abstractGameState") != "Preview":
                continue
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_team_raw = (home.get("team") or {}).get("abbreviation", "")
            away_team_raw = (away.get("team") or {}).get("abbreviation", "")
            if not home_team_raw or not away_team_raw:
                continue
            home_pitcher = home.get("probablePitcher") or {}
            away_pitcher = away.get("probablePitcher") or {}
            games.append(
                {
                    "game_id": str(game.get("gamePk", "")),
                    "date": str(game.get("officialDate", "")),
                    "home_team": _to_espn_abbreviation(home_team_raw),
                    "away_team": _to_espn_abbreviation(away_team_raw),
                    "home_full_name": (home.get("team") or {}).get("name", ""),
                    "away_full_name": (away.get("team") or {}).get("name", ""),
                    "home_starter_id": home_pitcher.get("id"),
                    "home_starter_name": home_pitcher.get("fullName", ""),
                    "away_starter_id": away_pitcher.get("id"),
                    "away_starter_name": away_pitcher.get("fullName", ""),
                }
            )
    return games


def _odds_rows_for_game(all_rows: list[dict[str, Any]], game: dict[str, Any]) -> list[dict[str, Any]]:
    home_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(game["home_full_name"])
    away_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(game["away_full_name"])
    return [
        r
        for r in all_rows
        if STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(r.get("home_team", "")) == home_espn
        and STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(r.get("away_team", "")) == away_espn
        and home_espn is not None
    ]


def collect_team_market_odds_chain(
    *,
    fanduel_provider: Optional[FanduelPublicMlbTeamMarketProvider] = None,
    odds_api_provider: Optional[TheOddsApiMlbTeamMarketProvider] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Real provider UNION, not a fallback-only chain: FanDuel's real,
    free, no-auth feed (moneyline + full-game total, confirmed live
    before this was wired in -- see fanduel_public_mlb_team_market_
    provider.py's own module docstring) is the primary source, and The
    Odds API is always also consulted to add real F5-total rows FanDuel's
    public page doesn't expose (a real THE_ODDS_API_KEY simply degrades
    to `missing_credentials` and contributes nothing when unset, exactly
    as it already did -- this never blocks on it). Rows from both real
    sources are pooled; select_mlb_same_game_bets.py's own real-consensus-
    line logic already handles multiple real books quoting the same
    market."""
    fanduel = fanduel_provider if fanduel_provider is not None else FanduelPublicMlbTeamMarketProvider()
    fanduel_result = fanduel.collect_team_market_odds()
    rows: list[dict[str, Any]] = list(fanduel_result.get("odds", [])) if fanduel_result.get("status") == "success" else []

    odds_api = odds_api_provider if odds_api_provider is not None else TheOddsApiMlbTeamMarketProvider()
    odds_api_result = odds_api.collect_team_market_odds()
    if odds_api_result.get("status") == "success":
        rows.extend(odds_api_result.get("odds", []))

    sources = {"fanduel_public": fanduel_result.get("status"), "the_odds_api": odds_api_result.get("status")}
    combined_status = "success" if rows else (fanduel_result.get("status") or odds_api_result.get("status"))
    return {"status": combined_status, "sources": sources}, rows


def build_daily_payload(
    *,
    run_date: date,
    team_universe_csv: Path = DEFAULT_TEAM_UNIVERSE,
    pitcher_universe_csv: Path = DEFAULT_PITCHER_UNIVERSE,
    calibration_ledger: Optional[Path] = DEFAULT_CALIBRATION_LEDGER,
    num_trials: int = NUM_TRIALS,
    schedule_payload: Optional[dict[str, Any]] = None,
    fanduel_provider: Optional[FanduelPublicMlbTeamMarketProvider] = None,
    odds_api_provider: Optional[TheOddsApiMlbTeamMarketProvider] = None,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "status": "ok", "generated_at_utc": generated_at, "run_date": run_date.isoformat(),
        "games": [],
    }

    schedule = schedule_payload if schedule_payload is not None else fetch_todays_schedule(run_date)
    games_today = extract_scheduled_games(schedule)
    if not games_today:
        payload["status"] = "no_real_games_scheduled_today"
        return payload

    historical_games = load_games(team_universe_csv)
    pitcher_rows = load_pitcher_rows(pitcher_universe_csv)

    home_field_advantage = base_model.compute_empirical_home_field_advantage(historical_games)
    team_history = base_model.build_cumulative_team_stats(historical_games)

    pooled_innings = [
        {"starter_outs": r[f"{side}_starter_outs"], "bullpen_outs": r[f"{side}_bullpen_outs"]}
        for r in pitcher_rows for side in ("home", "away")
    ]
    starter_innings_share = pitching.compute_empirical_starter_innings_share(pooled_innings)
    all_starts, all_bullpen_appearances = flatten_starts_and_bullpen(pitcher_rows)
    pitcher_history = pitching.build_cumulative_pitcher_stats(all_starts)
    bullpen_history = pitching.build_cumulative_bullpen_stats(all_bullpen_appearances)

    runs_dispersion_ratio = sim.compute_empirical_runs_dispersion(historical_games)
    f5_share = sim.compute_empirical_f5_share(historical_games)

    odds_summary, all_odds_rows = collect_team_market_odds_chain(
        fanduel_provider=fanduel_provider, odds_api_provider=odds_api_provider
    )
    payload["odds_status"] = odds_summary["status"]
    payload["odds_sources"] = odds_summary["sources"]

    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None
    run_date_str = run_date.isoformat()

    total_authorized = 0
    for game in games_today:
        entry: dict[str, Any] = {
            "game_id": game["game_id"], "date": game["date"],
            "home_team": game["home_team"], "away_team": game["away_team"],
            "home_starter_name": game["home_starter_name"], "away_starter_name": game["away_starter_name"],
        }

        home_team_stats = base_model.stats_as_of(team_history.get(game["home_team"], []), run_date_str)
        away_team_stats = base_model.stats_as_of(team_history.get(game["away_team"], []), run_date_str)
        if (
            home_team_stats is None or away_team_stats is None
            or home_team_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
            or away_team_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
        ):
            entry["status"] = "insufficient_real_team_history"
            payload["games"].append(entry)
            continue

        home_starter_stats = pitching.stats_as_of(pitcher_history.get(game["home_starter_id"], []), run_date_str) if game["home_starter_id"] else None
        away_starter_stats = pitching.stats_as_of(pitcher_history.get(game["away_starter_id"], []), run_date_str) if game["away_starter_id"] else None
        home_bullpen_stats = pitching.stats_as_of(bullpen_history.get(game["home_team"], []), run_date_str)
        away_bullpen_stats = pitching.stats_as_of(bullpen_history.get(game["away_team"], []), run_date_str)

        sides = enriched_model.expected_runs_per_side_enriched(
            home_team_stats, away_team_stats,
            home_starter_stats=home_starter_stats, home_bullpen_stats=home_bullpen_stats,
            away_starter_stats=away_starter_stats, away_bullpen_stats=away_bullpen_stats,
            starter_innings_share=starter_innings_share,
        )
        if sides is None:
            entry["status"] = "insufficient_real_data_for_expected_runs"
            payload["games"].append(entry)
            continue
        home_expected, away_expected = sides

        result = sim.simulate_game_outcomes(
            home_expected, away_expected,
            runs_dispersion_ratio=runs_dispersion_ratio, f5_share=f5_share,
            home_field_advantage=home_field_advantage, num_trials=num_trials,
            seed=abs(hash((game["game_id"], run_date_str))) % (2**31),
        )

        game_odds_rows = _odds_rows_for_game(all_odds_rows, game)
        combos = select.build_same_game_candidates(
            game, result, game_odds_rows,
            calibration_store=calibration_store, calibration_as_of=generated_at,
        )
        top_combos = select.top_combo_candidates(combos)
        authorized_count = sum(1 for c in combos if c.candidate_authorized)
        total_authorized += authorized_count

        entry["status"] = "ok" if game_odds_rows else "no_real_odds_priced_yet"
        entry["home_win_probability"] = result.home_win_probability
        entry["combo_candidates"] = [c.as_dict() for c in top_combos]
        entry["candidate_authorized_count"] = authorized_count
        payload["games"].append(entry)

    payload["candidate_authorized_count"] = total_authorized
    return payload


def write_web_payload(payload: dict[str, Any], *, web_data_root: Path = DEFAULT_WEB_DATA_ROOT) -> Path:
    web_data_root.mkdir(parents=True, exist_ok=True)
    out_path = web_data_root / "same_game_predictions.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", type=date.fromisoformat, default=None)
    parser.add_argument("--team-universe-csv", type=Path, default=DEFAULT_TEAM_UNIVERSE)
    parser.add_argument("--pitcher-universe-csv", type=Path, default=DEFAULT_PITCHER_UNIVERSE)
    parser.add_argument("--calibration-ledger", type=Path, default=DEFAULT_CALIBRATION_LEDGER)
    parser.add_argument("--web-data-root", type=Path, default=DEFAULT_WEB_DATA_ROOT)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_date = args.run_date or date.today()
    payload = build_daily_payload(
        run_date=run_date, team_universe_csv=args.team_universe_csv, pitcher_universe_csv=args.pitcher_universe_csv,
        calibration_ledger=args.calibration_ledger, num_trials=args.num_trials,
    )
    out_path = write_web_payload(payload, web_data_root=args.web_data_root)
    print(json.dumps({"status": payload["status"], "real_games": len(payload.get("games", [])), "candidate_authorized_count": payload.get("candidate_authorized_count", 0), "written": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

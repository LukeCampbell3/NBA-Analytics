#!/usr/bin/env python3
"""Daily MLB pitcher-strikeouts-only parlay run: real today's schedule +
real probable starters -> real season-to-date strikeout projection per
starter -> real live FanDuel pitcher-strikeout odds -> real cross-game
2-leg pitcher parlay selection -> a durable JSON payload, mirroring how
run_mlb_same_game_daily.py runs and publishes its own additive board.

Deliberately writes to its OWN payload file (pitcher_parlay_predictions.
json) rather than touching daily_predictions.json or
same_game_predictions.json -- a brand-new, additive product, same
standing constraint every other new MLB product this session has kept.

WHAT THIS SCRIPT DOES NOT DO: fabricate a starter, a strikeout
projection, or a price when the real data isn't there yet. No real
games scheduled today, no real probable starter posted yet, a starter
with too few real starts this season for a real projection, or no real
FanDuel strikeout line currently posted each simply produce no leg for
that starter -- never a guessed one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT / "sports" / "mlb" / "predictions",
    REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers",
    REPO_ROOT / "sports" / "mlb" / "parlay_v2",
    REPO_ROOT / "sports" / "mlb" / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import parlay_quality_frontier as quality  # noqa: E402
import select_mlb_pitcher_parlay as select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from fanduel_public_mlb_provider import FanduelPublicMlbProvider  # noqa: E402
from run_mlb_same_game_daily import extract_scheduled_games, fetch_todays_schedule  # noqa: E402

DEFAULT_CALIBRATION_LEDGER = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "calibration" / "reports" / "pitcher_parlay_calibration_ledger.jsonl"
DEFAULT_WEB_DATA_ROOT = REPO_ROOT / "sports" / "mlb" / "web" / "data"
DEFAULT_OUTPUT_FILENAME = "pitcher_parlay_predictions.json"


def build_starters(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Real probable starters for today's real, not-yet-played games --
    one row per side, home and away, skipping a side StatsAPI hasn't
    posted a real probable pitcher for yet."""
    starters: list[dict[str, Any]] = []
    for game in games:
        if game.get("home_starter_id") and game.get("home_starter_name"):
            starters.append(
                {
                    "pitcher_id": game["home_starter_id"], "pitcher_name": game["home_starter_name"],
                    "team": game["home_team"], "opponent": game["away_team"], "game_id": game["game_id"],
                }
            )
        if game.get("away_starter_id") and game.get("away_starter_name"):
            starters.append(
                {
                    "pitcher_id": game["away_starter_id"], "pitcher_name": game["away_starter_name"],
                    "team": game["away_team"], "opponent": game["home_team"], "game_id": game["game_id"],
                }
            )
    return starters


def build_daily_payload(
    *,
    run_date: date,
    calibration_ledger: Optional[Path] = DEFAULT_CALIBRATION_LEDGER,
    schedule_payload: Optional[dict[str, Any]] = None,
    fanduel_provider: Optional[FanduelPublicMlbProvider] = None,
    fetch_season_stats=select.k_model.fetch_pitcher_season_stats,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "status": "ok", "generated_at_utc": generated_at, "run_date": run_date.isoformat(),
        "model": "mlb_pitcher_k_parlay_v1",
        "selection_policy": "pitcher_alt_line_value_frontier_v1",
    }

    schedule = schedule_payload if schedule_payload is not None else fetch_todays_schedule(run_date)
    games = extract_scheduled_games(schedule)
    if not games:
        payload["status"] = "no_real_games_scheduled_today"
        return payload

    starters = build_starters(games)
    payload["real_starters_posted"] = len(starters)
    if not starters:
        payload["status"] = "no_real_probable_starters_posted_yet"
        return payload

    provider = fanduel_provider or FanduelPublicMlbProvider()
    odds_result = provider.collect_player_props()
    payload["odds_status"] = odds_result.get("status")
    odds_rows = odds_result.get("odds", []) if odds_result.get("status") == "success" else []

    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None
    # Keep every real FanDuel strikeout rung.  The legacy builder reduces a
    # pitcher to one consensus line before selection, which is appropriate for
    # a canonical market view but prevents the parlay from asking whether a
    # higher alt-over line can keep the hit-rate floor while materially
    # improving price/EV.
    legs = quality.build_pitcher_alt_line_legs(
        starters, odds_rows, season=run_date.year,
        calibration_store=calibration_store, calibration_as_of=generated_at,
        fetch_season_stats=fetch_season_stats,
    )
    payload["legs"] = [leg.as_dict() for leg in legs]
    payload["real_priced_legs"] = sum(1 for leg in legs if leg.price_confirmed)
    payload["real_priced_pitchers"] = len({leg.pitcher_id for leg in legs if leg.price_confirmed})

    frontier = quality.select_pitcher_value_frontier(legs)
    payload["quality_frontier"] = frontier.diagnostics()
    combo = frontier.candidate
    if combo is None:
        payload["parlay"] = None
        payload["parlay_status"] = "no_probability_safe_pair_from_two_distinct_priced_starters"
        return payload

    parlay = combo.as_dict()
    parlay["selection_mode"] = frontier.selection_mode
    parlay["quality_frontier"] = frontier.diagnostics()
    parlay["price_efficient"] = frontier.selection_mode == "frontier_value"
    parlay["economic_decision"] = (
        "positive_ev_probability_safe_shadow"
        if frontier.selection_mode == "frontier_value"
        else "no_bet_price_fail_probability_research_only"
    )
    payload["parlay"] = parlay
    # Preserve the long-standing ready/not-ready payload contract for the
    # frontend; selection_mode/economic_decision carry the richer distinction.
    payload["parlay_status"] = "ready"
    return payload


def write_web_payload(payload: dict[str, Any], *, web_data_root: Path, filename: str = DEFAULT_OUTPUT_FILENAME) -> Path:
    web_data_root.mkdir(parents=True, exist_ok=True)
    out_path = web_data_root / filename
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", type=date.fromisoformat, default=None)
    parser.add_argument("--calibration-ledger", type=Path, default=DEFAULT_CALIBRATION_LEDGER)
    parser.add_argument("--web-data-root", type=Path, default=DEFAULT_WEB_DATA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_date = args.run_date or date.today()
    payload = build_daily_payload(run_date=run_date, calibration_ledger=args.calibration_ledger)
    out_path = write_web_payload(payload, web_data_root=args.web_data_root)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "real_starters_posted": payload.get("real_starters_posted", 0),
                "real_priced_legs": payload.get("real_priced_legs", 0),
                "parlay_status": payload.get("parlay_status"),
                "selection_mode": (payload.get("quality_frontier") or {}).get("selection_mode"),
                "written": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

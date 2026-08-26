#!/usr/bin/env python3
"""Real single-leg MLB team-market predictions (moneyline, full-game
total, first-5-innings total) for the MAIN single-leg board.

This is the newest MLB predictor -- real joint Monte Carlo game
simulation (game_simulation_model.py) driven by real starting-pitcher/
bullpen enrichment (pitching_enriched_win_model.py) -- wired into the
live single-leg board (sports/mlb/web/data/daily_predictions.json)
instead of staying siloed in the separate, additive same-game combo
product (run_mlb_same_game_daily.py / same_game_predictions.json).

Reuses run_mlb_same_game_daily.prepare_todays_game_simulations exactly
(same real schedule fetch, same real historical stats, same real
simulation, same real team-market odds) so both pipelines see identical
real data -- this never re-derives its own copy. It also reuses
select_mlb_same_game_bets.build_single_leg_team_market_candidates, the
exact same real legs (and the exact same calibration/support.py REQUIRED
gate) that pipeline's combos are built from.

Gating is IDENTICAL to every other board in this repo: a leg is only
`leg_authorized=True` once it clears calibration/support.py's real,
>=20-prior-settled-observations REQUIRED gate. This reuses the SAME
calibration ledger the same-game combo pipeline writes to
(same_game_calibration_ledger.jsonl) -- a market/line/state bucket's
real evidence means the same thing regardless of whether a leg is shown
standalone or as part of a same-game combo. As of this pipeline's first
runs that ledger has no real settled observations for these buckets yet,
so every leg starts leg_authorized=False: an honest zero, not a guessed
one, exactly the posture every other board in this repo started from.

Only leg_authorized=True legs are ever written to the merged
daily_predictions.json output -- never an unauthorized suggestion
presented as a live pick.
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

import run_mlb_same_game_daily as same_game  # noqa: E402
import select_mlb_same_game_bets as select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402

MODEL_VERSION = "mlb_team_market_joint_sim_v1"

DEFAULT_CALIBRATION_LEDGER = same_game.DEFAULT_CALIBRATION_LEDGER
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "team_market_predictions.json"
DEFAULT_DAILY_PREDICTIONS_PATH = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"


def build_team_market_predictions(
    *,
    run_date: date,
    team_universe_csv: Path = same_game.DEFAULT_TEAM_UNIVERSE,
    pitcher_universe_csv: Path = same_game.DEFAULT_PITCHER_UNIVERSE,
    calibration_ledger: Optional[Path] = DEFAULT_CALIBRATION_LEDGER,
    num_trials: int = same_game.NUM_TRIALS,
    schedule_payload: Optional[dict[str, Any]] = None,
    fanduel_provider=None,
    odds_api_provider=None,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "status": "ok", "generated_at_utc": generated_at, "run_date": run_date.isoformat(),
        "model": MODEL_VERSION, "picks": [],
    }

    top_status, odds_summary, prepared_games = same_game.prepare_todays_game_simulations(
        run_date=run_date, team_universe_csv=team_universe_csv, pitcher_universe_csv=pitcher_universe_csv,
        num_trials=num_trials, schedule_payload=schedule_payload,
        fanduel_provider=fanduel_provider, odds_api_provider=odds_api_provider,
    )
    if top_status != "ok":
        payload["status"] = top_status
        return payload

    payload["odds_status"] = odds_summary["status"]
    payload["odds_sources"] = odds_summary["sources"]

    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None
    for prepared in prepared_games:
        if prepared.result is None:
            continue
        legs = select.build_single_leg_team_market_candidates(
            prepared.game, prepared.result, prepared.game_odds_rows,
            calibration_store=calibration_store, calibration_as_of=generated_at,
        )
        for leg in legs:
            if not leg.leg_authorized:
                continue
            payload["picks"].append(
                {
                    "sport": "mlb",
                    "market_type": MODEL_VERSION,
                    "game_id": prepared.game["game_id"],
                    "date": prepared.game["date"],
                    "home_team": prepared.game["home_team"],
                    "away_team": prepared.game["away_team"],
                    **leg.as_dict(),
                }
            )

    payload["authorized_pick_count"] = len(payload["picks"])
    return payload


def merge_into_daily_predictions(team_market_payload: dict[str, Any], *, daily_predictions_path: Path) -> dict[str, Any]:
    """Additive merge into the main board's already-exported
    daily_predictions.json: writes only the "mlb_team_market_plays" (and
    two small status/timestamp) keys, never touches "plays" or any other
    existing key -- nothing that already reads this file's player-prop
    shape is affected, whether or not this stage has ever run before."""
    if daily_predictions_path.exists():
        published = json.loads(daily_predictions_path.read_text(encoding="utf-8"))
    else:
        published = {}
    published["mlb_team_market_plays"] = team_market_payload.get("picks", [])
    published["mlb_team_market_status"] = team_market_payload.get("status")
    published["mlb_team_market_generated_at_utc"] = team_market_payload.get("generated_at_utc")
    daily_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    daily_predictions_path.write_text(json.dumps(published, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return published


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", type=date.fromisoformat, default=None)
    parser.add_argument("--team-universe-csv", type=Path, default=same_game.DEFAULT_TEAM_UNIVERSE)
    parser.add_argument("--pitcher-universe-csv", type=Path, default=same_game.DEFAULT_PITCHER_UNIVERSE)
    parser.add_argument("--calibration-ledger", type=Path, default=DEFAULT_CALIBRATION_LEDGER)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--daily-predictions-path", type=Path, default=None, action="append",
        help=(
            "Merge authorized picks into this main-board "
            "daily_predictions.json (additive -- 'mlb_team_market_plays' "
            "key only). Repeatable -- pass once per real published copy "
            "(e.g. sports/mlb/web/data/daily_predictions.json AND the "
            "paywalled private-content copy) so the one real simulation "
            "run backs every copy instead of re-simulating per copy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_date = args.run_date or date.today()
    payload = build_team_market_predictions(
        run_date=run_date, team_universe_csv=args.team_universe_csv, pitcher_universe_csv=args.pitcher_universe_csv,
        calibration_ledger=args.calibration_ledger,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    merge_targets = args.daily_predictions_path or []
    for daily_predictions_path in merge_targets:
        merge_into_daily_predictions(payload, daily_predictions_path=daily_predictions_path)

    print(
        json.dumps(
            {
                "status": payload["status"],
                "authorized_pick_count": payload.get("authorized_pick_count", 0),
                "written": str(args.output_path),
                "merged_into": [str(p) for p in merge_targets],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quality-gated daily MLB same-game shadow publication.

Reuses the existing real schedule/history/joint Monte Carlo/odds preparation
and same-game candidate builder unchanged. Publication is now deliberately
strict: >=50% model joint probability, >=3 percentage points edge versus the
no-vig market joint, and >=5% synthetic-price EV before a candidate can become
the headline. Low-quality positive-EV candidates remain diagnostics only.
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

import select_mlb_same_game_bets as select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from same_game_quality_selector import (  # noqa: E402
    MIN_HEADLINE_EXPECTED_VALUE,
    MIN_HEADLINE_PROBABILITY_EDGE,
    exploratory_ev_candidates,
    quality_safe_candidates,
)
from run_mlb_same_game_daily import (  # noqa: E402
    DEFAULT_CALIBRATION_LEDGER,
    DEFAULT_PITCHER_UNIVERSE,
    DEFAULT_TEAM_UNIVERSE,
    DEFAULT_WEB_DATA_ROOT,
    NUM_TRIALS,
    prepare_todays_game_simulations,
    write_web_payload,
)
from fanduel_public_mlb_team_market_provider import FanduelPublicMlbTeamMarketProvider  # noqa: E402
from the_odds_api_mlb_team_market_provider import TheOddsApiMlbTeamMarketProvider  # noqa: E402


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
        "status": "ok",
        "generated_at_utc": generated_at,
        "run_date": run_date.isoformat(),
        "games": [],
        "selection_policy": {
            "name": "same_game_tight_quality_v3_shadow",
            "joint_probability_floor": select.MIN_COMBO_JOINT_PROBABILITY,
            "minimum_probability_edge_vs_no_vig": MIN_HEADLINE_PROBABILITY_EDGE,
            "minimum_synthetic_price_ev": MIN_HEADLINE_EXPECTED_VALUE,
            "ranking_after_gates": "expected_value_per_unit_desc_then_joint_probability_desc",
            "actual_combined_sgp_quote_captured": False,
            "authority": "shadow_only",
            "exploratory_rejected_candidates_are_headline_eligible": False,
        },
    }

    top_status, odds_summary, prepared_games = prepare_todays_game_simulations(
        run_date=run_date,
        team_universe_csv=team_universe_csv,
        pitcher_universe_csv=pitcher_universe_csv,
        num_trials=num_trials,
        schedule_payload=schedule_payload,
        fanduel_provider=fanduel_provider,
        odds_api_provider=odds_api_provider,
    )
    if top_status != "ok":
        payload["status"] = top_status
        return payload

    payload["odds_status"] = odds_summary["status"]
    payload["odds_sources"] = odds_summary["sources"]
    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None

    total_authorized = 0
    total_quality_safe = 0
    total_exploratory = 0
    for prepared in prepared_games:
        entry = dict(prepared.entry)
        if prepared.result is None:
            payload["games"].append(entry)
            continue

        combos = select.build_same_game_candidates(
            prepared.game,
            prepared.result,
            prepared.game_odds_rows,
            calibration_store=calibration_store,
            calibration_as_of=generated_at,
        )
        safe = quality_safe_candidates(combos)
        exploratory = exploratory_ev_candidates(combos)

        authorized_count = sum(1 for combo in combos if combo.candidate_authorized)
        total_authorized += authorized_count
        total_quality_safe += len(safe)
        total_exploratory += len(exploratory)

        entry["combo_candidates"] = [combo.as_dict() for combo in safe]
        entry["exploratory_ev_candidates"] = [combo.as_dict() for combo in exploratory]
        entry["candidate_authorized_count"] = authorized_count
        entry["quality_safe_candidate_count"] = len(safe)
        entry["exploratory_candidate_count"] = len(exploratory)
        payload["games"].append(entry)

    payload["candidate_authorized_count"] = total_authorized
    payload["quality_safe_candidate_count"] = total_quality_safe
    payload["exploratory_candidate_count"] = total_exploratory
    payload["headline_status"] = "ready" if total_quality_safe else "abstain_no_tight_quality_combo"
    return payload


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
        run_date=run_date,
        team_universe_csv=args.team_universe_csv,
        pitcher_universe_csv=args.pitcher_universe_csv,
        calibration_ledger=args.calibration_ledger,
        num_trials=args.num_trials,
    )
    out_path = write_web_payload(payload, web_data_root=args.web_data_root)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "headline_status": payload.get("headline_status"),
                "quality_safe_candidates": payload.get("quality_safe_candidate_count", 0),
                "exploratory_candidates": payload.get("exploratory_candidate_count", 0),
                "written": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

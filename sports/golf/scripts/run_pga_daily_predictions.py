#!/usr/bin/env python3
"""Daily PGA Tour prediction run: real field/scores -> real recent-form
model -> real Monte Carlo outcome probabilities -> real market odds ->
real single-leg candidate selection -> a durable JSON payload for the
frontend and the calibration ledger, mirroring how every other sport in
this repo (MLB/NFL/NBA/F1) runs and publishes its daily board.

WHAT THIS SCRIPT DOES NOT DO: fabricate a field, a score, or a price when
the real data isn't there yet. A tournament with no field posted, or no
real market currently priced, publishes an honest "not available" state
-- never a guessed one. This mirrors NFL/MLB's own eligibility-gated
abstain posture exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT,
    REPO_ROOT / "sports" / "golf" / "scripts",
    REPO_ROOT / "sports" / "golf" / "predictions",
    REPO_ROOT / "sports" / "golf" / "parlay_v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fetch_pga_event as fetcher  # noqa: E402
import score_model as model  # noqa: E402
from odds_provider import TheOddsApiGolfProvider  # noqa: E402
import select_pga_bets as select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402

DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "golf" / "data" / "raw" / "espn"
DEFAULT_WEB_DATA_ROOT = REPO_ROOT / "sports" / "golf" / "web" / "data"
DEFAULT_CALIBRATION_LEDGER = REPO_ROOT / "sports" / "golf" / "parlay_v2" / "calibration" / "reports" / "calibration_ledger.jsonl"
RECENT_FORM_EVENT_LOOKBACK = 6
CUT_EVENT_MIN_FIELD_SIZE = 100  # real signal that a real cut applies (majors/full-field events); small playoff fields (<=78) never have a real cut


def has_real_cut(field_size: int) -> bool:
    return field_size > CUT_EVENT_MIN_FIELD_SIZE


def build_daily_payload(*, raw_root: Path, calibration_ledger: Path, num_simulations: int = 20000) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    calendar = fetcher.fetch_season_calendar()
    current_event = fetcher.resolve_current_or_next_event(calendar)
    if current_event is None:
        return {"status": "no_event_in_calendar", "generated_at_utc": generated_at, "top_10": [], "candidates": []}

    leaderboard = fetcher.fetch_event_leaderboard(current_event.event_id)
    players = [fetcher.extract_player_rounds(row) for row in leaderboard.pop("competitors", [])]
    leaderboard["players"] = players
    fetcher.persist_event_snapshot(current_event.event_id, leaderboard, raw_root=raw_root)

    payload: dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": generated_at,
        "event_id": current_event.event_id,
        "event_name": current_event.name,
        "event_start_utc": current_event.start_date,
        "event_end_utc": current_event.end_date,
        "field_status": leaderboard.get("status", "UNKNOWN"),
        "field_size": len(players),
        "top_10": [],
        "candidates": [],
    }
    if not players:
        payload["status"] = "field_not_posted"
        return payload

    recent_events_meta = fetcher.recent_completed_events(calendar, limit=RECENT_FORM_EVENT_LOOKBACK)
    recent_events: list[dict[str, Any]] = []
    for event in recent_events_meta:
        recent_leaderboard = fetcher.fetch_event_leaderboard(event.event_id)
        recent_leaderboard["players"] = [fetcher.extract_player_rounds(row) for row in recent_leaderboard.pop("competitors", [])]
        fetcher.persist_event_snapshot(event.event_id, recent_leaderboard, raw_root=raw_root)
        recent_events.append(recent_leaderboard)

    forms = model.build_recent_form(recent_events)
    scheduled_rounds = 4
    field_players = [{"player_id": p["player_id"], "player_name": p["player_name"], "headshot_url": p["headshot_url"]} for p in players]
    projections = model.project_field(field_players, forms, scheduled_rounds=scheduled_rounds)
    has_cut = has_real_cut(len(players))
    outcome_probabilities = model.simulate_tournament(
        projections, scheduled_rounds=scheduled_rounds, has_cut=has_cut, num_simulations=num_simulations
    )

    # Rank by projected total score (lower/better), not win probability --
    # "top 10 projected golfers" per the product spec is about the
    # projection itself, not a derived probability.
    ranked_projections = sorted(projections, key=lambda p: p.projected_total_score)[:10]
    outcomes_by_id = {r.player_id: r for r in outcome_probabilities}
    payload["top_10"] = [
        {
            "player_id": proj.player_id,
            "player_name": proj.player_name,
            "headshot_url": proj.headshot_url,
            "projected_round_score": round(proj.projected_round_score, 2),
            "projected_total_score": round(proj.projected_total_score, 2),
            "form_rounds_observed": proj.form_rounds_observed,
            "win_probability": outcomes_by_id[proj.player_id].win_probability if proj.player_id in outcomes_by_id else None,
            "top10_probability": outcomes_by_id[proj.player_id].top10_probability if proj.player_id in outcomes_by_id else None,
        }
        for proj in ranked_projections
    ]

    odds_provider = TheOddsApiGolfProvider()
    odds_result = odds_provider.collect_odds()
    payload["odds_status"] = odds_result.get("status")
    odds_rows = odds_result.get("odds", []) if odds_result.get("status") == "success" else []

    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None
    candidates = select.build_candidates(
        outcome_probabilities, odds_rows, event_id=current_event.event_id,
        calibration_store=calibration_store, calibration_as_of=generated_at,
    )
    top_candidates = select.top_candidates_per_market(candidates)
    payload["candidates"] = [c.as_dict() for c in top_candidates]
    payload["candidate_authorized_count"] = sum(1 for c in candidates if c.candidate_authorized)
    return payload


def write_web_payload(payload: dict[str, Any], *, web_data_root: Path) -> Path:
    web_data_root.mkdir(parents=True, exist_ok=True)
    out_path = web_data_root / "daily_predictions.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--web-data-root", type=Path, default=DEFAULT_WEB_DATA_ROOT)
    parser.add_argument("--calibration-ledger", type=Path, default=DEFAULT_CALIBRATION_LEDGER)
    parser.add_argument("--num-simulations", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_daily_payload(raw_root=args.raw_root, calibration_ledger=args.calibration_ledger, num_simulations=args.num_simulations)
    out_path = write_web_payload(payload, web_data_root=args.web_data_root)
    print(json.dumps({"status": payload["status"], "event_id": payload.get("event_id"), "field_size": payload.get("field_size", 0), "written": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

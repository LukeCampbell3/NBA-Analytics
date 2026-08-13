#!/usr/bin/env python3
"""Train the F1 model, collect winner odds, and publish the current board."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_PATH = Path(__file__).resolve()
F1_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = F1_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sports.f1.predictions.data_source import (  # noqa: E402
    JsonClient,
    entries_from_latest_race,
    fetch_driver_standings,
    fetch_history,
    fetch_schedule,
    fetch_starting_grid,
    select_next_event,
)
from sports.f1.predictions.model import predict_event, train_and_evaluate  # noqa: E402
from sports.f1.predictions.odds import attach_consensus_market, fetch_available_odds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the daily Formula 1 research board")
    parser.add_argument("--run-date", help="Board date in YYYY-MM-DD (defaults to UTC today)")
    parser.add_argument("--history-start", type=int, default=2022)
    parser.add_argument("--output", type=Path, default=F1_ROOT / "web/data/daily_predictions.json")
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--history-input", type=Path, help="Normalized history fixture for deterministic replay")
    parser.add_argument("--schedule-input", type=Path, help="Jolpica schedule fixture")
    parser.add_argument("--standings-input", type=Path, help="Normalized current entries fixture")
    parser.add_argument("--odds-input", type=Path, help="Normalized odds fixture")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def grade_archived_snapshots(history: list[dict]) -> dict:
    history_dir = F1_ROOT / "web/data/history"
    results_by_event = {
        (int(race["season"]), int(race["round"])): race
        for race in history
    }
    graded: list[dict] = []
    paths = sorted(history_dir.glob("????-??-??.json")) if history_dir.is_dir() else []
    for path in paths:
        try:
            archived = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(archived, dict) or not isinstance(archived.get("event"), dict):
            continue
        event = archived["event"]
        race = results_by_event.get((int(event.get("season") or 0), int(event.get("round") or 0)))
        if not race:
            continue
        winner = next((row for row in race["results"] if int(row["finish"]) == 1), None)
        projections = archived.get("projections") if isinstance(archived.get("projections"), list) else []
        if not winner or not projections:
            continue
        winner_projection = next((row for row in projections if row.get("driver_id") == winner["driver_id"]), None)
        top_pick = max(projections, key=lambda row: float(row.get("win_probability") or 0.0))
        winner_probability = float(winner_projection.get("win_probability") or 0.0) if winner_projection else 0.0
        settlement = {
            "status": "graded",
            "winner_driver_id": winner["driver_id"],
            "winner": winner["driver"],
            "winner_probability": winner_probability,
            "winner_log_loss": -math.log(max(winner_probability, 1e-9)),
            "top_pick": top_pick.get("driver"),
            "top_pick_hit": top_pick.get("driver_id") == winner["driver_id"],
            "winner_brier": sum(
                (float(row.get("win_probability") or 0.0) - float(row.get("driver_id") == winner["driver_id"])) ** 2
                for row in projections
            ) / len(projections),
            "graded_plays": len(archived.get("plays") or []),
            "winning_plays": sum(row.get("driver_id") == winner["driver_id"] for row in archived.get("plays") or []),
        }
        if archived.get("settlement") != settlement:
            archived["settlement"] = settlement
            write_json(path, archived)
        graded.append({"event": event, **settlement})
    race_keys = {(row["event"]["season"], row["event"]["round"]) for row in graded}
    total_plays = sum(row["graded_plays"] for row in graded)
    return {
        "status": "active" if graded else "awaiting_settled_snapshots",
        "settled_snapshots": len(graded),
        "settled_races": len(race_keys),
        "top_pick_accuracy": sum(row["top_pick_hit"] for row in graded) / len(graded) if graded else None,
        "winner_log_loss": sum(row["winner_log_loss"] for row in graded) / len(graded) if graded else None,
        "winner_brier": sum(row["winner_brier"] for row in graded) / len(graded) if graded else None,
        "graded_plays": total_plays,
        "play_hit_rate": sum(row["winning_plays"] for row in graded) / total_plays if total_plays else None,
    }


def archive_payload(payload: dict) -> None:
    history_dir = F1_ROOT / "web/data/history"
    write_json(history_dir / f"{payload['run_date']}.json", payload)
    dates = sorted(path.stem for path in history_dir.glob("????-??-??.json"))
    write_json(history_dir / "index.json", {"dates": dates, "latest": dates[-1] if dates else None})


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    run_day = date.fromisoformat(args.run_date) if args.run_date else datetime.now(timezone.utc).date()
    history = (
        list(read_json(args.history_input))
        if args.history_input
        else fetch_history(
            args.history_start,
            run_day.year,
            cache_path=F1_ROOT / "data/cache/jolpica_history.json",
        )
    )
    schedule = (
        list(read_json(args.schedule_input))
        if args.schedule_input
        else fetch_schedule(run_day.year)
    )
    event = select_next_event(schedule, run_day)
    generated_at = datetime.now(timezone.utc).isoformat()
    models, state, model_metadata = train_and_evaluate(history)
    prospective_evaluation = grade_archived_snapshots(history)

    if event is None:
        payload = {
            "schema_version": 1,
            "sport": "f1",
            "run_date": run_day.isoformat(),
            "generated_at_utc": generated_at,
            "mode": "live_shadow",
            "publication_status": "no_upcoming_event",
            "event": None,
            "model": model_metadata,
            "prospective_evaluation": prospective_evaluation,
            "market": {"provider": "not_requested", "status": "no_upcoming_event", "observations": 0},
            "data_quality": {"status": "withheld", "reason": "No remaining Formula 1 race was found in the current-season schedule."},
            "selection": {
                "market": "race_winner",
                "minimum_edge": 0.03,
                "minimum_books": 1,
                "maximum_plays": 5,
                "staking_enabled": False,
            },
            "projections": [],
            "plays": [],
        }
        write_json(args.output, payload)
        archive_payload(payload)
        print(json.dumps({"output": str(args.output), "status": payload["publication_status"]}, indent=2))
        return 0

    entries = (
        list(read_json(args.standings_input))
        if args.standings_input
        else fetch_driver_standings(run_day.year)
    )
    if not entries:
        entries = entries_from_latest_race(history, run_day.year)
    grid: dict[str, int] = {}
    if not args.skip_grid and not args.standings_input:
        try:
            grid = fetch_starting_grid(event, entries)
        except Exception as error:
            print(f"[warn] OpenF1 starting grid unavailable: {error}")
    for entry in entries:
        entry["grid"] = grid.get(entry["driver_id"], int(entry.get("grid") or 0))

    projections = predict_event(models, state, event, entries)
    observations: list[dict] = []
    if args.odds_input:
        fixture = read_json(args.odds_input)
        observations = list(fixture.get("observations", fixture) if isinstance(fixture, dict) else fixture)
        market_audit = {"provider": "fixture", "status": "success" if observations else "no_markets", "observations": len(observations)}
    elif args.skip_odds:
        market_audit = {"provider": "not_requested", "status": "skipped", "observations": 0}
    else:
        observations, market_audit = fetch_available_odds(
            event_name=event["race_name"],
            provider_priority=tuple(
                item.strip() for item in os.getenv("F1_ODDS_PROVIDER_PRIORITY", "polymarket,kalshi").split(",") if item.strip()
            ),
        )
    attach_consensus_market(projections, observations)
    plays = [
        {**row, "market": "race_winner", "research_only": True}
        for row in projections
        if row.get("edge") is not None and row["edge"] >= 0.03 and row.get("book_count", 0) >= 1
    ][:5]
    grid_known = sum(row.get("grid_position") is not None for row in projections)
    market_ready = bool(observations)
    reason = (
        "Winner prices were matched to the model field; signals remain shadow research while prospective results accumulate."
        if market_ready
        else "No supported Formula 1 winner market was returned, so market-edge signals are withheld. Model probabilities remain visible."
    )
    payload = {
        "schema_version": 1,
        "sport": "f1",
        "run_date": run_day.isoformat(),
        "generated_at_utc": generated_at,
        "mode": "live_shadow",
        "publication_status": "shadow_current_pool" if market_ready else "model_only",
        "event": event,
        "model": model_metadata,
        "prospective_evaluation": prospective_evaluation,
        "market": market_audit,
        "data_quality": {
            "status": "shadow" if market_ready else "withheld",
            "reason": reason,
            "drivers": len(projections),
            "starting_grid_positions": grid_known,
            "market_observations": len(observations),
        },
        "selection": {
            "market": "race_winner",
            "minimum_edge": 0.03,
            "minimum_books": 1,
            "maximum_plays": 5,
            "staking_enabled": False,
        },
        "projections": projections,
        "plays": plays,
    }
    write_json(args.output, payload)
    archive_payload(payload)
    if args.snapshot_output:
        write_json(args.snapshot_output, {"schema_version": 1, "audit": market_audit, "observations": observations})
    print(json.dumps({"output": str(args.output), "event": event["race_name"], "drivers": len(projections), "plays": len(plays)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Daily settlement of the previous day's real, now-completed NFL plays
into the PARLAY_V2 leg-level calibration ledger.

WHY THIS SCRIPT EXISTS: calibration/ingest.py is, by its own module
docstring, "the ONLY place observations are admitted to the calibration
ledger" -- but nothing in the scheduled nfl-predictions.yml workflow ever
called it. The live PARLAY_POLICY_V2 decision (stage_parlay_v2.py, run
every day) only ever READS the calibration ledger via
CalibrationStore.observations_as_of; it never writes to it. Without this
script running daily, the ledger only ever grows via a one-time manual
historical backfill (calibration/historical_backfill.py) -- real games
completed since then never get admitted, and the live decision can never
see fresh real evidence.

WHAT THIS DOES: for the previous Eastern calendar day, reads every real
daily production snapshot already committed to git by nfl-predictions.yml
("Build current NFL slate" -- sports/nfl/data/production/snapshots/
<date>/<run_id>.json, exactly the source calibration/ingest.py's own
docstring names as intended), resolves the real NFL (season, week) each
snapshot's plays belong to directly from their own real game_start_utc
timestamps against the real nflverse schedule (never guessed from
calendar-date heuristics), and admits every now-gradeable play into the
calibration ledger via ingest.ingest_settled_week. A play whose game
hasn't happened yet, or whose real box score isn't aggregated yet, is
silently skipped (see settlement_source.grade_play) -- this script is
always safe to run, even on a snapshot that isn't fully settled yet.

SCOPE NOTE -- pair-level settlement is intentionally NOT done here.
calibration/pair_ingest.py needs ONE settled week's COMPLETE frozen
candidate-play set to correctly re-derive that week's pairing (its own
module docstring: "pairing is a pure function of that week's frozen
archived plays"). stage_parlay_v2.py's own weekly snapshot
(parlay_v2/reports/weekly_plays/<week_id>.json) is overwritten (not
merged) on every run within a week, so if the live board's plays differ
across days within the same week (e.g. Thursday's game vs. Sunday's
slate), the archived file only ever reflects whichever day ran last --
automating pair-level settlement against that file today would risk
silently dropping real legs from the pairing universe. That is a
separate, real gap in stage_parlay_v2.py's own archiving (it would need
to accumulate plays across a week, not overwrite), left for a follow-up
change rather than worked around here.

    python -m sports.nfl.scripts.settle_parlay_v2_calibration --run-date 2026-09-15
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sports.nfl.parlay_v2.calibration import ingest
from sports.nfl.scripts.fetch_historical_nfl_props import SCHEDULE_URL, _kickoff_utc

NFL_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SNAPSHOTS_ROOT = NFL_ROOT / "data" / "production" / "snapshots"
CALIBRATION_LEDGER = NFL_ROOT / "parlay_v2" / "calibration" / "reports" / "calibration_ledger.jsonl"
SEASON_WEEK_MATCH_TOLERANCE_MINUTES = 10


def load_schedule() -> pd.DataFrame:
    """Real nflverse regular-season schedule, with a real UTC kickoff
    column -- the same source and conversion (_kickoff_utc) already used
    elsewhere in this pipeline (e.g. build_nfl_week_pool.py)."""
    schedule = pd.read_parquet(SCHEDULE_URL)
    schedule = schedule.loc[schedule["game_type"].astype(str) == "REG"].copy()
    schedule["commence_time_utc"] = _kickoff_utc(schedule)
    return schedule.dropna(subset=["commence_time_utc"])


def resolve_season_week(plays: list[dict[str, Any]], schedule: pd.DataFrame) -> Optional[tuple[int, int]]:
    """Resolves the real (season, week) a set of plays belongs to from
    each play's own real game_start_utc against the real schedule --
    never inferred from the run date. Every play with a resolvable
    timestamp must agree on the same (season, week); returns None (never
    a guess) if no play resolves, or if resolved plays disagree (a real
    data problem worth surfacing via an explicit skip, not silently
    picking one)."""
    if schedule.empty:
        return None
    resolved: set[tuple[int, int]] = set()
    for play in plays:
        raw_ts = play.get("game_start_utc")
        if not raw_ts:
            continue
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        deltas = (schedule["commence_time_utc"] - ts).abs()
        idx = deltas.idxmin()
        if deltas.loc[idx] > pd.Timedelta(minutes=SEASON_WEEK_MATCH_TOLERANCE_MINUTES):
            continue
        row = schedule.loc[idx]
        resolved.add((int(row["season"]), int(row["week"])))
    if len(resolved) == 1:
        return next(iter(resolved))
    return None


def settle_snapshot(snapshot_path: Path, schedule: pd.DataFrame, *, ledger_path: Path = CALIBRATION_LEDGER) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"snapshot": str(snapshot_path), "status": "unreadable", "error": str(exc)}

    plays = payload.get("plays") if isinstance(payload, dict) else None
    plays = list(plays) if isinstance(plays, list) else []
    if not plays:
        return {"snapshot": str(snapshot_path), "status": "no_plays"}

    resolved = resolve_season_week(plays, schedule)
    if resolved is None:
        return {"snapshot": str(snapshot_path), "status": "season_week_unresolved", "plays": len(plays)}

    season, week = resolved
    summary = ingest.ingest_settled_week(snapshot_path, season=season, week=week, ledger_path=ledger_path)
    return {"snapshot": str(snapshot_path), "status": "settled", "season": season, "week": week, **summary}


def settle_previous_day(run_date: str, *, ledger_path: Path = CALIBRATION_LEDGER) -> dict[str, Any]:
    run_day = date.fromisoformat(run_date)
    target_day = run_day - timedelta(days=1)
    target_dir = PRODUCTION_SNAPSHOTS_ROOT / target_day.isoformat()

    if not target_dir.is_dir():
        return {"target_date": target_day.isoformat(), "status": "no_snapshot_dir", "results": []}

    schedule = load_schedule()
    results = [
        settle_snapshot(snapshot_path, schedule, ledger_path=ledger_path)
        for snapshot_path in sorted(target_dir.glob("*.json"))
    ]
    return {"target_date": target_day.isoformat(), "status": "checked", "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True, help="Eastern run date (YYYY-MM-DD); settles the PREVIOUS day's production snapshot(s).")
    parser.add_argument("--ledger", type=Path, default=CALIBRATION_LEDGER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = settle_previous_day(args.run_date, ledger_path=args.ledger)
    except Exception as exc:  # noqa: BLE001 -- settlement must never break the daily board
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 0
    print(json.dumps(summary, indent=2))
    # Never fail the workflow over settlement -- a real network hiccup or
    # not-yet-aggregated box score is expected, not an error condition.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

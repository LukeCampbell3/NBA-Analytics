from __future__ import annotations

"""Real settlement -> calibration ledger admission. This is the ONLY
place observations are admitted to the calibration ledger --
run_parlay_v2.py (the live decision path) never writes to it, only reads
from it via CalibrationStore.observations_as_of. Ported from
sports/mlb/parlay_v2/calibration/ingest.py, replacing MLB's
multi_target_universe-based grading with settlement_source.grade_play
(real nflverse play-by-play aggregation) -- see settlement_source.py's
module docstring for why NFL needs its own bridge here.

Calling this on a snapshot whose week's real stats are not yet aggregated
is safe: ungraded plays are silently skipped (grade_play returns None),
so a too-early ingestion attempt just admits fewer/zero observations
rather than failing or fabricating an outcome. Running it again after the
season's aggregated cache is refreshed picks up whatever has since
settled (idempotent by source_id, forward-only by construction:
calibration_admitted_at is always "now" at ingestion time, never
backdated).

Only ever called with ONE real, live (season, week, snapshot) at a time.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sports.nfl.predictions.daily_policy import american_to_decimal

from .schema import build_observation
from .settlement_source import SETTLEMENT_SOURCE_VERSION, grade_play, load_season_actuals
from .store import CalibrationStore

INGEST_VERSION = "NFL_CALIBRATION_INGEST_V1"
PREDICTIVE_VERSION = "NFL_PASSING_LOSS_AWARE_META_POLICY_V2"


def _play_source_hash(play: dict[str, Any], win: bool) -> str:
    fields = {
        "player_id": play.get("player_id"),
        "event_id": play.get("event_id"),
        "target": play.get("target") or play.get("market"),
        "direction": play.get("direction"),
        "line": play.get("line"),
        "win": bool(win),
        "selected_side_price": play.get("selected_side_price"),
    }
    canonical = "|".join(f"{k}={fields[k]}" for k in sorted(fields))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ingest_settled_week(
    snapshot_path: Path,
    *,
    season: int,
    week: int,
    ledger_path: Path,
    actuals_cache_path: Path | None = None,
) -> dict:
    """Admits every graded play from ONE settled week's archived snapshot
    (sports/nfl/data/production/snapshots/<date>/<run_id>.json, already
    committed to git by nfl-predictions.yml) into the calibration ledger.
    Returns a summary dict; never raises for "not yet settled" plays
    (those are simply skipped), only for real errors (e.g. an unreadable
    snapshot file)."""
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_file():
        return {"week_id": f"{season}-W{week}", "action_eligible_rows": 0, "admitted": 0, "already_present": 0, "ledger_path": str(ledger_path), "reason": "snapshot_not_found"}

    with open(snapshot_path, encoding="utf-8") as f:
        payload = json.load(f)
    plays = payload.get("plays") if isinstance(payload, dict) else None
    plays = list(plays) if isinstance(plays, list) else []

    actuals = load_season_actuals(season, cache_path=actuals_cache_path)
    week_id = f"{season}-W{week:02d}"

    store = CalibrationStore(ledger_path)
    now = datetime.now(timezone.utc).isoformat()
    admitted = 0
    already_present = 0
    graded_rows = 0

    for play in plays:
        win = grade_play(play, actuals, season=season, week=week)
        if win is None:
            continue  # not yet settled / push / unmapped target -- see settlement_source.py
        graded_rows += 1

        decimal_price = None
        price = play.get("selected_side_price")
        if price is not None:
            decimal_price = float(american_to_decimal(float(price)))
        actual_unit_return = (decimal_price - 1.0) if (win and decimal_price is not None) else (-1.0 if not win else None)
        target = str(play.get("target") or play.get("market"))
        direction = str(play.get("direction"))
        line = float(play["line"])
        source_id = f"{week_id}|{play.get('player_id')}|{play.get('event_id')}|{target}|{direction}|{line}"

        observation = build_observation(
            slate_id=week_id,
            game_id=str(play.get("event_id") or ""),
            event_date=str(play.get("game_start_utc") or week_id)[:10],
            player_id=str(play.get("player_id") or ""),
            player_name=str(play.get("player") or ""),
            target=target,
            side=direction,
            line=line,
            book=str(play.get("selected_sportsbook_key") or ""),
            quote_decimal=(decimal_price if decimal_price is not None else 0.0),
            # weekly-snapshot granularity -- no finer quote timestamp
            # exists upstream, matching candidate_adapter.Leg.quote_timestamp's
            # own convention for the same reason.
            quote_timestamp=str(play.get("snapshot_time_utc") or week_id),
            prediction_value=float(play.get("projection") or 0.0),
            predictive_probability_if_available=float(play.get("model_hit_probability") or 0.0),
            state_version=SETTLEMENT_SOURCE_VERSION,
            predictive_version=PREDICTIVE_VERSION,
            market_bucket=target,
            line_bucket=f"{target}|{direction}|{line}",
            state_bucket=f"{PREDICTIVE_VERSION}|{SETTLEMENT_SOURCE_VERSION}",
            settlement_status="win" if win else "loss",
            actual_outcome=1.0 if win else 0.0,
            actual_unit_return=(actual_unit_return if actual_unit_return is not None else -1.0),
            # Best-effort: the real per-week decision_frozen_at from the
            # live run_parlay_v2.py decision that week is not separately
            # persisted anywhere today. This is a provenance/audit field
            # only; it is NEVER what enforces the forward-only invariant
            # (calibration_admitted_at is), so this approximation does not
            # create any leakage risk.
            decision_frozen_at=str(play.get("snapshot_time_utc") or f"{week_id}T00:00:00Z"),
            settled_at=now,
            calibration_admitted_at=now,
            source_id=source_id,
            source_hash=_play_source_hash(play, win),
        )
        if store.admit(observation):
            admitted += 1
        else:
            already_present += 1

    return {
        "week_id": week_id,
        "action_eligible_rows": graded_rows,
        "admitted": admitted,
        "already_present": already_present,
        "ledger_path": str(ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit one settled NFL week's graded plays into the forward-only calibration ledger.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to the archived weekly plays snapshot JSON.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--actuals-cache", type=Path, default=None)
    args = parser.parse_args()
    summary = ingest_settled_week(
        args.snapshot, season=args.season, week=args.week, ledger_path=args.ledger, actuals_cache_path=args.actuals_cache,
    )
    print(summary)


if __name__ == "__main__":
    main()

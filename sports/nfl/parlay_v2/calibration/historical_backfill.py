from __future__ import annotations

"""One-time REAL historical data backfill for the NFL calibration ledger
-- user-authorized (chat record, 2026-08-23), admitting REAL settled
market-selector pool rows from past seasons so real week-1 2026
candidates have a chance to clear market_support/line_support/
state_support instead of abstaining on zero evidence.

THIS IS NOT SIMULATION. Every row admitted here is a real graded NFL
prop already sitting in this repo's own backtest pool files
(sports/nfl/data/evaluation/*.csv) -- real market lines, real prices,
real bookmakers, real settled win/loss outcomes from real past games.
Nothing here is invented, sampled, or estimated.

Scope, exactly as authorized (do not widen without a fresh check-in):
  - 2025 season (recent_selector_pool_2025.csv): all 18 real weeks, used
    in full as genuine calibration evidence.
  - 2022 season (market_selector_pool_2022.csv): the FULL season (all 18
    real weeks) -- expanded from an initial weeks-1-2-only cap after the
    initial backfill left only 2 real line buckets with enough depth
    (N_LINE=20) to ever pass line_support, an explicitly authorized
    follow-up once that real, quantified shortfall was disclosed. Still
    the broader POOL file, never market_selector_validated_pool_2022.csv
    (the actual locked holdout run_nfl_production_replay.py grades
    sports/nfl/predictions/daily_policy.py's old shadow-parlay logic
    against) -- this backfill never reads that file and never touches
    that frozen result.
  - 2021 was explicitly NOT authorized for this backfill and is not used.

Labeled with its own honest predictive_version/state_version
(PREDICTIVE_VERSION below) -- deliberately DISTINCT from the live
pipeline's NFL_PASSING_LOSS_AWARE_META_POLICY_V2 label, since this data
was not produced by the live weekly select_live_board pipeline. This
labeling has no effect on support gating itself (market_support/
line_support match on market_bucket/line_bucket only; state_support
counts distinct slate_id only -- see calibration/support.py and
snapshot.py) but keeps the ledger's own provenance record honest.

This never touches sports/nfl/predictions/daily_policy.py, never re-runs
or re-validates any existing backtest/holdout claim, and never feeds back
into model training -- it is a one-time admission of already-real,
already-settled rows into the NEW system's forward-only calibration
ledger (calibration/store.py), through the exact same schema.build_observation
/ CalibrationStore.admit path every other real admission uses. Idempotent:
safe to re-run (existing observation_ids are skipped, never duplicated).
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sports.nfl.predictions.daily_policy import american_to_decimal

from .schema import build_observation
from .store import CalibrationStore

BACKFILL_VERSION = "NFL_HISTORICAL_POOL_BACKFILL_V1"
# Deliberately distinct from the live pipeline's own predictive_version --
# see module docstring.
PREDICTIVE_VERSION = "NFL_HISTORICAL_MARKET_SELECTOR_POOL"
STATE_VERSION = "HISTORICAL_BACKFILL_BROAD_V1"

REPO_ROOT = Path(__file__).resolve().parents[4]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"

# Exactly the sources authorized, nothing else -- see module docstring.
# weeks=None means "all real weeks in the file"; a tuple caps to exactly
# those week numbers.
SOURCES: tuple[dict[str, Any], ...] = (
    {"path": NFL_ROOT / "data/evaluation/recent_selector_pool_2025.csv", "weeks": None},
    {"path": NFL_ROOT / "data/evaluation/market_selector_pool_2022.csv", "weeks": None},
)


def _row_source_hash(row: pd.Series) -> str:
    fields = {
        "player_id": row["player_id"], "season": row["season"], "week": row["week"],
        "target": row["target"], "side": row["side"], "line": row["line"],
        "result": row["result"], "selected_price": row["selected_price"],
    }
    canonical = "|".join(f"{k}={fields[k]}" for k in sorted(fields))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def backfill_historical_pool(ledger_path: Path, *, sources: tuple[dict[str, Any], ...] = SOURCES) -> dict:
    store = CalibrationStore(ledger_path)
    now = datetime.now(timezone.utc).isoformat()
    admitted = 0
    already_present = 0
    skipped_incomplete = 0
    weeks_seen: set[str] = set()
    rows_considered = 0

    for source in sources:
        path = Path(source["path"])
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        if source["weeks"] is not None:
            df = df[df["week"].isin(source["weeks"])]

        for _, row in df.iterrows():
            rows_considered += 1
            price = row.get("selected_price")
            line = row.get("line")
            if pd.isna(price) or pd.isna(line) or pd.isna(row.get("player_id")):
                # Never fabricate a missing field -- skip incomplete rows
                # rather than guessing. Not expected to trigger on the
                # authorized source files (verified complete before this
                # module was written), but kept as a real guard.
                skipped_incomplete += 1
                continue

            season = int(row["season"])
            week = int(row["week"])
            slate_id = f"HIST-{season}-W{week:02d}"
            weeks_seen.add(slate_id)

            win = str(row["result"]).strip().lower() == "win"
            decimal_price = float(american_to_decimal(float(price)))
            actual_unit_return = (decimal_price - 1.0) if win else -1.0
            direction = str(row["side"]).upper()
            target = str(row["target"])
            line_value = float(line)
            game_id = f"{row.get('recent_team')}_vs_{row.get('opponent_team')}_{season}W{week}"
            source_id = f"{slate_id}|{row['player_id']}|{game_id}|{target}|{direction}|{line_value}"

            observation = build_observation(
                slate_id=slate_id,
                game_id=game_id,
                event_date=slate_id,
                player_id=str(row["player_id"]),
                player_name=str(row.get("player_display_name") or ""),
                target=target,
                side=direction,
                line=line_value,
                book=str(row.get("bookmaker") or ""),
                quote_decimal=decimal_price,
                quote_timestamp=str(row.get("snapshot_time_utc") or slate_id),
                prediction_value=float(row.get("current_prediction") or 0.0),
                predictive_probability_if_available=float(row.get("estimated_side_probability") or 0.0),
                state_version=STATE_VERSION,
                predictive_version=PREDICTIVE_VERSION,
                market_bucket=target,
                line_bucket=f"{target}|{direction}|{line_value}",
                state_bucket=f"{PREDICTIVE_VERSION}|{STATE_VERSION}",
                settlement_status="win" if win else "loss",
                actual_outcome=1.0 if win else 0.0,
                actual_unit_return=actual_unit_return,
                # Best-effort provenance timestamp -- real game commence
                # time when available, never enforcing the forward-only
                # invariant (calibration_admitted_at does that).
                decision_frozen_at=str(row.get("commence_time_utc") or f"{slate_id}T00:00:00Z"),
                settled_at=now,
                calibration_admitted_at=now,
                source_id=source_id,
                source_hash=_row_source_hash(row),
            )
            if store.admit(observation):
                admitted += 1
            else:
                already_present += 1

    return {
        "rows_considered": rows_considered,
        "admitted": admitted,
        "already_present": already_present,
        "skipped_incomplete": skipped_incomplete,
        "independent_weeks_admitted": len(weeks_seen),
        "ledger_path": str(ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit REAL historical NFL market-selector pool rows (2025 full season + 2022 weeks 1-2 only, per explicit authorization) into the forward-only calibration ledger.")
    parser.add_argument("--ledger", type=Path, default=NFL_ROOT / "parlay_v2/calibration/reports/calibration_ledger.jsonl")
    args = parser.parse_args()
    summary = backfill_historical_pool(args.ledger)
    print(summary)


if __name__ == "__main__":
    main()

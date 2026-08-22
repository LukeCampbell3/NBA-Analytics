from __future__ import annotations

"""Real settlement -> calibration ledger admission. This is the ONLY
place observations are admitted to the calibration ledger --
run_parlay_v2.py (the live decision path) never writes to it, only reads
from it via CalibrationStore.observations_as_of.

Reuses joint_position_builder_v2.multi_target_universe.build_multi_target_universe
UNCHANGED: it already does exactly what's needed here -- for a SETTLED
day, grade every action-eligible row against real outcomes recorded in
Player-Predictor/Data-Proc-MLB. Calling it on a day whose outcomes are not
yet in that processed data is safe: ungraded rows are silently skipped
(build_multi_target_universe's own existing behavior -- `if actual is
None: continue`), so a too-early ingestion attempt just admits fewer/zero
observations rather than failing or fabricating an outcome. Running it
again on a later day picks up whatever has since settled (idempotent by
source_id, forward-only by construction: calibration_admitted_at is
always "now" at ingestion time, never backdated).

Only ever called with ONE real, live stamp at a time (never
DEVELOPMENT_STAMPS/TEST_STAMPS -- see data_windows.py). frozen_bias(),
which build_multi_target_universe relies on, is itself computed once from
the already-frozen DERIVE_STAMPS window and does not touch the settled
day being ingested here, so this introduces no leakage into that frozen
research partition.
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sports.mlb.research.joint_position_builder_v2.multi_target_universe import (
    PRICED_TARGETS,
    action_universe,
    build_multi_target_universe,
)

from .schema import build_observation
from .store import CalibrationStore

INGEST_VERSION = "CALIBRATION_INGEST_V1"
PREDICTIVE_VERSION = "H_OVER_RANKER_V1+MULTI_TARGET"


def _row_source_hash(row: pd.Series) -> str:
    fields = {
        "player_key": row["player_key"],
        "game_id": row["game_id"],
        "target": row["target"],
        "direction": row["direction"],
        "market_line": row["market_line"],
        "win": bool(row["win"]),
        "decimal_price": row["decimal_price"],
    }
    canonical = "|".join(f"{k}={fields[k]}" for k in sorted(fields))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ingest_settled_slate(
    stamp: str,
    *,
    ledger_path: Path,
    targets: tuple[str, ...] = PRICED_TARGETS,
    mode: str = "broad",
) -> dict:
    """Admits every graded, action-eligible row for ONE settled day into
    the calibration ledger. Returns a summary dict; never raises for "not
    yet settled" rows (those are simply absent from `action`), only for
    real errors (e.g. an unreadable pool file, propagated from
    build_multi_target_universe).

    A day with a pool file but literally zero rows for any requested
    target (a real off day, or every game rained out/postponed) makes
    build_multi_target_universe return a columnless empty DataFrame --
    action_universe would raise a bare KeyError on that (it indexes
    universe["in_support"], which doesn't exist on an empty frame with no
    columns). Guarded here the same way world_gate_research.usable_stamps
    and pair_ingest.ingest_settled_pairs already guard it: an empty
    universe just means zero rows admitted, not a real error."""
    universe = build_multi_target_universe((stamp,), targets=targets, mode=mode)
    action = action_universe(universe) if not universe.empty else universe

    store = CalibrationStore(ledger_path)
    now = datetime.now(timezone.utc).isoformat()
    admitted = 0
    already_present = 0

    for _, row in action.iterrows():
        decimal_price = float(row["decimal_price"])
        win = bool(row["win"])
        actual_unit_return = (decimal_price - 1.0) if win else -1.0
        source_id = f"{stamp}|{row['player_key']}|{row['game_id']}|{row['target']}|{row['direction']}|{row['market_line']}"

        observation = build_observation(
            slate_id=stamp,
            game_id=str(row["game_id"]),
            event_date=stamp,
            player_id=str(row["player_key"]),
            player_name=str(row["player"]),
            target=str(row["target"]),
            side=str(row["direction"]),
            line=float(row["market_line"]),
            book=str(row.get("market_source") or ""),
            quote_decimal=decimal_price,
            # daily-pool granularity -- no finer quote timestamp exists
            # upstream, matching candidate_adapter.Leg.quote_timestamp's
            # own convention for the same reason.
            quote_timestamp=stamp,
            prediction_value=float(row["corrected_prediction"]),
            predictive_probability_if_available=float(row["marginal_probability"]),
            state_version=mode,
            predictive_version=PREDICTIVE_VERSION,
            market_bucket=str(row["target"]),
            line_bucket=f"{row['target']}|{row['direction']}|{row['market_line']}",
            state_bucket=f"{PREDICTIVE_VERSION}|{mode}",
            settlement_status="win" if win else "loss",
            actual_outcome=1.0 if win else 0.0,
            actual_unit_return=actual_unit_return,
            # Best-effort: the real per-slate decision_frozen_at from the
            # live run_parlay_v2.py decision that day is not separately
            # persisted anywhere today (its JSON output is transient CI
            # workspace state -- see run_daily_predictions.py). This is a
            # provenance/audit field only; it is NEVER what enforces the
            # forward-only invariant (calibration_admitted_at is), so this
            # approximation does not create any leakage risk.
            decision_frozen_at=f"{stamp}T17:00:00Z",
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
        "stamp": stamp,
        "action_eligible_rows": int(len(action)),
        "admitted": admitted,
        "already_present": already_present,
        "ledger_path": str(ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit one settled MLB slate's graded rows into the forward-only calibration ledger.")
    parser.add_argument("--stamp", required=True, help="Settled slate date stamp, e.g. 20260820")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--mode", choices=["narrow", "broad"], default="broad")
    args = parser.parse_args()
    summary = ingest_settled_slate(args.stamp, ledger_path=args.ledger, mode=args.mode)
    print(summary)


if __name__ == "__main__":
    main()

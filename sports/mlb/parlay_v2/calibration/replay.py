from __future__ import annotations

"""Deterministic replay for the CALIBRATION stream (mission section 11A).
Given the immutable ledger through time T, reproduce snapshot ids/hashes/
support counts/classifications/metrics. Never refits anything, never
alters existing ledger rows, and never reads observations admitted at or
after the cutoff (enforced by store.observations_as_of itself).

Policy-side replay (selected pair/abstention, cumulative G, anytime
bounds, PolicyStatus transitions) lives in
sports/mlb/research/parlay_certification_v2/replay.py -- see mission
section 11B.
"""

from dataclasses import dataclass

from .snapshot import CalibrationSnapshot, build_snapshot
from .store import CalibrationStore
from .support import CandidateSupport, evaluate_support


@dataclass(frozen=True)
class CalibrationReplayResult:
    snapshot: CalibrationSnapshot
    support: CandidateSupport


def replay_calibration_as_of(
    store: CalibrationStore,
    *,
    as_of: str,
    market_bucket: str,
    line_bucket: str,
    state_bucket: str,
) -> CalibrationReplayResult:
    """Re-derives the snapshot and one candidate's support from the ledger
    as it stood at `as_of`, purely from the append-only store -- no
    external state. Calling this twice with the same store contents and
    the same `as_of`/bucket arguments always returns identical results
    (see test_calibration_replay_is_deterministic)."""
    snapshot = build_snapshot(store, as_of=as_of)
    rows = store.observations_as_of(as_of)
    support = evaluate_support(
        rows,
        market_bucket=market_bucket,
        line_bucket=line_bucket,
        state_bucket=state_bucket,
        independent_slate_count=snapshot.independent_slate_count,
    )
    return CalibrationReplayResult(snapshot=snapshot, support=support)

from __future__ import annotations

"""Immutable, reproducible calibration snapshots (mission section 4).
Every V2 decision references exactly one snapshot, built via
`build_snapshot(store, as_of=decision.decision_frozen_at)`. A snapshot
is fully determined by the ledger's content as-of that cutoff plus the
frozen calibration_version -- same ledger state -> byte-identical
snapshot_id/sha256 (see replay.py's determinism test).

Enforces `calibration_as_of < decision_frozen_at` for every live
decision via `assert_snapshot_precedes_decision` -- callers must call
this before trusting a snapshot for a real decision.
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .store import CalibrationStore
from .versioning import CALIBRATION_VERSION

SNAPSHOT_VERSION = "CALIBRATION_SNAPSHOT_V1"


def _bucket_summary(rows: list[dict], bucket_key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(bucket_key, "unknown"))] += 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class CalibrationSnapshot:
    calibration_snapshot_id: str
    calibration_snapshot_sha256: str
    calibration_as_of: str
    observation_count: int
    independent_slate_count: int
    market_support_summary: dict[str, int]
    line_support_summary: dict[str, int]
    state_support_summary: dict[str, int]
    joint_support_summary: dict[str, int]
    calibration_version: str = CALIBRATION_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_snapshot(store: CalibrationStore, *, as_of: str) -> CalibrationSnapshot:
    rows = store.observations_as_of(as_of)  # forward-only: strictly before `as_of`
    rows_sorted = sorted(rows, key=lambda r: r["observation_id"])  # deterministic order

    market_summary = _bucket_summary(rows_sorted, "market_bucket")
    line_summary = _bucket_summary(rows_sorted, "line_bucket")
    state_summary = _bucket_summary(rows_sorted, "state_bucket")
    joint_summary = _bucket_summary(
        [{"joint_bucket": f"{r.get('market_bucket')}|{r.get('state_bucket')}"} for r in rows_sorted],
        "joint_bucket",
    )
    independent_slates = len({r["slate_id"] for r in rows_sorted})

    canonical_content = {
        "as_of": as_of,
        "calibration_version": CALIBRATION_VERSION,
        "observation_ids": [r["observation_id"] for r in rows_sorted],
    }
    canonical_json = json.dumps(canonical_content, sort_keys=True)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return CalibrationSnapshot(
        calibration_snapshot_id=digest[:16],
        calibration_snapshot_sha256=digest,
        calibration_as_of=as_of,
        observation_count=len(rows_sorted),
        independent_slate_count=independent_slates,
        market_support_summary=market_summary,
        line_support_summary=line_summary,
        state_support_summary=state_summary,
        joint_support_summary=joint_summary,
    )


def assert_snapshot_precedes_decision(snapshot: CalibrationSnapshot, decision_frozen_at: str) -> None:
    """Strict timestamp comparison (mission section 4: "Prefer a strict
    timestamp comparison"). Raises if violated -- this must never pass
    silently for a live decision."""
    if not str(snapshot.calibration_as_of) < str(decision_frozen_at):
        raise ValueError(
            f"calibration_as_of={snapshot.calibration_as_of!r} does not strictly precede "
            f"decision_frozen_at={decision_frozen_at!r} -- refusing to use this snapshot for a live decision"
        )

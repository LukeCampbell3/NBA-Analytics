from __future__ import annotations

"""Durable per-day DecisionRecord ledger (mission section 19/27).

DecisionRecords are already computed at decision time inside
run_parlay_v2.build_slate_payload, but that function's caller (the daily
CI pipeline) writes them only into an EPHEMERAL per-run JSON file under
sports/mlb/data/predictions/daily_runs/<stamp>/ -- a path that is neither
cached nor git-committed, so it does not survive past the CI run that
produced it. A later day's settlement ingestion (settle_evidence.py)
needs to read a PAST day's exact frozen decision, so it must come from
somewhere durable instead. This store is that durable home: one
git-committed JSON-lines file, appended to exactly once per day at
decision time (build_slate_payload calls `admit` itself, see
run_parlay_v2.py), never revised afterward -- a decision, once frozen, is
never rewritten, matching DecisionRecord's own "frozen at cutoff" nature.

Structurally separate from CalibrationStore/PairObservationStore (both
leg/pair-level, never touch policy evidence) and from EvidenceStore (the
SETTLED outcome, admitted later by settle_evidence.py once real results
exist) -- this store holds only the PRE-settlement decision itself.
"""

import json
from dataclasses import asdict
from pathlib import Path

from .evidence_store import DecisionRecord

DECISION_RECORD_STORE_VERSION = "DECISION_RECORD_STORE_V1"


class DecisionRecordStore:
    """Append-only JSON-lines store, idempotent by `date` -- one row per
    slate day, ever, regardless of how many times a day's step is
    re-run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def existing_dates(self) -> set[str]:
        return {row["date"] for row in self._read_all()}

    def admit(self, record: DecisionRecord) -> bool:
        """Returns True if appended, False if a record for this date
        already exists (idempotent no-op -- a frozen decision is never
        revised, even by a later re-run of the same day's step)."""
        if record.date in self.existing_dates():
            return False
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), sort_keys=True, default=str) + "\n")
        return True

    def all_records(self) -> list[dict]:
        return self._read_all()

    def record_for_date(self, date: str) -> dict | None:
        for row in self._read_all():
            if row["date"] == date:
                return row
        return None

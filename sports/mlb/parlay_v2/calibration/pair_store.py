from __future__ import annotations

"""Append-only, idempotent pair-observation ledger (mission section 9/10).
Structurally separate from CalibrationStore (leg-level) and EvidenceStore
(one-row-per-slate policy evidence) -- a different file, a different
schema, never cross-referenced except by shared slate_id/predictive_version
for audit purposes.

Never overwrites. Duplicate ingestion (re-running settlement for an
already-ingested day) is a no-op, exactly like CalibrationStore.
"""

import json
from pathlib import Path

from .pair_schema import PairObservation

STORE_VERSION = "PAIR_OBSERVATION_STORE_V1"


class PairObservationStore:
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

    def existing_observation_ids(self) -> set[str]:
        return {row["observation_id"] for row in self._read_all()}

    def admit(self, observation: PairObservation) -> bool:
        if observation.observation_id in self.existing_observation_ids():
            return False
        with open(self.path, "a") as f:
            f.write(json.dumps(observation.as_dict(), sort_keys=True, default=str) + "\n")
        return True

    def all_observations(self) -> list[dict]:
        return self._read_all()

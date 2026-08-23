from __future__ import annotations

"""Forward-only, append-only, idempotent calibration ledger (mission
section 3/18-F).

FORWARD-ONLY INVARIANT: `observations_as_of(cutoff)` returns only
observations with `calibration_admitted_at < cutoff` (STRICT). This is
the single choke point that enforces "day t outcome may influence day
t+1, never day t" (mission section 2's fundamental invariant) -- nothing
elsewhere in this package is allowed to read the raw ledger file directly
for a support calculation; everything must go through this method with an
explicit cutoff equal to the CURRENT decision's `decision_frozen_at`.

IDEMPOTENCY: `observation_id` is content-derived (schema.build_observation)
from exact-event identity + quote + settlement outcome + source_id.
Ingesting the same real settlement event twice (duplicate callback,
out-of-order redelivery) writes it at most once.

NEVER OVERWRITE: once written, a row is never rewritten. There is no
update/delete method on this class by design.
"""

import json
from pathlib import Path

from .schema import SCHEMA_VERSION, CalibrationObservation

STORE_VERSION = "CALIBRATION_STORE_V1"


class CalibrationStore:
    def __init__(self, path: Path, *, calibration_version: str = SCHEMA_VERSION):
        self.path = Path(path)
        self.calibration_version = calibration_version
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

    def admit(self, observation: CalibrationObservation) -> bool:
        """Append one observation. Returns True if newly admitted, False
        if this exact observation_id already exists (idempotent no-op).
        Requires settlement to already be final: refuses to admit an
        observation whose settled_at is empty/missing, or whose
        calibration_admitted_at is not >= settled_at (an observation
        cannot be admitted to calibration before it is settled)."""
        if observation.calibration_version != self.calibration_version:
            raise ValueError(
                f"observation calibration_version {observation.calibration_version!r} does not match "
                f"store calibration_version {self.calibration_version!r} -- refusing to pool observations across schema versions"
            )
        if not observation.settled_at:
            raise ValueError("cannot admit an observation with no settled_at -- settlement must be final first")
        if str(observation.calibration_admitted_at) < str(observation.settled_at):
            raise ValueError("calibration_admitted_at must be >= settled_at -- cannot admit before settlement is final")
        if observation.observation_id in self.existing_observation_ids():
            return False
        with open(self.path, "a") as f:
            f.write(json.dumps(observation.as_dict(), sort_keys=True, default=str) + "\n")
        return True

    def observations_as_of(self, cutoff_timestamp: str) -> list[dict]:
        """THE forward-only choke point. Returns only rows with
        calibration_admitted_at STRICTLY BEFORE cutoff_timestamp (string
        ISO-8601 comparison, consistent with the rest of this package).
        Callers computing today's support MUST pass today's own
        decision_frozen_at as the cutoff -- never "now", which could
        include same-day admissions from a differently-ordered pipeline
        run."""
        return [row for row in self._read_all() if str(row["calibration_admitted_at"]) < str(cutoff_timestamp)]

    def all_observations(self) -> list[dict]:
        """Full ledger contents, in append order -- for replay/audit only.
        NEVER use this for a live support calculation (use
        observations_as_of with an explicit cutoff instead)."""
        return self._read_all()

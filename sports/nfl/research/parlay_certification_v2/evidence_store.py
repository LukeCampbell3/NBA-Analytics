from __future__ import annotations

"""ATOMIC SLATE-DAY EVIDENCE (mission section 3) -- the evidence unit is
ONE eligible slate day, not one leg.

Two phases:
  1. At decision cutoff, freeze a DecisionRecord: eligible flag/reason,
     timestamp, policy/model versions, candidate universe size, selected
     wager or abstention, accepted quote/book, configured c/r/delta/R_max,
     world-certificate diagnostics.
  2. After final settlement, append EXACTLY ONE immutable
     FinalEvidenceRecord (E_t, A_t, ell_t, R_t, settlement status/timestamp,
     policy version, source/provenance id). Evidence is never updated
     before settlement is final, and never rewritten after.

Settlement ingestion is idempotent by `source_id` (a stable identifier for
the real-world settlement event) -- duplicate or out-of-order settlement
callbacks are safe no-ops, never duplicate evidence rows. One JSON-lines
file per policy_version enforces version isolation structurally (section
14/15.M): evidence from different policy versions cannot be pooled by
accident, because they physically live in different files.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EVIDENCE_STORE_VERSION = "EVIDENCE_STORE_V1"


@dataclass(frozen=True)
class DecisionRecord:
    date: str
    eligible: bool
    eligibility_reason: str
    eligibility_version: str
    decision_timestamp_utc: str
    policy_version: str
    predictive_model_version: str
    candidate_universe_size: int
    action: int  # A_t in {0, 1}
    selected_wager: str | None
    accepted_decimal_price: float | None
    accepted_book: str | None
    c: float
    r: float
    delta: float
    r_max: float
    world_certificate_diagnostics: dict[str, Any] | None = None
    # Additive field (mission: "Resolve the remaining PARLAY_V2 APS /
    # counterexample admission bottleneck") -- defaults to "REQUIRED" so
    # every existing/replayed record (including PARLAY_POLICY_V2_
    # PROSPECTIVE_002's, which never sets this explicitly) is read
    # correctly as having used the original REQUIRED admission rule.
    # EvidenceStore still pools by policy_version (the structural shape
    # identifier, unchanged) -- this field lets a single evidence file
    # distinguish which admission rule produced each row, since a
    # materially different world_gate_mode is a different prospective
    # policy attempt even when the structural policy_version is shared.
    world_gate_mode: str = "REQUIRED"


@dataclass(frozen=True)
class FinalEvidenceRecord:
    date: str
    policy_version: str
    eligible: int  # E_t
    action: int  # A_t
    loss: int  # ell_t
    realized_return: float  # R_t
    settlement_status: str
    settlement_timestamp_utc: str
    source_id: str  # idempotency key -- unique per real-world settlement event
    decision_record: DecisionRecord


class EvidenceStore:
    """Append-only JSON-lines store, one file per policy_version."""

    def __init__(self, root: Path, policy_version: str):
        self.root = Path(root)
        self.policy_version = policy_version
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / f"{policy_version}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def _load_existing(self) -> list[dict]:
        if not self._path.exists():
            return []
        rows: list[dict] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def existing_source_ids(self) -> set[str]:
        return {row["source_id"] for row in self._load_existing()}

    def append_final_settlement(self, record: FinalEvidenceRecord) -> bool:
        """Returns True if appended, False if a record with this
        source_id already exists (idempotent no-op)."""
        if record.policy_version != self.policy_version:
            raise ValueError(
                f"record policy_version {record.policy_version!r} does not match "
                f"store policy_version {self.policy_version!r} -- refusing to pool evidence across versions"
            )
        if record.source_id in self.existing_source_ids():
            return False
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return True

    def load_all(self) -> list[dict]:
        """Deterministic replay: rows in append order, exactly as written."""
        return self._load_existing()

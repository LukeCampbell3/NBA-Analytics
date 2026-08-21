from __future__ import annotations

"""Internal research/debug comparison artifact (mission section 16) --
NOT authorization logic. Answers: does the new theory-driven system
select differently from the old one? When do they agree? Immutable once
the slate decision is frozen (write-once; a second write for the same
date+policy_version is refused, matching evidence_store's idempotency
discipline elsewhere in this codebase).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .legacy_control import LegacyParlayControl

COMPARISON_ARTIFACT_VERSION = "PARLAY_V2_COMPARISON_V1"


@dataclass(frozen=True)
class ComparisonRecord:
    date: str
    policy_version: str
    old_control_pair: list[dict[str, Any]] | None
    new_v2_candidate: list[dict[str, Any]] | None
    same_pair: bool | None
    old_control_probability: float | None
    new_joint_score: float | None
    old_control_quote: dict[str, Any] | None
    new_quote: dict[str, Any] | None
    new_action: str  # "ACT" | "ABSTAIN"
    new_policy_status: str
    eventual_old_settlement: dict[str, Any] | None = None  # filled in later, append-only (see note below)
    eventual_new_settlement: dict[str, Any] | None = None


def _pair_signature(pair: list[dict[str, Any]] | None) -> frozenset | None:
    if not pair:
        return None
    return frozenset((leg.get("player"), leg.get("target"), leg.get("line")) for leg in pair)


def build_comparison_record(
    *,
    date: str,
    policy_version: str,
    legacy: LegacyParlayControl,
    new_v2_pair: list[dict[str, Any]] | None,
    new_joint_score: float | None,
    new_quote: dict[str, Any] | None,
    new_action: str,
    new_policy_status: str,
) -> ComparisonRecord:
    same_pair = None
    if legacy.old_control_pair is not None and new_v2_pair is not None:
        same_pair = _pair_signature(legacy.old_control_pair) == _pair_signature(new_v2_pair)
    return ComparisonRecord(
        date=date,
        policy_version=policy_version,
        old_control_pair=legacy.old_control_pair,
        new_v2_candidate=new_v2_pair,
        same_pair=same_pair,
        old_control_probability=legacy.old_control_probability,
        new_joint_score=new_joint_score,
        old_control_quote=legacy.old_control_quote,
        new_quote=new_quote,
        new_action=new_action,
        new_policy_status=new_policy_status,
    )


def write_comparison_record(root: Path, record: ComparisonRecord) -> bool:
    """Write-once per (date, policy_version): returns False (no-op) if a
    record already exists for that key -- the slate decision is immutable
    once frozen. Settlement fields are filled by a SEPARATE, explicit
    later call (settle_comparison_record), never by re-writing this
    initial record wholesale."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record.date}_{record.policy_version}.json"
    if path.exists():
        return False
    with open(path, "w") as f:
        json.dump(asdict(record), f, indent=2, sort_keys=True)
    return True


def settle_comparison_record(root: Path, date: str, policy_version: str, *, old_settlement: dict[str, Any] | None, new_settlement: dict[str, Any] | None) -> bool:
    """The only permitted post-freeze mutation: filling in settlement
    outcomes once known, strictly additive (never touches the frozen
    decision-time fields). Returns False if the record doesn't exist yet."""
    root = Path(root)
    path = root / f"{date}_{policy_version}.json"
    if not path.exists():
        return False
    with open(path) as f:
        payload = json.load(f)
    if old_settlement is not None:
        payload["eventual_old_settlement"] = old_settlement
    if new_settlement is not None:
        payload["eventual_new_settlement"] = new_settlement
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return True

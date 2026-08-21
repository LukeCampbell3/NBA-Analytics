from __future__ import annotations

"""CURRENT-DAY FREEZE BOUNDARY (mission section 11) -- a one-way marker
for `prospective_start_timestamp`.

Any slate inspected, or any policy/integration decision made, BEFORE this
marker is set counts only as DEVELOPMENT/SHADOW -- never as confirmatory
prospective evidence, even if it happens to look like it would have
passed. This module makes that boundary an explicit, auditable file
write rather than a constant edited in the frozen manifest (editing a
"frozen" source file to backdate or set a timestamp would defeat the
point of freezing it).

Setting the marker is ONE-WAY: once set for a given policy_version, it
cannot be moved earlier or re-set. A genuinely new attempt (new
eligibility/policy parameters) is a new policy_version with its own
marker, per the version-isolation discipline already used everywhere
else in this package.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROSPECTIVE_BOUNDARY_VERSION = "PROSPECTIVE_BOUNDARY_V1"


@dataclass(frozen=True)
class ProspectiveBoundary:
    policy_version: str
    prospective_start_timestamp_utc: str
    note: str


def _path_for(root: Path, policy_version: str) -> Path:
    return Path(root) / f"{policy_version}_prospective_start.json"


def read_prospective_start_timestamp(root: Path, policy_version: str) -> str | None:
    path = _path_for(root, policy_version)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["prospective_start_timestamp_utc"]


def set_prospective_start_timestamp(root: Path, policy_version: str, timestamp_utc: str, *, note: str = "") -> bool:
    """Returns True if set, False if a marker already exists for this
    policy_version (refuses to move it -- one-way). Callers MUST have
    already frozen every other manifest constant for this policy_version
    before calling this; this module does not check that itself (it has
    no visibility into manifest.py's review process), but the ordering is
    load-bearing and documented in manifest.py's CONCLUSION_REASONING."""
    path = _path_for(root, policy_version)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    boundary = ProspectiveBoundary(policy_version=policy_version, prospective_start_timestamp_utc=timestamp_utc, note=note)
    with open(path, "w") as f:
        json.dump(asdict(boundary), f, indent=2, sort_keys=True)
    return True


def is_prospective(root: Path, policy_version: str, slate_decision_timestamp_utc: str) -> bool:
    """A slate's decision counts as prospective evidence iff a boundary is
    set for this policy_version AND the slate's own decision timestamp is
    at or after it. Everything before the boundary (or with no boundary
    set at all) is DEVELOPMENT/SHADOW."""
    boundary = read_prospective_start_timestamp(root, policy_version)
    if boundary is None:
        return False
    return str(slate_decision_timestamp_utc) >= str(boundary)

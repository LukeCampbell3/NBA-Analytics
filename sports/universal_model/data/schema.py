"""Universal typed event schema (mission spec section 4).

Every training observation across every sport is represented as one
``UniversalEvent``. This is deliberately NOT a lossy flat schema: sport
identity, target family, and market fields are first-class, and
sport-specific signal lives in the separate namespaced feature system
(see ``feature_registry.py`` / ``compiler.py``), not jammed into this
dataclass.

``UniversalEvent`` carries only the fields the mission spec calls out as
required identity/target/market fields. All *feature values* (universal or
namespaced) attach separately via ``UniversalFeature`` records keyed by
``observation_id`` -- this keeps the event schema stable while the feature
vocabulary grows (adding a sport should not require editing this file).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "1.0.0"


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class UniversalEvent:
    """One universal, cross-sport training/inference observation.

    Field semantics follow mission spec section 4 exactly. ``target`` is the
    raw settled value's identity (e.g. "H", "TB", win/loss line), not a
    normalized target -- normalization is a separate, DERIVE-only fitted
    step (see ``normalization.py``).
    """

    observation_id: str
    sport: str
    league: str
    season: str
    event_id: str
    event_time: str  # ISO8601 UTC, real event start time
    prediction_cutoff_time: str  # ISO8601 UTC; feature/target availability boundary

    entity_id: str
    entity_name: str
    entity_type: str  # e.g. "player" | "team" | "driver" | "golfer"
    team_id: Optional[str]
    opponent_id: Optional[str]
    role: Optional[str]
    position: Optional[str]
    home_away: Optional[str]  # "home" | "away" | "neutral" | None

    target: str  # target/stat identity, e.g. "H", "TB", "spread", "win_probability"
    target_family: str  # coarse grouping used for target-family holdout tests

    market_type: Optional[str]  # e.g. "player_prop", "moneyline", "spread", "outright"
    side: Optional[str]  # "over" | "under" | "home" | "away" | None
    line: Optional[float]
    sportsbook: Optional[str]
    decimal_price: Optional[float]
    american_price: Optional[float]
    market_timestamp: Optional[str]  # ISO8601 UTC, when the quoted price was observed
    no_vig_market_probability: Optional[float]

    actual_value: Optional[float]  # settled real-world value, None until settlement
    binary_result: Optional[int]  # 1/0/None (None = push or unsettled)
    settlement_status: str  # "settled" | "pending" | "void" | "push"

    source: str  # originating dataset, e.g. "mlb_historical_pool_universe_2026"
    source_version: str
    feature_timestamp: str  # ISO8601 UTC; when features were computed for this row

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Mandatory temporal leakage guard at construction time (spec section 9):
        # a target may only be considered settled strictly after cutoff.
        if self.settlement_status == "settled" and self.actual_value is None and self.binary_result is None:
            raise ValueError(
                f"observation {self.observation_id}: settlement_status=settled but no actual_value/binary_result"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UniversalEvent":
        payload = dict(payload)
        payload.pop("schema_version", None)
        return cls(**payload, schema_version=SCHEMA_VERSION)

    def row_key(self) -> str:
        """Grouping key for chronological/leakage-safe splitting.

        Splitting MUST be done at the whole-event level (spec section 9/10):
        every observation belonging to the same real-world sporting event
        must land on the same side of every split boundary. Using
        ``(sport, event_id)`` rather than ``observation_id`` is what
        enforces that.
        """
        return f"{self.sport}:{self.event_id}"


@dataclass(frozen=True)
class UniversalFeature:
    """One typed feature value attached to a ``UniversalEvent``.

    Two-level feature system (spec section 5): ``namespace`` is either
    "universal" (Level A semantic families shared across sports) or a sport
    code like "mlb"/"nba" (Level B namespaced sport-specific features, e.g.
    ``mlb.batting_order``).
    """

    observation_id: str
    namespace: str  # "universal" | "mlb" | "nba" | "nfl" | "f1" | "golf"
    semantic_family: str  # e.g. "opportunity", "usage", "role_state", "market_state"
    feature_name: str  # fully qualified, e.g. "mlb.batting_order" or "opportunity.volume"
    feature_type: str  # "numeric" | "categorical" | "boolean" | "timestamp"
    value: Optional[Any]
    missing: bool
    timestamp: str  # ISO8601 UTC; when this feature value was known/available
    provenance: str  # "observed" | "derived" | "reconstructed" | "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schema_hash() -> str:
    """Stable hash of the schema's field identity, used in dataset/split
    manifests so caches invalidate whenever the schema changes shape."""
    fields = {
        "UniversalEvent": sorted(UniversalEvent.__dataclass_fields__.keys()),
        "UniversalFeature": sorted(UniversalFeature.__dataclass_fields__.keys()),
        "schema_version": SCHEMA_VERSION,
    }
    blob = json.dumps(fields, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


TARGET_FAMILIES = {
    # Coarse target-family grouping used for target-family holdout tests
    # (spec section 11.D) and masked multi-task loss routing (section 16).
    "batting_volume": {"H", "TB", "R", "RBI"},
    "power": {"HR"},
    "pitching": {"K", "ER"},
    "game_outcome": {"win_probability", "moneyline", "spread", "total"},
    "fantasy_points": {"fantasy_points"},
    "field_finish": {"top5", "top10", "top20", "make_cut", "winner"},
    "race_finish": {"win", "podium", "points_finish"},
}


def target_family_for(sport: str, target: str) -> str:
    for family, members in TARGET_FAMILIES.items():
        if target in members:
            return family
    return "unmapped"

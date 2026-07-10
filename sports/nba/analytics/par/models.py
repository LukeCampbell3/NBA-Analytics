"""Typed records for the PAR atom ledger and player outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ReplacementBaseline:
    season: str
    role: str
    atom_type: str
    sample_size: int
    replacement_value: float
    uncertainty: float
    baseline_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValueAtom:
    atom_id: str
    possession_id: str
    game_id: str
    event_time: str
    season: str
    player_id: str
    team_id: str
    opponent_id: str
    primary_value_label: str
    category: str
    overlap_group_id: str
    context_labels: list[str]
    source_event_ids: list[str]
    source_type: str
    source_tier: str
    raw_value: float
    replacement_baseline: float
    value_above_replacement: float
    reliability_weight: float
    shrinkage_factor: float
    overlap_adjustment: float
    par_value: float
    label_entropy: float
    confidence_tier: str
    player_credit_json: dict[str, Any]
    category_rollup_json: dict[str, Any]
    residual_value: float
    par_model_version: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerMeta:
    player_id: str
    player_name: str
    team_id: str
    team: str
    season: str
    role: str
    minutes: float
    games_played: int
    salary_millions: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

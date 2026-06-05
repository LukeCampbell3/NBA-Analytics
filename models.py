from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlayerInfo:
    id: str
    name: str
    team: str
    season: int
    position: str
    age: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerCard:
    player: dict[str, Any]
    identity: dict[str, Any]
    offense: dict[str, Any]
    defense: dict[str, Any]
    impact: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_player_card(card: dict[str, Any]) -> bool:
    required = {"player", "identity", "offense", "defense", "impact", "metadata"}
    return isinstance(card, dict) and required.issubset(card.keys())

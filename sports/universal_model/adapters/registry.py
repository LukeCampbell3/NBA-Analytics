"""Central adapter registry. Adding a future sport = adding one entry here
plus one adapter module -- the model/training code never imports a
per-sport adapter directly (spec section 40 acceptance condition)."""
from __future__ import annotations

from sports.universal_model.adapters.base import SportAdapter
from sports.universal_model.adapters.f1 import F1Adapter
from sports.universal_model.adapters.golf import GolfAdapter
from sports.universal_model.adapters.mlb import MLBAdapter
from sports.universal_model.adapters.nba import NBAAdapter
from sports.universal_model.adapters.nfl import NFLAdapter

ALL_ADAPTERS: dict[str, type[SportAdapter]] = {
    "mlb": MLBAdapter,
    "nfl": NFLAdapter,
    "nba": NBAAdapter,
    "golf": GolfAdapter,
    "f1": F1Adapter,
}


def build_adapter(sport: str) -> SportAdapter:
    try:
        return ALL_ADAPTERS[sport]()
    except KeyError as exc:
        raise KeyError(f"no adapter registered for sport={sport!r}; known: {sorted(ALL_ADAPTERS)}") from exc

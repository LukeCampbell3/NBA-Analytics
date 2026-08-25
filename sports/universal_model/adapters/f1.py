"""F1 adapter -- implements the SportAdapter contract; reports insufficient
training data honestly.

No ``data/raw`` directory exists for F1 at all (verified directly);
``sports/f1/predictions/data_source.py`` fetches live data with no
persisted historical archive in this repository.
"""
from __future__ import annotations

from pathlib import Path

from sports.universal_model.adapters.base import SourceCoverage, SportAdapter
from sports.universal_model.data.schema import UniversalEvent, UniversalFeature

REPO_ROOT = Path(__file__).resolve().parents[3]
F1_DIR = REPO_ROOT / "sports/f1"


class F1Adapter(SportAdapter):
    sport = "f1"

    def discover_sources(self) -> list[str]:
        raw_dir = F1_DIR / "data" / "raw"
        return [str(raw_dir.relative_to(REPO_ROOT))] if raw_dir.exists() else []

    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        coverage = SourceCoverage(
            sport="f1",
            sufficient_for_training=False,
            event_count=0,
            row_count=0,
            date_span=None,
            reason=(
                "No sports/f1/data/raw directory exists in this repository; predictions/"
                "data_source.py fetches live without a persisted historical archive. "
                "No historical dataset is available to build observations from. "
                "Excluded from DERIVE/SELECT/TEST training."
            ),
        )
        return [], coverage

    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        return events

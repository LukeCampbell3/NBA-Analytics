"""Golf adapter -- implements the SportAdapter contract; reports
insufficient training data honestly.

``sports/golf/data/raw/espn/`` contains only current-tournament leaderboard
snapshots (17 files, 1.3 MB; see reports/INVENTORY.md) -- no historical,
settled, per-golfer outcome ledger. ``sports/golf/predictions/score_model.py``
is a live scoring-projection model, not a training corpus.
"""
from __future__ import annotations

from pathlib import Path

from sports.universal_model.adapters.base import SourceCoverage, SportAdapter
from sports.universal_model.data.schema import UniversalEvent, UniversalFeature

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "sports/golf/data/raw/espn"


class GolfAdapter(SportAdapter):
    sport = "golf"

    def discover_sources(self) -> list[str]:
        return [str(RAW_DIR.relative_to(REPO_ROOT))] if RAW_DIR.exists() else []

    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        snapshot_count = len(list(RAW_DIR.rglob("*.json"))) if RAW_DIR.exists() else 0
        coverage = SourceCoverage(
            sport="golf",
            sufficient_for_training=False,
            event_count=0,
            row_count=snapshot_count,
            date_span=None,
            reason=(
                f"{snapshot_count} live current-tournament leaderboard snapshots exist; no "
                "historical settled per-golfer outcome ledger was found. Excluded from "
                "DERIVE/SELECT/TEST training."
            ),
        )
        return [], coverage

    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        return events

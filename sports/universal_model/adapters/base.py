"""SportAdapter contract (mission spec section 7).

Adding a new sport should require implementing this interface only --
never editing the Transformer/MoE internals (spec section 40, the new-sport
onboarding acceptance test). Each adapter is intentionally thin: it turns
sport-native sources into ``UniversalEvent``/``UniversalFeature`` records
and nothing else. All temporal/leakage discipline is enforced centrally by
``validate_timestamps``/``validate_provenance`` here and re-checked by the
central leakage audit in ``splits.py`` -- an adapter is not trusted to be
the only line of defense.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sports.universal_model.data.schema import UniversalEvent, UniversalFeature


@dataclass(frozen=True)
class SourceCoverage:
    """Honest self-report of what an adapter actually has to work with.

    ``build_observations`` on an insufficient-data sport should not
    silently return an empty list -- it should return a ``SourceCoverage``
    explaining why, so INVENTORY.md-style honesty survives into the
    compiled-dataset stage rather than becoming a mysterious zero-row split.
    """

    sport: str
    sufficient_for_training: bool
    event_count: int
    row_count: int
    date_span: tuple[str, str] | None
    reason: str


class SportAdapter(ABC):
    """Contract every per-sport adapter implements (spec section 7)."""

    sport: str

    @abstractmethod
    def discover_sources(self) -> list[str]:
        """Return the real, on-disk source file(s)/dataset(s) this adapter
        reads from. Used for provenance and for the dataset manifest's
        source hashes."""

    @abstractmethod
    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        """Return every observation this sport can honestly contribute,
        plus a coverage self-report. Returning ``(events, coverage)`` where
        ``coverage.sufficient_for_training`` is False (and ``events`` is
        empty or a small architecture-compatibility sample) is a valid,
        expected result -- see NBA/golf/F1 adapters."""

    @abstractmethod
    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        """Level A: universal semantic-family features (spec section 5)."""

    @abstractmethod
    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        """Level B: namespaced sport-specific features, e.g. ``mlb.batting_order``."""

    @abstractmethod
    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        """Attach/verify target + settlement fields on events (may be a
        no-op if build_observations already populated them)."""

    def validate_timestamps(self, events: Iterable[UniversalEvent]) -> list[str]:
        """Central leakage guard (spec section 9): every event must satisfy
        feature_available_at <= prediction_cutoff_time < target settlement
        (proxied here by feature_timestamp <= prediction_cutoff_time and,
        for settled rows, prediction_cutoff_time <= event_time -- targets
        cannot settle before the event they belong to even starts).
        Returns a list of violation strings; empty list == pass."""
        violations: list[str] = []
        for e in events:
            cutoff = _parse(e.prediction_cutoff_time)
            feat_ts = _parse(e.feature_timestamp)
            event_ts = _parse(e.event_time)
            if feat_ts > cutoff:
                violations.append(
                    f"{e.observation_id}: feature_timestamp ({e.feature_timestamp}) "
                    f"> prediction_cutoff_time ({e.prediction_cutoff_time})"
                )
            if cutoff > event_ts:
                violations.append(
                    f"{e.observation_id}: prediction_cutoff_time ({e.prediction_cutoff_time}) "
                    f"> event_time ({e.event_time})"
                )
        return violations

    def validate_provenance(self, events: Iterable[UniversalEvent]) -> list[str]:
        """Every event must declare a real source/source_version; adapters
        must not emit synthetic rows without labeling them as such."""
        violations: list[str] = []
        for e in events:
            if not e.source or not e.source_version:
                violations.append(f"{e.observation_id}: missing source/source_version")
        return violations


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

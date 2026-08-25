"""NBA adapter -- implements the SportAdapter contract, but reports
insufficient training data honestly rather than fabricating rows.

Real sources exist (``sports/nba/data/raw/espn_team_games/`` -- 81 files;
``sports/nba/predictions/Player-Predictor/`` prop-odds history), but no
compiled, settled, per-observation outcome ledger was found (see
reports/INVENTORY.md): the one candidate,
``sports/nba/validation/production_shadow/player_simulation/backtests/2025_preseason/simulation_backtest_2025_preseason_rows.csv``,
has zero data rows (header only, verified directly).

This adapter still implements every abstract method -- proving the
architecture does not need to change to onboard a new sport (spec section
40) -- but ``build_observations`` returns an empty event list with
``SourceCoverage.sufficient_for_training=False`` and a real reason, rather
than inventing synthetic games to pad it out.
"""
from __future__ import annotations

from pathlib import Path

from sports.universal_model.adapters.base import SourceCoverage, SportAdapter
from sports.universal_model.data.schema import UniversalEvent, UniversalFeature

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "sports/nba/data/raw/espn_team_games"
CANDIDATE_BACKTEST = (
    REPO_ROOT
    / "sports/nba/validation/production_shadow/player_simulation/backtests/2025_preseason/simulation_backtest_2025_preseason_rows.csv"
)


class NBAAdapter(SportAdapter):
    sport = "nba"

    def discover_sources(self) -> list[str]:
        sources = []
        if RAW_DIR.exists():
            sources.append(str(RAW_DIR.relative_to(REPO_ROOT)))
        if CANDIDATE_BACKTEST.exists():
            sources.append(str(CANDIDATE_BACKTEST.relative_to(REPO_ROOT)))
        return sources

    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        raw_file_count = len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0
        backtest_rows = 0
        if CANDIDATE_BACKTEST.exists():
            with CANDIDATE_BACKTEST.open() as f:
                backtest_rows = max(0, sum(1 for _ in f) - 1)  # minus header
        coverage = SourceCoverage(
            sport="nba",
            sufficient_for_training=False,
            event_count=raw_file_count,
            row_count=backtest_rows,
            date_span=None,
            reason=(
                f"{raw_file_count} raw ESPN team-game files exist but are not settled/labeled "
                f"per-observation rows; the one compiled backtest candidate "
                f"({CANDIDATE_BACKTEST.name}) has {backtest_rows} data rows (header only). "
                "No settled, dated, per-observation outcome ledger exists for NBA in this "
                "repository. Excluded from DERIVE/SELECT/TEST training; adapter kept to "
                "demonstrate the SportAdapter contract requires no architecture change per sport."
            ),
        )
        return [], coverage

    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        return events

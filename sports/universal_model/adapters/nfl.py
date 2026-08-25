"""NFL adapter -- secondary sport, small real single-season dataset.

Source: ``sports/nfl/data/evaluation/backtest_rows.csv`` (5,328 rows,
season 2025, weeks 1-18; see reports/INVENTORY.md). Real, dated (by
season/week, not calendar date), settled (``actual`` vs. ``prediction``)
per-player-week observations.

DISCLOSED LIMITATION -- no calendar date at all, only ``season``/``week``.
This adapter maps ``(season, week)`` to a synthetic but monotonic calendar
date (``Sept 1 of season + (week-1)*7 days``) purely so this sport can sit
on the same real-valued timeline as other sports for global-chronological
splitting (spec section 11.B). This is NOT a claim about real kickoff
dates -- within-sport chronological order (season, week) is exact and real;
only the specific calendar date is a placeholder. Flagged in
FINAL_REPORT.md.

Given only 5,328 rows and one season, this adapter is marked
``sufficient_for_training=True`` only in the narrow sense of "usable for
leave-one-sport-out transfer and small-data-regime tests" (spec sections
11.C/11.F) -- NOT as a sport with enough data to support a full
within-sport DERIVE/SELECT/TEST split on its own. See ``splits.py`` for how
this distinction is actually enforced.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sports.universal_model.adapters.base import SourceCoverage, SportAdapter
from sports.universal_model.data.schema import (
    UniversalEvent,
    UniversalFeature,
    target_family_for,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "sports/nfl/data/evaluation/backtest_rows.csv"
SOURCE_NAME = "nfl_backtest_rows"
SOURCE_VERSION = "v1"

_TARGET_TO_FAMILY = {"passing": "fantasy_points", "rushing": "fantasy_points", "receiving": "fantasy_points"}


def _week_date(season: int, week: int) -> datetime:
    base = datetime(int(season), 9, 1, tzinfo=timezone.utc)
    return base + timedelta(days=7 * (int(week) - 1))


class NFLAdapter(SportAdapter):
    sport = "nfl"

    def discover_sources(self) -> list[str]:
        return [str(SOURCE_PATH.relative_to(REPO_ROOT))]

    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        df = pd.read_csv(SOURCE_PATH)
        events: list[UniversalEvent] = []
        for r in df.itertuples(index=False):
            row = r._asdict()
            season, week = int(row["season"]), int(row["week"])
            event_dt = _week_date(season, week)
            cutoff = event_dt - timedelta(hours=1)
            feature_ts = cutoff - timedelta(days=1)
            target = str(row["target"])
            actual = float(row["actual"]) if pd.notna(row["actual"]) else None
            obs_raw = f"nfl|{row['player_id']}|{season}|{week}|{target}"
            observation_id = "nfl:" + hashlib.sha1(obs_raw.encode("utf-8")).hexdigest()[:20]
            events.append(
                UniversalEvent(
                    observation_id=observation_id,
                    sport="nfl",
                    league="NFL",
                    season=str(season),
                    event_id=f"{season}-w{week}-{row['recent_team']}-{row['opponent_team']}",
                    event_time=event_dt.isoformat(),
                    prediction_cutoff_time=cutoff.isoformat(),
                    entity_id=str(row["player_id"]),
                    entity_name=str(row["player_display_name"]),
                    entity_type="player",
                    team_id=str(row["recent_team"]) if pd.notna(row["recent_team"]) else None,
                    opponent_id=str(row["opponent_team"]) if pd.notna(row["opponent_team"]) else None,
                    role=str(row["position"]) if pd.notna(row["position"]) else None,
                    position=str(row["position"]) if pd.notna(row["position"]) else None,
                    home_away=None,
                    target=target,
                    target_family=_TARGET_TO_FAMILY.get(target, target_family_for("nfl", target)),
                    market_type=None,
                    side=None,
                    line=None,
                    sportsbook=None,
                    decimal_price=None,
                    american_price=None,
                    market_timestamp=None,
                    no_vig_market_probability=None,
                    actual_value=actual,
                    binary_result=None,
                    settlement_status="settled" if actual is not None else "pending",
                    source=SOURCE_NAME,
                    source_version=SOURCE_VERSION,
                    feature_timestamp=feature_ts.isoformat(),
                )
            )
        seasons = sorted(df["season"].unique().tolist())
        coverage = SourceCoverage(
            sport="nfl",
            sufficient_for_training=True,
            event_count=int(df.drop_duplicates(["season", "week", "recent_team", "opponent_team"]).shape[0]),
            row_count=len(events),
            date_span=(f"{seasons[0]}-w1", f"{seasons[-1]}-w18") if seasons else None,
            reason=(
                "5,328-row real single-season (2025) settled ledger. Sufficient for "
                "leave-one-sport-out transfer and small-data-regime tests; too small and "
                "single-season to support a standalone within-sport chronological split."
            ),
        )
        return events, coverage

    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        return events

"""MLB adapter -- primary sport, real 242k-row settled/outcome dataset.

Source: ``sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv``
(see reports/INVENTORY.md). Reads pregame-classified columns from
``feature_registry.json`` only; anything classified UNUSABLE/
POSTGAME_FORBIDDEN there is never touched here.

DISCLOSED LIMITATION -- date-level cutoff granularity: ``Commence_Time_UTC``
is 100% missing in this dataset (verified by direct inspection, not
assumed), so real per-game start-of-first-pitch time is not available.
This adapter therefore uses date-level timestamps:

    prediction_cutoff_time = Game_Date 23:59:59.999999Z
    event_time             = (Game_Date + 1 day) 00:00:00Z

i.e. "the whole calendar day is the pregame window; settlement is credited
to the following day." This is intentionally the direction that keeps
correctness (cutoff strictly before settlement) rather than the direction
that would silently discard legitimate same-day market quotes. It means
this adapter CANNOT rule out, at exact-timestamp granularity, a same-day
market quote captured after a given game's actual first pitch (there is no
recorded commence time to check against) -- flagged in FINAL_REPORT.md
rather than hidden. Chronological DERIVE/SELECT/TEST splitting is
unaffected by this, since it only needs Game_Date, which is real.
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
SOURCE_PATH = REPO_ROOT / "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv"
SOURCE_NAME = "mlb_historical_pool_universe_2026"
SOURCE_VERSION = "v1"


def _day_end(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (d + timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)).isoformat()


def _next_day_start(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (d + timedelta(days=1)).isoformat()


def _day_start(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat()


class MLBAdapter(SportAdapter):
    sport = "mlb"

    def discover_sources(self) -> list[str]:
        return [str(SOURCE_PATH.relative_to(REPO_ROOT))]

    def build_observations(self) -> tuple[list[UniversalEvent], SourceCoverage]:
        df = pd.read_csv(SOURCE_PATH, low_memory=False)
        events: list[UniversalEvent] = []
        for row in df.itertuples(index=False):
            r = row._asdict()
            game_date = str(r["Game_Date"])
            cutoff = _day_end(game_date)
            event_time = _next_day_start(game_date)
            feature_ts = _day_start(str(r["Prediction_Run_Date"]))
            # feature_timestamp must be <= cutoff; if the pipeline's run date
            # somehow postdates the game date, clamp rather than fabricate.
            if feature_ts > cutoff:
                feature_ts = cutoff

            result = str(r["Result"]) if pd.notna(r["Result"]) else None
            actual = float(r["Actual"]) if pd.notna(r["Actual"]) else None
            binary_result = None
            settlement_status = "pending"
            if result in ("win", "loss"):
                binary_result = 1 if result == "win" else 0
                settlement_status = "settled"
            elif result == "push":
                settlement_status = "push"

            target = str(r["Target"])
            obs_id_raw = f"mlb|{r['Game_ID']}|{r['Player_ID']}|{target}|{r['Market_Line']}|{game_date}"
            observation_id = "mlb:" + hashlib.sha1(obs_id_raw.encode("utf-8")).hexdigest()[:20]

            over_price_time = r.get("Market_Over_Price_Time")
            market_timestamp = None
            price = None
            if pd.notna(over_price_time):
                try:
                    parsed = datetime.fromisoformat(str(over_price_time).replace("Z", "+00:00"))
                    # Only trust same-day quotes as pregame (see module docstring
                    # limitation); a quote dated on a different day than the
                    # game cannot be verified pregame and is dropped.
                    if parsed.date().isoformat() == game_date:
                        market_timestamp = parsed.isoformat()
                        price = float(r["Market_Over_Price"]) if pd.notna(r["Market_Over_Price"]) else None
                except ValueError:
                    market_timestamp = None

            events.append(
                UniversalEvent(
                    observation_id=observation_id,
                    sport="mlb",
                    league="MLB",
                    season=game_date[:4],
                    event_id=str(r["Game_ID"]),
                    event_time=event_time,
                    prediction_cutoff_time=cutoff,
                    entity_id=str(r["Player_ID"]),
                    entity_name=str(r["Player"]),
                    entity_type="player",
                    team_id=str(r["Team"]) if pd.notna(r["Team"]) else None,
                    opponent_id=str(r["Opponent"]) if pd.notna(r["Opponent"]) else None,
                    role=str(r["Player_Type"]) if pd.notna(r["Player_Type"]) else None,
                    position=None,
                    home_away="home" if r.get("Is_Home") == 1 else ("away" if r.get("Is_Home") == 0 else None),
                    target=target,
                    target_family=target_family_for("mlb", target),
                    market_type="player_prop",
                    side="over",
                    line=float(r["Market_Line"]) if pd.notna(r["Market_Line"]) else None,
                    sportsbook=str(r["Market_Over_Book"]) if pd.notna(r.get("Market_Over_Book")) else None,
                    decimal_price=None,
                    american_price=price,
                    market_timestamp=market_timestamp,
                    no_vig_market_probability=None,
                    actual_value=actual,
                    binary_result=binary_result,
                    settlement_status=settlement_status,
                    source=SOURCE_NAME,
                    source_version=SOURCE_VERSION,
                    feature_timestamp=feature_ts,
                )
            )
        dates = sorted(df["Game_Date"].astype(str).unique())
        coverage = SourceCoverage(
            sport="mlb",
            sufficient_for_training=True,
            event_count=int(df["Game_ID"].nunique()),
            row_count=len(events),
            date_span=(dates[0], dates[-1]) if dates else None,
            reason="242k-row settled, dated, market-priced observation history; sufficient for chronological DERIVE/SELECT/TEST training.",
        )
        return events, coverage

    def map_universal_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        """Level A features not already structurally captured by
        UniversalEvent itself (home_away/role/market_* are schema fields,
        not re-emitted here). Reads the same source in the same row order
        as build_observations() -- 1:1 positional zip with ``events``."""
        df = pd.read_csv(SOURCE_PATH, low_memory=False)
        features: list[UniversalFeature] = []
        for event, row in zip(events, df.itertuples(index=False)):
            r = row._asdict()
            cutoff_dt = datetime.fromisoformat(event.prediction_cutoff_time)
            if pd.notna(r["History_Rows"]):
                features.append(
                    UniversalFeature(
                        observation_id=event.observation_id,
                        namespace="universal",
                        semantic_family="support",
                        feature_name="universal.sample_support_rows",
                        feature_type="numeric",
                        value=float(r["History_Rows"]),
                        missing=False,
                        timestamp=event.feature_timestamp,
                        provenance="observed",
                    )
                )
            last_hist = r.get("Last_History_Date")
            days_since = None
            missing = True
            if pd.notna(last_hist):
                try:
                    parsed = datetime.strptime(str(last_hist), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    days_since = (cutoff_dt - parsed).days
                    missing = False
                except ValueError:
                    pass
            features.append(
                UniversalFeature(
                    observation_id=event.observation_id,
                    namespace="universal",
                    semantic_family="recency",
                    feature_name="universal.days_since_last_history",
                    feature_type="numeric",
                    value=float(days_since) if days_since is not None else None,
                    missing=missing,
                    timestamp=event.feature_timestamp,
                    provenance="derived",
                )
            )
        return features

    def map_namespaced_features(self, events: list[UniversalEvent]) -> list[UniversalFeature]:
        return []

    def build_targets(self, events: list[UniversalEvent]) -> list[UniversalEvent]:
        return events

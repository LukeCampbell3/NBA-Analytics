#!/usr/bin/env python3
"""Provider-neutral MLB player-prop observation contract."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd


PARSER_VERSION = "mlb-odds-parser-v1"
NORMALIZATION_VERSION = "mlb-odds-contract-v1"

CONTRACT_COLUMNS = [
    "source",
    "source_market_id",
    "sportsbook",
    "event_id",
    "external_event_id",
    "player_id",
    "external_player_id",
    "player_name",
    "home_team",
    "away_team",
    "game_start_utc",
    "league",
    "market_type",
    "side",
    "line",
    "price_american",
    "price_decimal",
    "observed_at_utc",
    "source_updated_at_utc",
    "source_url_or_endpoint",
    "acquisition_method",
    "raw_record_hash",
    "parser_version",
    "normalization_version",
    "validation_status",
]

RECONCILIATION_COLUMNS = [
    "source_count",
    "agreeing_source_count",
    "line_disagreement",
    "price_disagreement",
    "timestamp_spread",
    "identity_confidence",
    "canonical_selected",
]

MARKET_ALIASES = {
    "batter_runs": "batter_runs_scored",
    "pitcher_walks_allowed": "pitcher_walks",
    "pitcher_outs_recorded": "pitcher_outs",
}

SUPPORTED_MARKETS = {
    "batter_home_runs", "batter_hits", "batter_total_bases", "batter_rbis",
    "batter_runs_scored", "batter_hits_runs_rbis", "batter_singles", "batter_doubles",
    "batter_triples", "batter_walks", "batter_strikeouts", "batter_stolen_bases",
    "pitcher_strikeouts", "pitcher_hits_allowed", "pitcher_walks", "pitcher_earned_runs",
    "pitcher_outs", "pitcher_pitches_thrown",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def american_to_decimal(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price) or (-100.0 < price < 100.0):
        return None
    if price > 0:
        return round(1.0 + price / 100.0, 6)
    return round(1.0 + 100.0 / abs(price), 6)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _series(df: pd.DataFrame, names: Iterable[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def ensure_contract(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    acquisition_method: str | None = None,
    source_endpoint: str | None = None,
    parser_version: str = PARSER_VERSION,
) -> pd.DataFrame:
    """Convert canonical or legacy provider rows to the common contract."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=CONTRACT_COLUMNS + RECONCILIATION_COLUMNS)

    src = frame.copy().reset_index(drop=True)
    out = pd.DataFrame(index=src.index)
    out["source"] = _series(src, ["source", "provider_name"], source or "")
    if source:
        out["source"] = out["source"].replace("", source).fillna(source)
    out["sportsbook"] = _series(src, ["sportsbook", "book", "bookmaker_key"])
    out["external_event_id"] = _series(src, ["external_event_id", "source_event_id", "game_id", "event_id"])
    out["event_id"] = _series(src, ["event_id", "game_id", "source_event_id", "external_event_id"])
    out["player_name"] = _series(src, ["player_name", "player", "player_name_raw"])
    out["external_player_id"] = _series(src, ["external_player_id", "player_id_source"])
    out["player_id"] = _series(src, ["player_id", "player_id_source"])
    out["home_team"] = _series(src, ["home_team"])
    out["away_team"] = _series(src, ["away_team"])
    out["game_start_utc"] = _series(src, ["game_start_utc", "commence_time_utc"])
    out["league"] = _series(src, ["league"], "MLB").replace("", "MLB").fillna("MLB")
    out["market_type"] = _series(src, ["market_type", "market_canonical", "market_key", "market"])
    out["market_type"] = out["market_type"].replace(MARKET_ALIASES)
    out["side"] = _series(src, ["side"]).astype(str).str.lower()
    out["line"] = pd.to_numeric(_series(src, ["line"]), errors="coerce")
    out["price_american"] = pd.to_numeric(_series(src, ["price_american", "odds"]), errors="coerce")
    supplied_decimal = pd.to_numeric(_series(src, ["price_decimal"], np.nan), errors="coerce")
    derived_decimal = out["price_american"].map(american_to_decimal)
    out["price_decimal"] = supplied_decimal.where(supplied_decimal.notna(), derived_decimal)
    out["observed_at_utc"] = _series(src, ["observed_at_utc", "snapshot_time_utc", "fetched_at_utc"])
    out["source_updated_at_utc"] = _series(
        src,
        ["source_updated_at_utc", "last_update", "observed_at_utc", "snapshot_time_utc", "fetched_at_utc"],
    )
    out["source_url_or_endpoint"] = _series(src, ["source_url_or_endpoint"], source_endpoint or "")
    out["acquisition_method"] = _series(src, ["acquisition_method"], acquisition_method or "api")
    out["parser_version"] = _series(src, ["parser_version"], parser_version).replace("", parser_version)
    out["normalization_version"] = _series(src, ["normalization_version"], NORMALIZATION_VERSION).replace(
        "", NORMALIZATION_VERSION
    )

    out["player_id"] = out["player_id"].where(
        out["player_id"].astype(str).str.strip().ne(""), out["player_name"].map(normalize_identity)
    )
    out["external_player_id"] = out["external_player_id"].where(
        out["external_player_id"].astype(str).str.strip().ne(""), out["player_id"]
    )
    out["event_id"] = out["event_id"].where(
        out["event_id"].astype(str).str.strip().ne(""), out["external_event_id"]
    )
    out["source_market_id"] = _series(src, ["source_market_id", "odd_id"])

    hashes = _series(src, ["raw_record_hash"])
    for idx in out.index:
        if not str(hashes.at[idx] or "").strip():
            hashes.at[idx] = stable_hash(src.iloc[idx].to_dict())
        if not str(out.at[idx, "source_market_id"] or "").strip():
            out.at[idx, "source_market_id"] = stable_hash(
                {
                    "source": out.at[idx, "source"],
                    "event": out.at[idx, "external_event_id"],
                    "player": out.at[idx, "external_player_id"],
                    "market": out.at[idx, "market_type"],
                    "side": out.at[idx, "side"],
                    "line": out.at[idx, "line"],
                    "sportsbook": out.at[idx, "sportsbook"],
                }
            )[:24]
    out["raw_record_hash"] = hashes
    out["validation_status"] = _series(src, ["validation_status"], "UNVALIDATED")

    # Backward-compatible aliases used by the evidence and selector pipelines.
    out["provider_name"] = out["source"]
    out["source_event_id"] = out["external_event_id"]
    out["game_id"] = out["event_id"]
    out["player"] = out["player_name"]
    out["player_id_source"] = out["external_player_id"]
    out["commence_time_utc"] = out["game_start_utc"]
    out["market_canonical"] = out["market_type"]
    out["market"] = out["market_type"]
    out["book"] = out["sportsbook"]
    out["odds"] = out["price_american"]
    out["snapshot_time_utc"] = out["observed_at_utc"]
    return out


def validate_contract(
    frame: pd.DataFrame,
    *,
    max_age_seconds: int,
    now: datetime | None = None,
    reject_started: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Annotate every row and return only fresh, internally valid observations."""
    checked = ensure_contract(frame)
    if checked.empty:
        return checked, {"rows": 0, "valid_rows": 0, "invalid_rows": 0, "rejection_reasons": {"SOURCE_EMPTY": 0}}

    current = pd.Timestamp(now or utc_now())
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    observed = pd.to_datetime(checked["observed_at_utc"], utc=True, errors="coerce")
    updated = pd.to_datetime(checked["source_updated_at_utc"], utc=True, errors="coerce")
    starts = pd.to_datetime(checked["game_start_utc"], utc=True, errors="coerce")
    reasons: list[list[str]] = [[] for _ in checked.index]

    required = [
        "source", "sportsbook", "event_id", "external_event_id", "player_id", "external_player_id",
        "player_name", "home_team", "away_team", "game_start_utc", "market_type", "side", "line",
        "price_american", "price_decimal", "observed_at_utc", "source_updated_at_utc",
        "source_url_or_endpoint", "acquisition_method", "raw_record_hash", "parser_version",
        "normalization_version",
    ]
    for column in required:
        missing = checked[column].isna() | checked[column].astype(str).str.strip().isin({"", "nan", "None"})
        for pos in np.flatnonzero(missing.to_numpy()):
            reasons[pos].append(f"MISSING_{column.upper()}")

    invalid_side = ~checked["side"].isin({"over", "under"})
    invalid_price = ~((checked["price_american"] <= -100) | (checked["price_american"] >= 100))
    invalid_decimal = checked["price_decimal"].isna() | (checked["price_decimal"] <= 1.0)
    expected_decimal = checked["price_american"].map(american_to_decimal)
    price_conflict = expected_decimal.notna() & checked["price_decimal"].notna() & (
        (expected_decimal - checked["price_decimal"]).abs() > 0.02
    )
    unsupported_market = ~checked["market_type"].isin(SUPPORTED_MARKETS)
    stale = observed.isna() | ((current - observed).dt.total_seconds() > max_age_seconds)
    source_stale = updated.isna() | ((current - updated).dt.total_seconds() > max_age_seconds)
    future_observation = observed.notna() & ((observed - current).dt.total_seconds() > 300)
    event_started = starts.notna() & (starts <= current) if reject_started else pd.Series(False, index=checked.index)

    masks = {
        "INVALID_SIDE": invalid_side,
        "INVALID_AMERICAN_PRICE": invalid_price,
        "INVALID_DECIMAL_PRICE": invalid_decimal,
        "PRICE_FORMAT_CONFLICT": price_conflict,
        "UNSUPPORTED_MARKET": unsupported_market,
        "STALE_ODDS": stale | source_stale,
        "FUTURE_OBSERVATION": future_observation,
        "EVENT_STARTED": event_started,
    }
    for reason, mask in masks.items():
        for pos in np.flatnonzero(mask.fillna(True).to_numpy()):
            reasons[pos].append(reason)

    statuses = ["VALID" if not row_reasons else "|".join(sorted(set(row_reasons))) for row_reasons in reasons]
    checked["validation_status"] = statuses
    valid = checked.loc[checked["validation_status"] == "VALID"].copy()
    counts: dict[str, int] = {}
    for row_reasons in reasons:
        for reason in set(row_reasons):
            counts[reason] = counts.get(reason, 0) + 1
    return valid, {
        "rows": int(len(checked)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(checked) - len(valid)),
        "valid_record_rate": float(len(valid) / len(checked)),
        "rejection_reasons": counts,
    }


def reconcile_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve sources, remove exact repeats, and mark the freshest canonical row."""
    if frame.empty:
        return ensure_contract(frame)
    out = ensure_contract(frame)
    out["_observed"] = pd.to_datetime(out["observed_at_utc"], utc=True, errors="coerce")
    source_order = {source: index for index, source in enumerate(out["source"].drop_duplicates().tolist())}
    out["_source_rank"] = out["source"].map(source_order).fillna(len(source_order)).astype(int)
    out["_team_pair"] = out.apply(
        lambda row: "|".join(sorted([normalize_identity(row["home_team"]), normalize_identity(row["away_team"])])),
        axis=1,
    )
    out["_player"] = out["player_name"].map(normalize_identity)
    out["_start_minute"] = pd.to_datetime(out["game_start_utc"], utc=True, errors="coerce").dt.floor("min")
    exact_key = [
        "league", "_start_minute", "_team_pair", "_player", "market_type", "side", "line", "sportsbook",
    ]
    source_key = exact_key + ["source"]
    out = out.sort_values(["_observed", "_source_rank"], kind="stable").drop_duplicates(subset=source_key, keep="last").copy()

    market_key = ["league", "_start_minute", "_team_pair", "_player", "market_type", "side", "sportsbook"]
    line_counts = out.groupby(market_key, dropna=False)["line"].transform("nunique")
    out["line_disagreement"] = line_counts > 1
    out["source_count"] = out.groupby(exact_key, dropna=False)["source"].transform("nunique").astype(int)
    out["agreeing_source_count"] = out.groupby(exact_key + ["price_american"], dropna=False)["source"].transform(
        "nunique"
    ).astype(int)
    out["price_disagreement"] = out.groupby(exact_key, dropna=False)["price_american"].transform("nunique") > 1
    min_ts = out.groupby(exact_key, dropna=False)["_observed"].transform("min")
    max_ts = out.groupby(exact_key, dropna=False)["_observed"].transform("max")
    out["timestamp_spread"] = (max_ts - min_ts).dt.total_seconds().fillna(0.0)
    out["identity_confidence"] = np.where(
        out[["event_id", "player_id", "home_team", "away_team"]].astype(str).apply(lambda col: col.str.strip().ne("" )).all(axis=1),
        "high",
        "medium",
    )
    freshest = out.groupby(exact_key, dropna=False)["_observed"].transform("max")
    candidates = out["_observed"].eq(freshest)
    out["canonical_selected"] = False
    candidate_rows = out.loc[candidates].sort_values("_source_rank", kind="stable").copy()
    selected_indices = candidate_rows.groupby(exact_key, dropna=False, sort=False).head(1).index
    out.loc[selected_indices, "canonical_selected"] = True
    return out.drop(columns=["_observed", "_source_rank", "_team_pair", "_player", "_start_minute"])

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.common import build_candidate_id, safe_float, series_text
from research.market_quality.price_normalization import (
    american_odds_to_break_even as american_odds_to_break_even_fn,
    american_odds_to_decimal as american_odds_to_decimal_fn,
    compute_no_vig_probabilities as compute_no_vig_probabilities_fn,
    price_is_invalid as price_is_invalid_fn,
)


PRICE_SOURCE_TYPES = {
    "LIVE_ENTRY",
    "ARCHIVED_ENTRY",
    "CLOSE_ONLY_DIAGNOSTIC",
    "SYNTHETIC_DIAGNOSTIC",
    "MISSING",
    "UNKNOWN",
}

PRICE_VALIDITY_STATUSES = {
    "PRICE_VALID",
    "MISSING_PRICE",
    "INVALID_PRICE",
    "STALE_PRICE",
    "DIAGNOSTIC_ONLY",
    "PRICE_SOURCE_UNKNOWN",
}

IDENTITY_FIELDS = [
    "candidate_id",
    "game_id",
    "game_date",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "market_type",
    "side",
    "line",
    "book",
    "provider",
    "snapshot_id",
]

SIDE_PRICE_FIELDS = [
    "market_side_price",
    "market_side_break_even",
    "market_side_decimal_odds",
    "market_side_implied_probability",
    "over_price",
    "under_price",
    "over_break_even",
    "under_break_even",
    "no_vig_over_probability",
    "no_vig_under_probability",
]

TIMING_FIELDS = [
    "odds_snapshot_time",
    "prediction_snapshot_time",
    "selector_run_time",
    "market_commence_time_utc",
    "minutes_between_odds_and_prediction",
    "price_staleness_seconds",
    "line_staleness_seconds",
    "explicit_prelock_run_flag",
    "timestamp_safety_basis",
    "timestamp_safety_blocked_reason",
]

SOURCE_FIELDS = [
    "price_source",
    "price_source_type",
    "price_validity_status",
    "diagnostic_only_flag",
    "timestamp_safe_flag",
    "event_time_source",
    "event_time_confidence",
    "event_time_resolution_reason",
    "event_time_resolution_warning",
]

MOVEMENT_FIELDS = [
    "line_at_prediction",
    "line_at_odds_snapshot",
    "line_moved_since_prediction",
    "odds_moved_since_prediction",
    "corrected_price",
    "corrected_break_even",
    "corrected_edge",
    "edge_decay",
]

WARNING_FIELDS = [
    "price_provenance_warning",
    "edge_price_untrusted_flag",
    "stale_price_dependency_candidate_flag",
    "price_gap_blocks_validation_flag",
]

ALL_PRICE_PROVENANCE_FIELDS = (
    IDENTITY_FIELDS
    + SIDE_PRICE_FIELDS
    + TIMING_FIELDS
    + SOURCE_FIELDS
    + MOVEMENT_FIELDS
    + WARNING_FIELDS
)

DEFAULT_PRICE_STALE_SECONDS = 180.0 * 60.0


def american_odds_to_decimal(odds: Any) -> float:
    return american_odds_to_decimal_fn(odds)


def american_odds_to_break_even(odds: Any) -> float:
    return american_odds_to_break_even_fn(odds)


def compute_no_vig_probabilities(over_prices: pd.Series, under_prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    return compute_no_vig_probabilities_fn(over_prices, under_prices)


def price_is_invalid(odds: Any) -> bool:
    return price_is_invalid_fn(odds)


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            out = out.fillna(pd.to_numeric(frame[column], errors="coerce"))
    if np.isnan(default):
        return out
    return out.fillna(float(default))


def _coalesce_text(frame: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].fillna("").astype(str)
        out = np.where(pd.Series(out, index=frame.index).astype(str).eq(""), candidate, out)
        out = pd.Series(out, index=frame.index, dtype="object")
    return out.fillna(default).astype(str)


def _coalesce_bool(frame: pd.DataFrame, columns: list[str], default: bool = False) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype=bool)
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        if values.dtype == bool:
            out = out | values.fillna(False)
            continue
        normalized = values.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
        out = out | normalized
    return out.astype(bool)


def _derive_source_type(
    *,
    price_source: pd.Series,
    source_hint: pd.Series,
    market_side_price: pd.Series,
    odds_snapshot_time: pd.Series,
) -> pd.Series:
    source_text = price_source.fillna("").astype(str).str.lower()
    hint_text = source_hint.fillna("").astype(str).str.lower()
    out = pd.Series("UNKNOWN", index=price_source.index, dtype="object")
    close_mask = source_text.str.contains("close", na=False) | hint_text.str.contains("close", na=False)
    synthetic_mask = source_text.str.contains("synthetic", na=False) | hint_text.str.contains("synthetic", na=False)
    missing_mask = market_side_price.isna()
    archived_mask = odds_snapshot_time.notna()
    live_mask = source_text.str.contains("live_entry", na=False)
    out = out.mask(missing_mask, "MISSING")
    out = out.mask(archived_mask & ~missing_mask, "ARCHIVED_ENTRY")
    out = out.mask(live_mask & ~missing_mask, "LIVE_ENTRY")
    out = out.mask(synthetic_mask, "SYNTHETIC_DIAGNOSTIC")
    out = out.mask(close_mask, "CLOSE_ONLY_DIAGNOSTIC")
    return out


def _validity_status(
    *,
    market_side_price: pd.Series,
    invalid_price_flag: pd.Series,
    timestamp_safe_flag: pd.Series,
    diagnostic_only_flag: pd.Series,
    price_source_type: pd.Series,
    stale_price_flag: pd.Series,
) -> pd.Series:
    out = pd.Series("PRICE_VALID", index=market_side_price.index, dtype="object")
    out = out.mask(market_side_price.isna(), "MISSING_PRICE")
    out = out.mask(invalid_price_flag, "INVALID_PRICE")
    out = out.mask(price_source_type.eq("UNKNOWN") & market_side_price.notna(), "PRICE_SOURCE_UNKNOWN")
    out = out.mask(diagnostic_only_flag, "DIAGNOSTIC_ONLY")
    out = out.mask((~timestamp_safe_flag | stale_price_flag) & market_side_price.notna() & ~diagnostic_only_flag & ~invalid_price_flag, "STALE_PRICE")
    return out


def _warning_from_status(status: pd.Series, *, timestamp_safe_flag: pd.Series, diagnostic_only_flag: pd.Series) -> pd.Series:
    warning = pd.Series("", index=status.index, dtype="object")
    warning = warning.mask(status.eq("MISSING_PRICE"), "missing_market_side_price")
    warning = warning.mask(status.eq("INVALID_PRICE"), "invalid_market_side_price")
    warning = warning.mask(status.eq("PRICE_SOURCE_UNKNOWN"), "unknown_price_source")
    warning = warning.mask(status.eq("DIAGNOSTIC_ONLY"), "diagnostic_only_price_source")
    warning = warning.mask(status.eq("STALE_PRICE") & ~timestamp_safe_flag, "timestamp_unsafe_price")
    warning = warning.mask(status.eq("STALE_PRICE") & timestamp_safe_flag, "stale_price_snapshot")
    warning = warning.mask(diagnostic_only_flag & warning.eq(""), "diagnostic_only_price_source")
    return warning


def ensure_price_provenance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ALL_PRICE_PROVENANCE_FIELDS:
        if column in out.columns:
            continue
        if column.endswith("_flag"):
            out[column] = False
        elif column.endswith("_time"):
            out[column] = pd.NaT
        elif column == "market_commence_time_utc":
            out[column] = pd.NaT
        elif column in {
            "market_side_price",
            "market_side_break_even",
            "market_side_decimal_odds",
            "market_side_implied_probability",
            "over_price",
            "under_price",
            "over_break_even",
            "under_break_even",
            "no_vig_over_probability",
            "no_vig_under_probability",
            "minutes_between_odds_and_prediction",
            "price_staleness_seconds",
            "line_staleness_seconds",
            "line",
            "line_at_prediction",
            "line_at_odds_snapshot",
            "line_moved_since_prediction",
            "odds_moved_since_prediction",
            "corrected_price",
            "corrected_break_even",
            "corrected_edge",
            "edge_decay",
            "player_id",
        }:
            out[column] = np.nan
        elif column == "explicit_prelock_run_flag":
            out[column] = False
        else:
            out[column] = ""
    return out


def load_market_snapshot_manifest(snapshot_path: Path) -> dict[str, Any]:
    candidates = [
        snapshot_path.parent / "latest_manifest.json",
        snapshot_path.parent.parent / "latest_manifest.json",
    ]
    candidates.extend(sorted(snapshot_path.parent.glob("current_market_snapshot_manifest_*.json")))
    candidates.extend(sorted(snapshot_path.parent.glob("current_market_snapshot_*_manifest.json")))
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def derive_snapshot_id(*, provider: str, odds_snapshot_time: Any, fallback_label: str = "") -> str:
    provider_text = str(provider or "").strip() or "unknown_provider"
    odds_time = str(odds_snapshot_time or "").strip()
    if odds_time:
        return f"{provider_text}:{odds_time}"
    fallback = str(fallback_label or "").strip()
    if fallback:
        return f"{provider_text}:{fallback}"
    return provider_text


def annotate_price_provenance_frame(
    frame: pd.DataFrame,
    *,
    stale_seconds_threshold: float = DEFAULT_PRICE_STALE_SECONDS,
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return ensure_price_provenance_columns(out)

    if "candidate_id" not in out.columns and {"target", "direction"}.issubset(set(out.columns)):
        out["candidate_id"] = build_candidate_id(out)
    if "game_id" not in out.columns:
        out["game_id"] = _coalesce_text(out, ["market_event_id", "game_id"])
    if "game_date" not in out.columns:
        out["game_date"] = _coalesce_text(out, ["market_date", "game_date", "run_date"])
    if "player_name" not in out.columns:
        out["player_name"] = _coalesce_text(out, ["player_name", "player", "market_player_raw"])
    if "side" not in out.columns:
        out["side"] = _coalesce_text(out, ["direction", "side"]).str.upper().str.strip()
    else:
        out["side"] = _coalesce_text(out, ["side", "direction"]).str.upper().str.strip()
    if "market_type" not in out.columns and {"target", "direction"}.issubset(set(out.columns)):
        out["market_type"] = series_text(out, "target").str.upper().str.strip() + "_" + series_text(out, "direction").str.upper().str.strip()
    if "line" not in out.columns:
        out["line"] = _coalesce_numeric(out, ["market_line", "line"])
    if "team" not in out.columns:
        out["team"] = _coalesce_text(out, ["team", "actual_team", "market_home_team"])
    if "opponent" not in out.columns:
        out["opponent"] = _coalesce_text(out, ["opponent", "market_away_team"])
    out["provider"] = _coalesce_text(out, ["provider", "market_provider", "snapshot_provider"], default="")
    out["book"] = _coalesce_text(out, ["book", "market_book", "snapshot_book"], default="")
    out.loc[out["book"].eq(""), "book"] = "aggregate_market_snapshot"
    out["snapshot_id"] = _coalesce_text(out, ["snapshot_id", "market_snapshot_id", "market_price_snapshot_id"], default="")

    over_price = _coalesce_numeric(out, ["over_price", "market_over_price", "snapshot_over_price"])
    under_price = _coalesce_numeric(out, ["under_price", "market_under_price", "snapshot_under_price"])
    market_side_price = _coalesce_numeric(out, ["market_side_price", "existing_market_side_price"])
    chosen_from_side = np.where(out["side"].eq("OVER"), over_price, np.where(out["side"].eq("UNDER"), under_price, np.nan))
    market_side_price = market_side_price.where(market_side_price.notna(), pd.Series(chosen_from_side, index=out.index, dtype="float64"))
    market_side_break_even = _coalesce_numeric(out, ["market_side_break_even", "existing_market_side_break_even"])
    market_side_break_even = market_side_break_even.where(market_side_break_even.notna(), market_side_price.map(american_odds_to_break_even))
    market_side_decimal_odds = _coalesce_numeric(out, ["market_side_decimal_odds"])
    market_side_decimal_odds = market_side_decimal_odds.where(market_side_decimal_odds.notna(), market_side_price.map(american_odds_to_decimal))
    market_side_implied_probability = _coalesce_numeric(out, ["market_side_implied_probability", "implied_probability"])
    market_side_implied_probability = market_side_implied_probability.where(market_side_implied_probability.notna(), market_side_break_even)
    over_break_even = _coalesce_numeric(out, ["over_break_even"])
    under_break_even = _coalesce_numeric(out, ["under_break_even"])
    over_break_even = over_break_even.where(over_break_even.notna(), over_price.map(american_odds_to_break_even))
    under_break_even = under_break_even.where(under_break_even.notna(), under_price.map(american_odds_to_break_even))
    no_vig_over, no_vig_under = compute_no_vig_probabilities(over_price, under_price)
    no_vig_over = _coalesce_numeric(out, ["no_vig_over_probability"]).where(
        _coalesce_numeric(out, ["no_vig_over_probability"]).notna(),
        no_vig_over,
    )
    no_vig_under = _coalesce_numeric(out, ["no_vig_under_probability"]).where(
        _coalesce_numeric(out, ["no_vig_under_probability"]).notna(),
        no_vig_under,
    )

    odds_snapshot_time = _normalize_timestamp_series(_coalesce_text(out, ["odds_snapshot_time", "market_fetched_at_utc", "Market_Fetched_At_UTC"]))
    selector_run_time = _normalize_timestamp_series(_coalesce_text(out, ["selector_run_time"]))
    prediction_snapshot_time = _normalize_timestamp_series(_coalesce_text(out, ["prediction_snapshot_time"]))
    prediction_snapshot_time = prediction_snapshot_time.where(prediction_snapshot_time.notna(), selector_run_time)
    reference_time = prediction_snapshot_time.where(prediction_snapshot_time.notna(), selector_run_time)

    price_age_seconds = (reference_time - odds_snapshot_time).dt.total_seconds()
    minutes_between = price_age_seconds / 60.0
    line_at_prediction = _coalesce_numeric(out, ["line_at_prediction", "market_line", "line"])
    line_at_odds_snapshot = _coalesce_numeric(out, ["line_at_odds_snapshot", "snapshot_market_line", "market_line", "line"])
    line_moved = _coalesce_numeric(out, ["line_moved_since_prediction"])
    line_moved = line_moved.where(line_moved.notna(), line_at_odds_snapshot - line_at_prediction)
    odds_moved = _coalesce_numeric(out, ["odds_moved_since_prediction"])
    odds_moved = odds_moved.where(odds_moved.notna(), pd.Series(np.nan, index=out.index, dtype="float64"))

    commence_time_ts = _normalize_timestamp_series(
        _coalesce_text(out, ["market_commence_time_utc", "Market_Commence_Time_UTC", "market_commence_time"])
    )
    source_hint = _coalesce_text(
        out,
        ["price_source_hint", "snapshot_source", "price_source", "price_source_type", "snapshot_price_source_type"],
        default="",
    )
    close_only_flag = source_hint.str.contains("close", case=False, na=False)
    synthetic_flag = source_hint.str.contains("synthetic", case=False, na=False)
    commence_known = commence_time_ts.notna()
    invalid_price_flag = market_side_price.map(price_is_invalid).fillna(False)
    price_source = _coalesce_text(out, ["price_source"], default="")
    explicit_source_type = _coalesce_text(out, ["price_source_type", "snapshot_price_source_type", "market_price_source_type"], default="").str.upper().str.strip()
    derived_source_type = _derive_source_type(
        price_source=price_source,
        source_hint=source_hint,
        market_side_price=market_side_price,
        odds_snapshot_time=odds_snapshot_time,
    )
    source_type = explicit_source_type.where(explicit_source_type.isin(PRICE_SOURCE_TYPES), derived_source_type)
    source_type = source_type.mask(market_side_price.isna(), "MISSING")
    source_type = source_type.mask(synthetic_flag, "SYNTHETIC_DIAGNOSTIC")
    source_type = source_type.mask(close_only_flag, "CLOSE_ONLY_DIAGNOSTIC")
    diagnostic_only_flag = (
        _coalesce_bool(out, ["diagnostic_only_flag"], default=False)
        | close_only_flag
        | synthetic_flag
        | source_type.isin({"CLOSE_ONLY_DIAGNOSTIC", "SYNTHETIC_DIAGNOSTIC"})
    )

    postevent_flag = odds_snapshot_time.notna() & commence_known & odds_snapshot_time.ge(commence_time_ts)
    allowed_entry_source = source_type.isin({"LIVE_ENTRY", "ARCHIVED_ENTRY"})
    explicit_prelock_run_flag = _coalesce_bool(out, ["explicit_prelock_run_flag"], default=False)
    price_prerequisites = (
        odds_snapshot_time.notna()
        & market_side_price.notna()
        & market_side_break_even.notna()
        & allowed_entry_source
        & ~diagnostic_only_flag
        & ~invalid_price_flag
    )
    event_start_verified = price_prerequisites & commence_known & odds_snapshot_time.lt(commence_time_ts)
    prelock_verified = (
        price_prerequisites
        & ~event_start_verified
        & explicit_prelock_run_flag
        & prediction_snapshot_time.notna()
        & odds_snapshot_time.lt(prediction_snapshot_time)
    )
    timestamp_safe_flag = event_start_verified | prelock_verified
    timestamp_safety_basis = pd.Series("NOT_VERIFIED", index=out.index, dtype="object")
    timestamp_safety_basis = timestamp_safety_basis.mask(event_start_verified, "EVENT_START_VERIFIED")
    timestamp_safety_basis = timestamp_safety_basis.mask(prelock_verified, "PRELOCK_RUN_VERIFIED")

    timestamp_safety_blocked_reason = pd.Series("", index=out.index, dtype="object")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(odds_snapshot_time.isna(), "missing_odds_snapshot_time")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(market_side_price.isna(), "missing_market_side_price")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(market_side_break_even.isna(), "missing_market_side_break_even")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(~allowed_entry_source, "price_source_not_live_or_archived_entry")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(diagnostic_only_flag, "diagnostic_only_price_source")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(invalid_price_flag, "invalid_market_side_price")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(postevent_flag, "odds_snapshot_not_before_event_start")
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(
        price_prerequisites & ~commence_known & ~explicit_prelock_run_flag,
        "missing_event_time_and_no_explicit_prelock_run",
    )
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(
        price_prerequisites
        & explicit_prelock_run_flag
        & ~event_start_verified
        & ~prelock_verified
        & prediction_snapshot_time.notna(),
        "prediction_snapshot_not_after_odds_snapshot",
    )
    timestamp_safety_blocked_reason = timestamp_safety_blocked_reason.mask(timestamp_safe_flag, "")

    stale_by_age_flag = timestamp_safe_flag & price_age_seconds.notna() & (price_age_seconds > float(stale_seconds_threshold))
    stale_price_flag = _coalesce_bool(out, ["stale_price_flag"], default=False) | stale_by_age_flag
    price_source = price_source.where(
        price_source.str.strip().ne(""),
        np.where(
            event_start_verified | prelock_verified,
            "current_market_snapshot_pre_event",
            np.where(postevent_flag, "current_market_snapshot_postevent", ""),
        ),
    )
    price_source = pd.Series(price_source, index=out.index, dtype="object")
    diagnostic_only_flag = diagnostic_only_flag | postevent_flag
    validity = _validity_status(
        market_side_price=market_side_price,
        invalid_price_flag=invalid_price_flag,
        timestamp_safe_flag=timestamp_safe_flag,
        diagnostic_only_flag=diagnostic_only_flag,
        price_source_type=source_type,
        stale_price_flag=stale_price_flag,
    )

    correction_supported = (
        timestamp_safe_flag
        & market_side_price.notna()
        & ~diagnostic_only_flag
        & ~invalid_price_flag
        & source_type.ne("UNKNOWN")
    )
    corrected_price = _coalesce_numeric(out, ["corrected_price"])
    corrected_price = corrected_price.where(corrected_price.notna(), market_side_price.where(correction_supported, np.nan))
    corrected_break_even = _coalesce_numeric(out, ["corrected_break_even"])
    corrected_break_even = corrected_break_even.where(corrected_break_even.notna(), corrected_price.map(american_odds_to_break_even))
    stress_probability = _coalesce_numeric(out, ["stress_probability", "expected_win_rate"])
    original_edge = _coalesce_numeric(out, ["original_edge", "edge", "stress_edge"])
    original_edge = original_edge.where(
        original_edge.notna(),
        stress_probability - market_side_break_even,
    )
    corrected_edge = _coalesce_numeric(out, ["corrected_edge"])
    corrected_edge = corrected_edge.where(corrected_edge.notna(), stress_probability - corrected_break_even)
    edge_decay = _coalesce_numeric(out, ["edge_decay"])
    edge_decay = edge_decay.where(edge_decay.notna(), corrected_edge - original_edge)

    edge_price_untrusted_flag = validity.ne("PRICE_VALID")
    stale_candidate_flag = (
        timestamp_safe_flag
        & corrected_break_even.notna()
        & original_edge.notna()
        & corrected_edge.notna()
        & (
            ((original_edge > 0.0) & (corrected_edge <= 0.015))
            | stale_price_flag
            | line_moved.abs().fillna(0.0).gt(0.0)
            | odds_moved.abs().fillna(0.0).gt(0.0)
        )
    )
    price_gap_blocks_validation_flag = (
        market_side_price.isna()
        | market_side_break_even.isna()
        | odds_snapshot_time.isna()
        | price_source.str.strip().eq("")
        | ~timestamp_safe_flag
        | diagnostic_only_flag
        | validity.eq("STALE_PRICE")
        | validity.eq("PRICE_SOURCE_UNKNOWN")
    )
    warning = _warning_from_status(validity, timestamp_safe_flag=timestamp_safe_flag, diagnostic_only_flag=diagnostic_only_flag)

    out["market_side_price"] = market_side_price
    out["market_side_break_even"] = market_side_break_even
    out["market_side_decimal_odds"] = market_side_decimal_odds
    out["market_side_implied_probability"] = market_side_implied_probability
    out["over_price"] = over_price
    out["under_price"] = under_price
    out["over_break_even"] = over_break_even
    out["under_break_even"] = under_break_even
    out["no_vig_over_probability"] = no_vig_over
    out["no_vig_under_probability"] = no_vig_under
    out["odds_snapshot_time"] = odds_snapshot_time
    out["prediction_snapshot_time"] = prediction_snapshot_time
    out["selector_run_time"] = selector_run_time
    out["market_commence_time_utc"] = commence_time_ts
    out["minutes_between_odds_and_prediction"] = minutes_between
    out["price_staleness_seconds"] = price_age_seconds
    out["line_staleness_seconds"] = price_age_seconds
    out["explicit_prelock_run_flag"] = explicit_prelock_run_flag.astype(bool)
    out["timestamp_safety_basis"] = timestamp_safety_basis
    out["timestamp_safety_blocked_reason"] = timestamp_safety_blocked_reason
    out["price_source"] = price_source
    out["price_source_type"] = source_type
    out["price_validity_status"] = validity
    out["diagnostic_only_flag"] = diagnostic_only_flag.astype(bool)
    out["timestamp_safe_flag"] = timestamp_safe_flag.astype(bool)
    out["event_time_source"] = _coalesce_text(out, ["event_time_source"], default="MISSING").replace("", "MISSING")
    out["event_time_confidence"] = _coalesce_text(out, ["event_time_confidence"], default="missing").replace("", "missing")
    out["event_time_resolution_reason"] = _coalesce_text(out, ["event_time_resolution_reason"], default="")
    out["event_time_resolution_warning"] = _coalesce_text(out, ["event_time_resolution_warning"], default="")
    out["line_at_prediction"] = line_at_prediction
    out["line_at_odds_snapshot"] = line_at_odds_snapshot
    out["line_moved_since_prediction"] = line_moved
    out["odds_moved_since_prediction"] = odds_moved
    out["corrected_price"] = corrected_price
    out["corrected_break_even"] = corrected_break_even
    out["corrected_edge"] = corrected_edge
    out["edge_decay"] = edge_decay
    out["price_provenance_warning"] = warning
    out["edge_price_untrusted_flag"] = edge_price_untrusted_flag.astype(bool)
    out["stale_price_dependency_candidate_flag"] = stale_candidate_flag.astype(bool)
    out["price_gap_blocks_validation_flag"] = price_gap_blocks_validation_flag.astype(bool)
    out["snapshot_id"] = _coalesce_text(out, ["snapshot_id"], default="")
    derived_snapshot_ids = pd.Series(
        [
            derive_snapshot_id(provider=provider, odds_snapshot_time=odds_time, fallback_label=game_id)
            for provider, odds_time, game_id in zip(
                out["provider"].tolist(),
                out["odds_snapshot_time"].astype(str).replace("NaT", "").tolist(),
                out["game_id"].tolist(),
            )
        ],
        index=out.index,
        dtype="object",
    )
    out["snapshot_id"] = out["snapshot_id"].where(out["snapshot_id"].ne(""), derived_snapshot_ids)
    return ensure_price_provenance_columns(out)

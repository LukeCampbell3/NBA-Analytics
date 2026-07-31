"""Authentic historical NFL player-prop grading.

The evaluator intentionally does not scrape results into pseudo-lines.  It
accepts normalized sportsbook observations, rejects synthetic sources, joins
only posted player/market rows, and grades the model's side of each line.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MARKET_ALIASES = {
    "passing": "passing",
    "pass": "passing",
    "pass_yds": "passing",
    "passing_yards": "passing",
    "player_pass_yds": "passing",
    "rushing": "rushing",
    "rush": "rushing",
    "rush_yds": "rushing",
    "rushing_yards": "rushing",
    "player_rush_yds": "rushing",
    "receiving": "receiving",
    "rec": "receiving",
    "rec_yds": "receiving",
    "receiving_yards": "receiving",
    "player_reception_yds": "receiving",
    "player_receiving_yds": "receiving",
}

SYNTHETIC_MARKERS = (
    "synthetic",
    "result_derived",
    "actual_as_line",
    "fake",
    "mock",
    "test_fixture",
)

PROVIDER_CLOSING_SOURCES = {"sportsgameodds_historical_close"}


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def normalize_player_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    tokens = re.sub(r"\s+", " ", text).split()
    collapsed: list[str] = []
    initials: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            initials.append(token)
            continue
        if initials:
            collapsed.append("".join(initials))
            initials = []
        collapsed.append(token)
    if initials:
        collapsed.append("".join(initials))
    return " ".join(collapsed)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_market_archive(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source, low_memory=False)


def normalize_market_archive(markets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = markets.copy()
    normalized = pd.DataFrame(index=frame.index)
    normalized["player"] = _first_column(
        frame, ("player", "player_display_name", "Player", "description", "participant")
    )
    normalized["target_raw"] = _first_column(
        frame, ("target", "market", "market_key", "Market", "stat")
    )
    normalized["line"] = pd.to_numeric(
        _first_column(frame, ("line", "point", "market_line", "Line")), errors="coerce"
    )
    normalized["over_price"] = pd.to_numeric(
        _first_column(frame, ("over_price", "price_over", "over_odds")), errors="coerce"
    )
    normalized["under_price"] = pd.to_numeric(
        _first_column(frame, ("under_price", "price_under", "under_odds")), errors="coerce"
    )
    normalized["season"] = pd.to_numeric(
        _first_column(frame, ("season", "Season")), errors="coerce"
    )
    normalized["week"] = pd.to_numeric(
        _first_column(frame, ("week", "Week")), errors="coerce"
    )
    normalized["bookmaker"] = _first_column(
        frame, ("bookmaker", "bookmaker_key", "sportsbook", "book")
    ).fillna("")
    normalized["source"] = _first_column(
        frame, ("source", "provider", "market_source", "Market_Source")
    ).fillna("")
    normalized["snapshot_time_utc"] = pd.to_datetime(
        _first_column(frame, ("snapshot_time_utc", "fetched_at_utc", "timestamp_utc")),
        utc=True,
        errors="coerce",
    )
    normalized["commence_time_utc"] = pd.to_datetime(
        _first_column(frame, ("commence_time_utc", "event_start_utc", "commence_time")),
        utc=True,
        errors="coerce",
    )
    normalized["line_phase"] = _first_column(
        frame, ("line_phase", "market_phase", "odds_phase")
    ).fillna("")
    normalized["pregame_verified"] = _first_column(
        frame, ("pregame_verified", "is_pregame_verified")
    ).map(_truthy)
    normalized["verification_method"] = _first_column(
        frame, ("verification_method", "pregame_evidence")
    ).fillna("")
    normalized["player_key"] = normalized["player"].map(normalize_player_name)
    normalized["target"] = (
        normalized["target_raw"].astype(str).str.strip().str.lower().map(MARKET_ALIASES)
    )

    synthetic = normalized["source"].astype(str).str.lower().apply(
        lambda value: any(marker in value for marker in SYNTHETIC_MARKERS)
    )
    if "is_synthetic" in frame.columns:
        synthetic |= frame["is_synthetic"].fillna(False).astype(bool)
    after_start = (
        normalized["snapshot_time_utc"].notna()
        & normalized["commence_time_utc"].notna()
        & (normalized["snapshot_time_utc"] >= normalized["commence_time_utc"])
    )
    explicit_provider_close = (
        normalized["source"].astype(str).str.lower().isin(PROVIDER_CLOSING_SOURCES)
        & normalized["line_phase"].astype(str).str.lower().eq("closing_pregame")
        & normalized["pregame_verified"]
        & normalized["verification_method"].astype(str).str.lower().eq(
            "provider_explicit_close_fields"
        )
        & normalized["commence_time_utc"].notna()
    )
    missing_required = (
        normalized["player_key"].eq("")
        | normalized["target"].isna()
        | normalized["line"].isna()
        | normalized["season"].isna()
        | normalized["week"].isna()
        | normalized["source"].astype(str).str.strip().eq("")
        | normalized["bookmaker"].astype(str).str.strip().eq("")
    )
    accepted = normalized.loc[~synthetic & ~after_start & ~missing_required].copy()
    accepted["season"] = accepted["season"].astype(int)
    accepted["week"] = accepted["week"].astype(int)
    accepted["timestamp_verified"] = (
        accepted["snapshot_time_utc"].notna()
        & accepted["commence_time_utc"].notna()
        & (accepted["snapshot_time_utc"] < accepted["commence_time_utc"])
    ) | explicit_provider_close.loc[accepted.index]
    accepted = accepted.sort_values("snapshot_time_utc", na_position="first").drop_duplicates(
        ["season", "week", "player_key", "target", "bookmaker"], keep="last"
    )
    audit = {
        "input_rows": int(len(frame)),
        "accepted_market_rows": int(len(accepted)),
        "accepted_pregame_rows": int(len(accepted)),
        "rejected_synthetic_rows": int(synthetic.sum()),
        "rejected_at_or_after_start_rows": int((after_start & ~synthetic).sum()),
        "rejected_missing_contract_rows": int((missing_required & ~synthetic & ~after_start).sum()),
        "timestamp_verified_rows": int(accepted["timestamp_verified"].sum()),
        "provider_closing_verified_rows": int(explicit_provider_close.loc[accepted.index].sum()),
        "source_rows": {
            str(key): int(value)
            for key, value in accepted["source"].value_counts().sort_index().items()
        },
        "bookmaker_rows": {
            str(key): int(value)
            for key, value in accepted["bookmaker"].value_counts().sort_index().items()
        },
    }
    return accepted.reset_index(drop=True), audit


def _american_profit(price: float) -> float:
    if not np.isfinite(price) or price == 0:
        return math.nan
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _wilson_interval(wins: int, losses: int, z: float = 1.96) -> tuple[float | None, float | None]:
    total = wins + losses
    if total == 0:
        return None, None
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    wins = int(group["result"].eq("win").sum())
    losses = int(group["result"].eq("loss").sum())
    pushes = int(group["result"].eq("push").sum())
    lower, upper = _wilson_interval(wins, losses)
    priced = group.loc[group["profit_units"].notna()]
    return {
        "bets": wins + losses + pushes,
        "graded_decisions": wins + losses,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / (wins + losses), 4) if wins + losses else None,
        "hit_rate_wilson_95": [round(lower, 4), round(upper, 4)] if lower is not None else None,
        "priced_bets": int(len(priced)),
        "roi": round(float(priced["profit_units"].sum() / len(priced)), 4) if len(priced) else None,
        "profit_units": round(float(priced["profit_units"].sum()), 4) if len(priced) else None,
    }


def evaluate_market_backtest(
    scored_rows: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    minimum_edge_yards: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    normalized, audit = normalize_market_archive(markets)
    scored = scored_rows.copy()
    scored["player_key"] = scored["player_display_name"].map(normalize_player_name)
    join_keys = ["season", "week", "player_key", "target"]
    joined = scored.merge(normalized, on=join_keys, how="inner", validate="one_to_many")
    joined["edge"] = joined["prediction"].astype(float) - joined["line"].astype(float)
    joined = joined.loc[joined["edge"].abs().ge(float(minimum_edge_yards)) & joined["edge"].ne(0)].copy()
    joined["side"] = np.where(joined["edge"].gt(0), "over", "under")
    actual = joined["actual"].astype(float)
    over = joined["side"].eq("over")
    win = np.where(over, actual > joined["line"], actual < joined["line"])
    push = actual.eq(joined["line"])
    joined["result"] = np.where(push, "push", np.where(win, "win", "loss"))
    joined["selected_price"] = np.where(over, joined["over_price"], joined["under_price"])
    profits = []
    for result, price in zip(joined["result"], joined["selected_price"]):
        if result == "push":
            profits.append(0.0 if np.isfinite(price) else math.nan)
        elif result == "loss":
            profits.append(-1.0 if np.isfinite(price) else math.nan)
        else:
            profits.append(_american_profit(float(price)))
    joined["profit_units"] = profits

    overall = _summary(joined)
    by_target = [
        {"target": str(target), **_summary(group)}
        for target, group in joined.groupby("target", sort=True)
    ]
    by_position = [
        {"position": str(position), **_summary(group)}
        for position, group in joined.groupby("position", sort=True)
    ]
    target_decisions = [item["graded_decisions"] for item in by_target]
    distinct_weeks = int(joined[["season", "week"]].drop_duplicates().shape[0])
    performance_gate_passed = bool(
        overall["graded_decisions"] >= 200
        and len(by_target) == 3
        and min(target_decisions, default=0) >= 50
        and distinct_weeks >= 8
        and overall["hit_rate_wilson_95"]
        and overall["hit_rate_wilson_95"][0] > 0.5
        and overall["roi"] is not None
        and overall["roi"] > 0
        and overall["priced_bets"] == overall["bets"]
    )
    market_gate_passed = bool(
        performance_gate_passed
        and audit["accepted_pregame_rows"] == audit["timestamp_verified_rows"]
    )
    report = {
        "status": "validated" if market_gate_passed else "insufficient_or_failed",
        "minimum_edge_yards": float(minimum_edge_yards),
        "line_audit": audit,
        "matched_market_rows": int(len(joined)),
        "distinct_season_weeks": distinct_weeks,
        "overall": overall,
        "by_target": by_target,
        "by_position": by_position,
        "performance_gate": {
            "status": "passed" if performance_gate_passed else "failed",
            "purpose": (
                "Measures whether the model beat the available historical lines; "
                "it does not waive source-timing requirements for deployment."
            ),
            "criteria": {
                "minimum_overall_graded_decisions": 200,
                "minimum_graded_decisions_per_target": 50,
                "minimum_distinct_season_weeks": 8,
                "overall_hit_rate_wilson_95_lower_bound_above": 0.5,
                "positive_real_price_roi": True,
                "all_graded_rows_have_real_prices": True,
            },
        },
        "promotion_gate": {
            "status": "passed" if market_gate_passed else "failed",
            "criteria": {
                "minimum_overall_graded_decisions": 200,
                "minimum_graded_decisions_per_target": 50,
                "minimum_distinct_season_weeks": 8,
                "overall_hit_rate_wilson_95_lower_bound_above": 0.5,
                "positive_real_price_roi": True,
                "all_graded_rows_have_real_prices": True,
                "all_rows_verified_pregame_or_provider_close": True,
            },
        },
    }
    return report, joined

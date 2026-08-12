"""Frozen NFL live-board and independently gated parlay policy."""

from __future__ import annotations

import itertools
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


POLICY_VERSION = "nfl_passing_loss_aware_meta_policy_v2"
PARLAY_POLICY_VERSION = "nfl_distinct_game_parlay_v1"
VALIDATED_TARGETS = {"passing"}
MINIMUM_SIDE_PROBABILITY = 0.56
MINIMUM_NO_VIG_ADVANTAGE = 0.025
MINIMUM_AMERICAN_PRICE = -150.0
MAXIMUM_AMERICAN_PRICE = 130.0
MAXIMUM_WEEKLY_PICKS = 6
MINIMUM_BOOKS = 2
MINIMUM_COMMON_BOOKS = 1
COMMON_BOOKS = {
    "bet365",
    "betmgm",
    "caesars",
    "draftkings",
    "fanduel",
    "fanatics",
}


def implied_probability(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def american_to_decimal(price: float) -> float:
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def score_market_offers(
    rows: pd.DataFrame,
    over_probability: np.ndarray,
    *,
    now_utc: datetime | None = None,
    max_age_seconds: int = 7_200,
) -> pd.DataFrame:
    scored = rows.copy()
    scored["over_probability"] = np.asarray(over_probability, dtype=float)
    scored["side"] = np.where(scored["over_probability"].ge(0.5), "over", "under")
    scored["estimated_side_probability"] = np.maximum(
        scored["over_probability"], 1.0 - scored["over_probability"]
    )
    over_implied = scored["over_price"].astype(float).map(implied_probability)
    under_implied = scored["under_price"].astype(float).map(implied_probability)
    no_vig_over = over_implied / (over_implied + under_implied)
    scored["no_vig_side_probability"] = np.where(
        scored["side"].eq("over"), no_vig_over, 1.0 - no_vig_over
    )
    scored["probability_advantage"] = (
        scored["estimated_side_probability"] - scored["no_vig_side_probability"]
    )
    scored["selected_price"] = np.where(
        scored["side"].eq("over"), scored["over_price"], scored["under_price"]
    ).astype(float)

    now = now_utc or datetime.now(timezone.utc)
    snapshots = pd.to_datetime(scored["snapshot_time_utc"], utc=True, errors="coerce")
    starts = pd.to_datetime(scored["commence_time_utc"], utc=True, errors="coerce")
    ages = (pd.Timestamp(now) - snapshots).dt.total_seconds()
    scored["price_age_seconds"] = ages
    scored["price_fresh"] = ages.between(0, max_age_seconds, inclusive="both")
    scored["pregame"] = snapshots.notna() & starts.notna() & snapshots.lt(starts)
    scored["model_eligible"] = (
        scored["target"].isin(VALIDATED_TARGETS)
        & scored["estimated_side_probability"].ge(MINIMUM_SIDE_PROBABILITY)
        & scored["probability_advantage"].ge(MINIMUM_NO_VIG_ADVANTAGE)
    )
    scored["execution_eligible"] = (
        scored["selected_price"].between(
            MINIMUM_AMERICAN_PRICE, MAXIMUM_AMERICAN_PRICE, inclusive="both"
        )
        & scored["price_fresh"]
        & scored["pregame"]
    )
    return scored


def select_live_board(scored: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scored.empty:
        return [], {"eligible_offers": 0, "consolidated_candidates": 0}
    frame = scored.copy()
    frame["player_key"] = frame["player_key"].astype(str)
    group_columns = ["event_id", "player_key", "target", "side", "line"]
    plays: list[dict[str, Any]] = []
    for _, group in frame.groupby(group_columns, sort=False):
        meta_eligible = (
            group["meta_eligible"].astype(bool)
            if "meta_eligible" in group
            else pd.Series(False, index=group.index)
        )
        executable = group.loc[
            group["model_eligible"] & group["execution_eligible"] & meta_eligible
        ].copy()
        books = sorted(set(group["bookmaker"].astype(str).str.lower()))
        common_books = sorted(set(books).intersection(COMMON_BOOKS))
        if executable.empty or len(books) < MINIMUM_BOOKS or len(common_books) < MINIMUM_COMMON_BOOKS:
            continue
        best = executable.sort_values(
            ["selected_price", "snapshot_time_utc"], ascending=[False, False]
        ).iloc[0]
        offer_map = {
            str(row.bookmaker).lower(): {
                "price": float(row.selected_price),
                "snapshot_time_utc": str(row.snapshot_time_utc),
            }
            for row in executable.itertuples(index=False)
        }
        plays.append(
            {
                "player": str(best["player"]),
                "player_id": str(best["player_id"]),
                "position": str(best.get("position") or ""),
                "team": str(best.get("recent_team") or ""),
                "opponent": str(best.get("opponent_team") or ""),
                "event_id": str(best["event_id"]),
                "game_start_utc": str(best["commence_time_utc"]),
                "market": str(best["market"]),
                "target": str(best["target"]),
                "direction": str(best["side"]).upper(),
                "line": float(best["line"]),
                "projection": round(float(best["prediction"]), 2),
                "raw_model_probability": round(
                    float(best["raw_model_probability"]), 6
                ),
                "calibrated_hit_probability": round(
                    float(best["calibrated_hit_probability"]), 6
                ),
                "model_hit_probability": round(
                    float(best["calibrated_hit_probability"]), 6
                ),
                "no_vig_probability": round(float(best["no_vig_side_probability"]), 6),
                "probability_advantage": round(float(best["probability_advantage"]), 6),
                "meta_policy_score": round(float(best["meta_policy_score"]), 6),
                "confidence_in_support": bool(best["confidence_in_support"]),
                "selected_side_price": float(best["selected_price"]),
                "selected_sportsbook_key": str(best["bookmaker"]).lower(),
                "market_books": len(books),
                "market_common_books": len(common_books),
                "available_sportsbooks": books,
                "offers": offer_map,
                "market_source": str(best["source"]),
                "price_confirmed": True,
                "snapshot_time_utc": str(best["snapshot_time_utc"]),
                "price_age_seconds": int(max(0, float(best["price_age_seconds"]))),
                "policy_version": POLICY_VERSION,
                "candidate_authorized": False,
                "action_status": "review",
                "risk_flags": ["prospective_certificate_inactive"],
            }
        )
    plays.sort(
        key=lambda row: (
            -row["meta_policy_score"],
            -row["model_hit_probability"],
            -row["probability_advantage"],
            row["player"],
        )
    )
    selected = plays[:MAXIMUM_WEEKLY_PICKS]
    return selected, {
        "eligible_offers": int(
            (
                frame["model_eligible"]
                & frame["execution_eligible"]
                & frame.get("meta_eligible", False)
            ).sum()
        ),
        "consolidated_candidates": len(plays),
        "selected_candidates": len(selected),
    }


def build_shadow_parlay(plays: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct the auditable ticket but withhold it after failed holdout evidence."""

    candidates: list[dict[str, Any]] = []
    for left, right in itertools.combinations(plays, 2):
        if left["event_id"] == right["event_id"] or left["player_id"] == right["player_id"]:
            continue
        books = sorted(set(left.get("offers", {})).intersection(right.get("offers", {})))
        for book in books:
            prices = [left["offers"][book]["price"], right["offers"][book]["price"]]
            decimal = math.prod(american_to_decimal(float(price)) for price in prices)
            probability = float(left["model_hit_probability"]) * float(
                right["model_hit_probability"]
            )
            candidates.append(
                {
                    "sportsbook_key": book,
                    "leg_count": 2,
                    "legs": [left, right],
                    "combined_decimal_price": round(decimal, 4),
                    "projected_probability": round(probability, 6),
                    "expected_return_per_unit": round(probability * decimal - 1.0, 6),
                    "same_sportsbook_confirmed": True,
                    "candidate_authorized": False,
                }
            )
    best = max(
        candidates,
        key=lambda row: (row["projected_probability"], row["expected_return_per_unit"]),
        default=None,
    )
    return {
        "policy_version": PARLAY_POLICY_VERSION,
        "status": "withheld",
        "available": best is not None,
        "selected_ticket": best,
        "validation_status": "failed_locked_holdout",
        "candidate_authorized": False,
        "reason": (
            "The deterministic two-leg rule was 2-16 on the locked 2022 holdout; "
            "the ticket remains shadow-only."
        ),
    }

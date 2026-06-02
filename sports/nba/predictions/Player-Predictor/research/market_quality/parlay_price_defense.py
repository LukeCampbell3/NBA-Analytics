from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .price_normalization import decimal_odds_to_break_even


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _dependency_penalty(legs: list[dict[str, Any]]) -> float:
    if len(legs) <= 1:
        return 0.0
    same_game = len({str(leg.get("game_key") or leg.get("game_id") or leg.get("market_event_id") or "") for leg in legs if str(leg.get("game_key") or leg.get("game_id") or leg.get("market_event_id") or "")}) == 1
    same_team = len({str(leg.get("team") or "") for leg in legs if str(leg.get("team") or "")}) == 1
    same_player = len({str(leg.get("player") or leg.get("player_name") or "") for leg in legs if str(leg.get("player") or leg.get("player_name") or "")}) < len(legs)
    same_target = len({str(leg.get("target") or "") for leg in legs if str(leg.get("target") or "")}) == 1
    same_direction = len({str(leg.get("direction") or leg.get("side") or "") for leg in legs if str(leg.get("direction") or leg.get("side") or "")}) == 1
    penalty = 0.0
    penalty += 0.10 if same_game else 0.0
    penalty += 0.05 if same_team else 0.0
    penalty += 0.12 if same_player else 0.0
    penalty += 0.03 if same_target else 0.0
    penalty += 0.02 if same_direction else 0.0
    return float(np.clip(penalty, 0.0, 0.35))


def evaluate_parlay_price_defense(
    legs: list[dict[str, Any]],
    *,
    parlay_decimal_odds: float | None = None,
    parlay_american_odds: float | None = None,
    price_mode: str | None = None,
) -> dict[str, Any]:
    if not legs:
        return {
            "parlay_price_validity_status": "MISSING_PRICE",
            "parlay_edge_defendability_tier": "EDGE_UNTRUSTED_PRICE",
        }

    leg_decimal = [
        _safe_float(leg.get("market_side_decimal_odds"), default=np.nan)
        for leg in legs
    ]
    if parlay_decimal_odds is None and parlay_american_odds is not None:
        from .price_normalization import american_odds_to_decimal

        parlay_decimal_odds = american_odds_to_decimal(parlay_american_odds)

    explicit_price_mode = str(price_mode or "").strip().upper()
    same_game = len({str(leg.get("game_key") or leg.get("game_id") or leg.get("market_event_id") or "") for leg in legs if str(leg.get("game_key") or leg.get("game_id") or leg.get("market_event_id") or "")}) == 1
    if explicit_price_mode:
        resolved_mode = explicit_price_mode
    elif np.isfinite(_safe_float(parlay_decimal_odds, default=np.nan)):
        resolved_mode = "BOOK_QUOTED_PARLAY"
    elif same_game:
        resolved_mode = "SYNTHETIC_DIAGNOSTIC"
    else:
        resolved_mode = "SYNTHETIC_PRODUCT"

    if not np.isfinite(_safe_float(parlay_decimal_odds, default=np.nan)):
        finite_leg_decimals = [value for value in leg_decimal if np.isfinite(value)]
        if finite_leg_decimals:
            parlay_decimal_odds = float(np.prod(finite_leg_decimals))

    joint_raw = float(np.prod([np.clip(_safe_float(leg.get("model_probability"), default=0.50), 0.0, 1.0) for leg in legs]))
    joint_stress = float(np.prod([np.clip(_safe_float(leg.get("stress_probability"), default=0.50), 0.0, 1.0) for leg in legs]))
    joint_lcb = float(np.prod([np.clip(_safe_float(leg.get("lcb_probability"), default=0.50), 0.0, 1.0) for leg in legs]))
    dependency_penalty = _dependency_penalty(legs)
    joint_raw *= max(0.0, 1.0 - dependency_penalty)
    joint_stress *= max(0.0, 1.0 - dependency_penalty)
    joint_lcb *= max(0.0, 1.0 - dependency_penalty)

    parlay_break_even = decimal_odds_to_break_even(parlay_decimal_odds)
    validity_status = "PRICE_VALID" if resolved_mode == "BOOK_QUOTED_PARLAY" and np.isfinite(_safe_float(parlay_break_even, default=np.nan)) else "DIAGNOSTIC_ONLY"
    tier = "EDGE_DEFENDABLE" if validity_status == "PRICE_VALID" and np.isfinite(parlay_break_even) and joint_lcb > parlay_break_even else "EDGE_DIAGNOSTIC_ONLY"
    if validity_status == "PRICE_VALID" and np.isfinite(parlay_break_even) and joint_lcb <= parlay_break_even:
        tier = "EDGE_FAILS_PRICE"

    payout = _safe_float(parlay_decimal_odds, default=np.nan) - 1.0 if np.isfinite(_safe_float(parlay_decimal_odds, default=np.nan)) else np.nan
    joint_ev = np.nan
    if np.isfinite(payout):
        joint_ev = joint_stress * payout - max(0.0, 1.0 - joint_stress)

    return {
        "parlay_price_mode": resolved_mode,
        "parlay_decimal_odds": _safe_float(parlay_decimal_odds, default=np.nan),
        "parlay_break_even": _safe_float(parlay_break_even, default=np.nan),
        "joint_raw_probability": joint_raw,
        "joint_stress_probability": joint_stress,
        "joint_lcb_probability": joint_lcb,
        "joint_raw_edge": joint_raw - parlay_break_even if np.isfinite(_safe_float(parlay_break_even, default=np.nan)) else np.nan,
        "joint_stress_edge": joint_stress - parlay_break_even if np.isfinite(_safe_float(parlay_break_even, default=np.nan)) else np.nan,
        "joint_lcb_edge": joint_lcb - parlay_break_even if np.isfinite(_safe_float(parlay_break_even, default=np.nan)) else np.nan,
        "joint_ev": joint_ev,
        "parlay_price_validity_status": validity_status,
        "parlay_edge_defendability_tier": tier,
        "parlay_dependency_penalty": dependency_penalty,
    }


def annotate_parlay_payload(
    payload: dict[str, Any],
    *,
    book_quoted_price_field: str = "parlay_decimal_odds",
) -> dict[str, Any]:
    out = {
        "plays": [dict(play) for play in payload.get("plays", [])],
        "pairs": [dict(pair) for pair in payload.get("pairs", [])],
        "summary": dict(payload.get("summary", {})),
    }
    play_lookup = {str(play.get("play_key")): play for play in out["plays"]}
    annotated_pairs: list[dict[str, Any]] = []
    for pair in out["pairs"]:
        legs = []
        for leg in pair.get("legs", []):
            play_key = str(leg.get("play_key") or "")
            if play_key in play_lookup:
                legs.append(play_lookup[play_key])
        defense = evaluate_parlay_price_defense(
            legs,
            parlay_decimal_odds=_safe_float(pair.get(book_quoted_price_field), default=np.nan),
        )
        merged = dict(pair)
        merged.update(defense)
        annotated_pairs.append(merged)
    out["pairs"] = annotated_pairs
    if annotated_pairs:
        defendable = sum(1 for pair in annotated_pairs if pair.get("parlay_edge_defendability_tier") == "EDGE_DEFENDABLE")
        out["summary"]["price_defendable_pairs"] = int(defendable)
        out["summary"]["diagnostic_only_pairs"] = int(
            sum(1 for pair in annotated_pairs if pair.get("parlay_price_validity_status") == "DIAGNOSTIC_ONLY")
        )
    return out

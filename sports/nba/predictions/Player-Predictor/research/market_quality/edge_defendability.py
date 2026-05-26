from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .price_normalization import break_even_to_american_odds


DEFAULT_STRESS_MARGIN = 0.005
DEFAULT_FORECASTABILITY_THRESHOLD = 0.50
DEFAULT_SCENARIO_AGREEMENT_THRESHOLD = 0.45
DEFAULT_MAX_CHAOS_SCORE = 0.60


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _string_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _clip01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def compute_forecastability_score(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric_series(frame, "forecastability_score", np.nan)
    feasibility = _clip01(_numeric_series(frame, "feasibility", 0.50))
    recency = _clip01(_numeric_series(frame, "recency_factor", 0.75))
    belief_conf = _clip01(_numeric_series(frame, "belief_confidence_factor", 0.50))
    coach_trust = _clip01(_numeric_series(frame, "coach_trust_score", 0.50))
    rotation_stability = 1.0 - _clip01(_numeric_series(frame, "rotation_volatility_score", 0.50))
    score = (
        0.28 * feasibility
        + 0.20 * recency
        + 0.20 * belief_conf
        + 0.18 * coach_trust
        + 0.14 * rotation_stability
    ).clip(lower=0.0, upper=1.0)
    return explicit.where(explicit.notna(), score)


def compute_scenario_agreement(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric_series(frame, "scenario_agreement", np.nan)
    model_probability = _clip01(_numeric_series(frame, "model_probability", 0.50))
    stress_probability = _clip01(_numeric_series(frame, "stress_probability", 0.50))
    line_probability = _clip01(
        _numeric_series(
            frame,
            "line_action_expected_win_rate",
            np.nan,
        ).fillna(_numeric_series(frame, "line_chosen_direction_prob", 0.50))
    )
    delta_component = (1.0 - ((model_probability - stress_probability).abs() / 0.25)).clip(lower=0.0, upper=1.0)
    line_component = (1.0 - ((stress_probability - line_probability).abs() / 0.25)).clip(lower=0.0, upper=1.0)
    return explicit.where(explicit.notna(), (0.60 * delta_component + 0.40 * line_component).clip(lower=0.0, upper=1.0))


def compute_chaos_score(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric_series(frame, "chaos_score", np.nan)
    uncertainty = _clip01(
        _numeric_series(frame, "belief_uncertainty_normalized", np.nan).fillna(
            _numeric_series(frame, "belief_uncertainty", 1.0).clip(lower=0.0, upper=1.0)
        )
    )
    fragility = _clip01(_numeric_series(frame, "line_decision_fragility_score", 0.0))
    instability = _clip01(_numeric_series(frame, "line_decision_instability_score", 0.0))
    noise = _clip01(_numeric_series(frame, "noise_score", 0.0))
    contradiction = _clip01(_numeric_series(frame, "contradiction_score", 0.0))
    chaos = (
        0.28 * uncertainty
        + 0.22 * fragility
        + 0.18 * instability
        + 0.16 * noise
        + 0.16 * contradiction
    ).clip(lower=0.0, upper=1.0)
    return explicit.where(explicit.notna(), chaos)


def compute_expected_value(probability: pd.Series, decimal_odds: pd.Series, p_push: pd.Series) -> pd.Series:
    prob = _clip01(probability)
    push = _clip01(p_push)
    odds = pd.to_numeric(decimal_odds, errors="coerce")
    payout = odds - 1.0
    loss_prob = (1.0 - prob - push).clip(lower=0.0, upper=1.0)
    ev = prob * payout - loss_prob
    return ev.where(odds.notna(), np.nan)


def price_required_to_clear_lcb_edge(
    probability: pd.Series,
    p_push: pd.Series,
) -> pd.Series:
    out = []
    for prob, push in zip(probability.tolist(), p_push.tolist()):
        out.append(break_even_to_american_odds(prob, push_probability=push))
    return pd.Series(out, index=probability.index, dtype="float64")


def annotate_edge_defendability(
    frame: pd.DataFrame,
    *,
    stress_margin: float = DEFAULT_STRESS_MARGIN,
    forecastability_threshold: float = DEFAULT_FORECASTABILITY_THRESHOLD,
    scenario_agreement_threshold: float = DEFAULT_SCENARIO_AGREEMENT_THRESHOLD,
    max_chaos_score: float = DEFAULT_MAX_CHAOS_SCORE,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    validity = _string_series(out, "price_validity_status").str.upper().str.strip()
    source_type = _string_series(out, "price_source_type").str.upper().str.strip()
    market_side_break_even = _numeric_series(out, "market_side_break_even", np.nan)
    market_side_decimal_odds = _numeric_series(out, "market_side_decimal_odds", np.nan)
    model_probability = _clip01(_numeric_series(out, "model_probability", 0.50))
    stress_probability = _clip01(_numeric_series(out, "stress_probability", model_probability))
    lcb_probability = _clip01(_numeric_series(out, "lcb_probability", stress_probability))
    p_push = _clip01(_numeric_series(out, "p_push", _numeric_series(out, "expected_push_rate", 0.0)))
    forecastability_score = compute_forecastability_score(out)
    scenario_agreement = compute_scenario_agreement(out)
    chaos_score = compute_chaos_score(out)

    raw_edge = _numeric_series(out, "raw_edge", model_probability - market_side_break_even)
    stress_edge = _numeric_series(out, "stress_edge", stress_probability - market_side_break_even)
    lcb_edge = _numeric_series(out, "lcb_edge", lcb_probability - market_side_break_even)

    minimum_acceptable_odds = price_required_to_clear_lcb_edge(stress_probability, p_push)
    price_required_to_clear = price_required_to_clear_lcb_edge(lcb_probability, p_push)

    raw_ev = _numeric_series(out, "raw_ev", compute_expected_value(model_probability, market_side_decimal_odds, p_push))
    stress_ev = _numeric_series(out, "stress_ev", compute_expected_value(stress_probability, market_side_decimal_odds, p_push))
    lcb_ev = _numeric_series(out, "lcb_ev", compute_expected_value(lcb_probability, market_side_decimal_odds, p_push))

    is_diagnostic = validity.eq("DIAGNOSTIC_ONLY") | source_type.isin({"CLOSE_ONLY_DIAGNOSTIC", "SYNTHETIC_DIAGNOSTIC"})
    is_untrusted = validity.isin({"MISSING_PRICE", "INVALID_PRICE", "STALE_PRICE", "PRICE_SOURCE_UNKNOWN"})
    valid_price = validity.eq("PRICE_VALID")

    tier = pd.Series("EDGE_FAILS_PRICE", index=out.index, dtype="object")
    reason = pd.Series("lcb_edge_not_positive", index=out.index, dtype="object")

    tier = tier.mask(is_untrusted, "EDGE_UNTRUSTED_PRICE")
    reason = reason.mask(validity.eq("MISSING_PRICE"), "missing_timestamp_safe_side_price")
    reason = reason.mask(validity.eq("INVALID_PRICE"), "invalid_market_side_price")
    reason = reason.mask(validity.eq("STALE_PRICE"), "stale_or_timestamp_unsafe_price")
    reason = reason.mask(validity.eq("PRICE_SOURCE_UNKNOWN"), "price_source_unknown")

    tier = tier.mask(is_diagnostic, "EDGE_DIAGNOSTIC_ONLY")
    reason = reason.mask(is_diagnostic & source_type.eq("CLOSE_ONLY_DIAGNOSTIC"), "close_only_price_research_only")
    reason = reason.mask(is_diagnostic & source_type.eq("SYNTHETIC_DIAGNOSTIC"), "synthetic_price_research_only")
    reason = reason.mask(is_diagnostic & reason.eq("lcb_edge_not_positive"), "diagnostic_only_price_source")

    defendable_mask = (
        valid_price
        & market_side_break_even.notna()
        & stress_probability.gt(market_side_break_even + float(stress_margin))
        & lcb_edge.gt(0.0)
        & forecastability_score.ge(float(forecastability_threshold))
        & scenario_agreement.ge(float(scenario_agreement_threshold))
        & chaos_score.le(float(max_chaos_score))
    )
    tier = tier.mask(defendable_mask, "EDGE_DEFENDABLE")
    reason = reason.mask(defendable_mask, "timestamp_safe_price_and_lcb_clear")

    price_dependent_mask = (
        valid_price
        & ~defendable_mask
        & market_side_break_even.notna()
        & (
            raw_edge.gt(0.0)
            | stress_edge.gt(0.0)
            | lcb_probability.gt(0.0)
        )
        & (
            lcb_edge.le(0.0)
            | stress_probability.le(market_side_break_even + float(stress_margin))
        )
    )
    tier = tier.mask(price_dependent_mask, "EDGE_PRICE_DEPENDENT")
    reason = reason.mask(price_dependent_mask, "current_price_does_not_clear_conservative_edge")

    fails_price_mask = valid_price & ~defendable_mask & ~price_dependent_mask
    tier = tier.mask(fails_price_mask, "EDGE_FAILS_PRICE")
    reason = reason.mask(
        fails_price_mask & forecastability_score.lt(float(forecastability_threshold)),
        "forecastability_below_threshold",
    )
    reason = reason.mask(
        fails_price_mask & scenario_agreement.lt(float(scenario_agreement_threshold)),
        "scenario_agreement_below_threshold",
    )
    reason = reason.mask(
        fails_price_mask & chaos_score.gt(float(max_chaos_score)),
        "chaos_score_too_high",
    )
    reason = reason.mask(
        fails_price_mask & reason.eq("lcb_edge_not_positive") & stress_edge.le(0.0),
        "price_break_even_exceeds_stress_probability",
    )

    decision = pd.Series("PASS_AT_PRICE", index=out.index, dtype="object")
    decision = decision.mask(tier.eq("EDGE_DEFENDABLE"), "KEEP")
    decision = decision.mask(tier.eq("EDGE_PRICE_DEPENDENT"), "PRICE_DEPENDENT")
    decision = decision.mask(tier.eq("EDGE_DIAGNOSTIC_ONLY"), "DIAGNOSTIC_ONLY")
    decision = decision.mask(tier.eq("EDGE_UNTRUSTED_PRICE"), "UNTRUSTED")

    out["forecastability_score"] = forecastability_score
    out["scenario_agreement"] = scenario_agreement
    out["chaos_score"] = chaos_score
    out["raw_edge"] = raw_edge
    out["stress_edge"] = stress_edge
    out["lcb_edge"] = lcb_edge
    out["raw_ev"] = raw_ev
    out["stress_ev"] = stress_ev
    out["lcb_ev"] = lcb_ev
    out["minimum_acceptable_odds"] = minimum_acceptable_odds
    out["price_required_to_clear_lcb_edge"] = price_required_to_clear
    out["edge_defendability_tier"] = tier
    out["edge_defendability_reason"] = reason
    out["price_valid_decision"] = decision
    out["would_fail_price_defense"] = tier.isin({"EDGE_UNTRUSTED_PRICE", "EDGE_DIAGNOSTIC_ONLY", "EDGE_FAILS_PRICE"})
    out["would_be_price_dependent"] = tier.eq("EDGE_PRICE_DEPENDENT")
    out["would_be_defendable"] = tier.eq("EDGE_DEFENDABLE")
    out["price_gap_blocks_validation"] = tier.isin({"EDGE_UNTRUSTED_PRICE", "EDGE_DIAGNOSTIC_ONLY"})
    out["diagnostic_only_blocks_validation"] = tier.eq("EDGE_DIAGNOSTIC_ONLY")
    return out

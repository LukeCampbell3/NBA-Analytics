#!/usr/bin/env python3
"""Slate-clustered residual recovery selector (shadow experiment only).

V1 asked whether a local neighborhood had a high absolute hit rate. Its frozen
H OVER 0.5 replay rejected that hypothesis. V2 instead tests the calibration
residual that could justify recovery::

    residual = settled_outcome - balanced_probability

Only strictly earlier, same-family propositions are comparable. Residuals and
hit rates are first averaged within slate, then slates receive equal weight.
The usable correction is a conservative one-sided lower confidence bound on
the mean slate residual, not its point estimate. A separate slate-clustered
hit-rate lower bound caps the recovered probability. When multiple candidate
regions are scanned, select_node() applies Bonferroni simultaneous coverage so
the act of choosing a survivor cannot silently consume the error budget.

This module is deliberately disconnected from publication and v19.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_CONFIDENCE = 0.975
DEFAULT_FAMILYWISE_ALPHA = 0.05


@dataclass(frozen=True)
class ResidualNodeScore:
    candidate_id: str
    family: tuple[str, str, float, str] | None
    neighbor_rows: int
    independent_slates: int
    confidence_level: float
    mean_slate_residual: float | None
    residual_lcb: float | None
    mean_slate_hit_rate: float | None
    support_probability: float | None
    safe_correction: float | None
    recovered_probability: float | None
    expected_value: float | None
    eligible: bool
    reasons: tuple[str, ...]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def american_to_decimal(price: Any) -> float | None:
    number = _finite(price)
    if number is None or (-100.0 < number < 100.0):
        return None
    return 1.0 + (number / 100.0 if number > 0 else 100.0 / abs(number))


def proposition_family(row: dict[str, Any]) -> tuple[str, str, float, str] | None:
    target = str(row.get("target") or "").strip().upper()
    direction = str(row.get("direction") or "").strip().upper()
    line = _finite(row.get("line"))
    if not target or direction not in {"OVER", "UNDER"} or line is None:
        return None
    # If a caller identifies materially different candidate construction, it
    # becomes part of the family. Empty values remain mutually comparable.
    construction = str(row.get("candidate_construction") or "").strip().lower()
    return target, direction, round(line, 6), construction


def _normal_quantile(probability: float) -> float:
    return statistics.NormalDist().inv_cdf(probability)


def _student_t_critical(confidence: float, df: int) -> float:
    """Cornish-Fisher approximation for a one-sided Student-t quantile."""
    if df <= 0:
        return math.inf
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    z = _normal_quantile(confidence)
    inv = 1.0 / df
    return (
        z
        + (z**3 + z) * inv / 4.0
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) * inv**2 / 96.0
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) * inv**3 / 384.0
    )


def one_sided_mean_lcb(values: Iterable[float], *, confidence: float = DEFAULT_CONFIDENCE) -> float | None:
    observations = [float(value) for value in values if _finite(value) is not None]
    if len(observations) < 2:
        return None
    mean = statistics.fmean(observations)
    standard_error = statistics.stdev(observations) / math.sqrt(len(observations))
    return mean - _student_t_critical(confidence, len(observations) - 1) * standard_error


def _distance(
    candidate: dict[str, Any],
    analogue: dict[str, Any],
    *,
    balanced_scale: float,
    market_scale: float,
    market_weight: float,
) -> float | None:
    candidate_balanced = _finite(candidate.get("balanced_probability"))
    candidate_market = _finite(candidate.get("market_probability"))
    analogue_balanced = _finite(analogue.get("balanced_probability"))
    analogue_market = _finite(analogue.get("market_probability"))
    if None in {candidate_balanced, candidate_market, analogue_balanced, analogue_market}:
        return None
    return math.sqrt(
        ((analogue_balanced - candidate_balanced) / balanced_scale) ** 2
        + market_weight * ((analogue_market - candidate_market) / market_scale) ** 2
    )


def score_candidate(
    candidate: dict[str, Any],
    prior_history: Iterable[dict[str, Any]],
    *,
    min_independent_slates: int = 15,
    min_residual_lcb: float = 0.02,
    max_neighbors_per_slate: int = 20,
    max_distance: float = 2.0,
    balanced_scale: float = 0.05,
    market_scale: float = 0.05,
    market_weight: float = 1.0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ResidualNodeScore:
    candidate_id = str(candidate.get("candidate_id") or candidate.get("play_key") or "")
    family = proposition_family(candidate)
    balanced = _finite(candidate.get("balanced_probability"))
    market = _finite(candidate.get("market_probability"))
    reasons: list[str] = []
    if family is None or balanced is None or market is None:
        return ResidualNodeScore(
            candidate_id, family, 0, 0, confidence, None, None, None, None, None, None, None,
            False, ("candidate_evidence_unavailable",),
        )
    if balanced_scale <= 0 or market_scale <= 0 or market_weight < 0:
        raise ValueError("distance scales must be positive and market_weight non-negative")

    candidate_date = str(candidate.get("slate_date") or "")
    by_slate: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for row in prior_history:
        row_date = str(row.get("slate_date") or "")
        if not candidate_date or not row_date or row_date >= candidate_date:
            continue
        if proposition_family(row) != family or row.get("win") not in (0, 1, False, True):
            continue
        distance = _distance(
            candidate,
            row,
            balanced_scale=balanced_scale,
            market_scale=market_scale,
            market_weight=market_weight,
        )
        if distance is None or distance > max_distance:
            continue
        by_slate.setdefault(row_date, []).append((distance, row))

    slate_residuals: list[float] = []
    slate_hit_rates: list[float] = []
    neighbor_rows = 0
    for rows in by_slate.values():
        rows.sort(key=lambda item: item[0])
        neighbors = rows[: max(0, int(max_neighbors_per_slate))]
        if not neighbors:
            continue
        neighbor_rows += len(neighbors)
        residuals = [int(bool(row.get("win"))) - float(row["balanced_probability"]) for _, row in neighbors]
        hits = [int(bool(row.get("win"))) for _, row in neighbors]
        slate_residuals.append(statistics.fmean(residuals))
        slate_hit_rates.append(statistics.fmean(hits))

    independent_slates = len(slate_residuals)
    if independent_slates < min_independent_slates:
        reasons.append("insufficient_independent_slates")

    mean_residual = statistics.fmean(slate_residuals) if slate_residuals else None
    residual_lcb = one_sided_mean_lcb(slate_residuals, confidence=confidence)
    mean_hit_rate = statistics.fmean(slate_hit_rates) if slate_hit_rates else None
    support_lcb = one_sided_mean_lcb(slate_hit_rates, confidence=confidence)
    support_probability = None if support_lcb is None else max(0.0, min(1.0, support_lcb))

    if residual_lcb is None or residual_lcb <= min_residual_lcb:
        reasons.append("residual_lcb_not_meaningful")
    safe_correction = None if residual_lcb is None else max(0.0, residual_lcb)
    recovered = None
    if safe_correction is not None and support_probability is not None:
        recovered = min(1.0, balanced + safe_correction, support_probability)
        if recovered <= balanced:
            reasons.append("support_cap_prevents_recovery")

    decimal = american_to_decimal(candidate.get("price"))
    expected_value = recovered * decimal - 1.0 if recovered is not None and decimal is not None else None
    if expected_value is None:
        reasons.append("price_or_recovered_probability_unavailable")
    elif expected_value <= 0.0:
        reasons.append("recovered_ev_not_positive")

    return ResidualNodeScore(
        candidate_id=candidate_id,
        family=family,
        neighbor_rows=neighbor_rows,
        independent_slates=independent_slates,
        confidence_level=confidence,
        mean_slate_residual=mean_residual,
        residual_lcb=residual_lcb,
        mean_slate_hit_rate=mean_hit_rate,
        support_probability=support_probability,
        safe_correction=safe_correction,
        recovered_probability=recovered,
        expected_value=expected_value,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def select_node(
    candidates: Iterable[dict[str, Any]],
    prior_history: Iterable[dict[str, Any]],
    **score_kwargs: Any,
) -> tuple[dict[str, Any] | None, list[ResidualNodeScore]]:
    candidate_rows = list(candidates)
    history = list(prior_history)
    familywise_alpha = score_kwargs.pop("familywise_alpha", DEFAULT_FAMILYWISE_ALPHA)
    base_confidence = float(score_kwargs.pop("confidence", DEFAULT_CONFIDENCE))
    testable_hypotheses = sum(
        proposition_family(candidate) is not None
        and _finite(candidate.get("balanced_probability")) is not None
        and _finite(candidate.get("market_probability")) is not None
        for candidate in candidate_rows
    )
    if familywise_alpha is None:
        simultaneous_confidence = base_confidence
    else:
        if not 0.0 < float(familywise_alpha) < 0.5:
            raise ValueError("familywise_alpha must be in (0, 0.5) or None")
        simultaneous_confidence = max(
            base_confidence,
            1.0 - float(familywise_alpha) / max(1, testable_hypotheses),
        )
    pairs = [
        (
            candidate,
            score_candidate(candidate, history, confidence=simultaneous_confidence, **score_kwargs),
        )
        for candidate in candidate_rows
    ]
    eligible = [(candidate, score) for candidate, score in pairs if score.eligible]
    if not eligible:
        return None, [score for _, score in pairs]
    eligible.sort(
        key=lambda pair: (
            pair[1].expected_value if pair[1].expected_value is not None else -math.inf,
            pair[1].residual_lcb if pair[1].residual_lcb is not None else -math.inf,
            pair[1].recovered_probability if pair[1].recovered_probability is not None else -math.inf,
        ),
        reverse=True,
    )
    return eligible[0][0], [score for _, score in pairs]

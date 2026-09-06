from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from .schema import (
    AdvancedCandidateContext,
    BatterProcessProfile,
    DirectMatchupProcess,
    PitcherProcessProfile,
    SequentialPAResult,
)

MODEL_VERSION = "sequential_pa_contact_model_v1"
LEAGUE_K_RATE = 0.225
LEAGUE_BB_RATE = 0.085
LEAGUE_HBP_RATE = 0.012
LEAGUE_HR_RATE = 0.030
LEAGUE_CONTACT_XBA = 0.320
LEAGUE_CONTACT_XSLG = 0.510
DEFAULT_TRIALS = 20000
MIN_TRIALS = 5000
MAX_TRIALS = 100000


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _logit(value: float) -> float:
    p = _clip(float(value), 1e-5, 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def matchup_rate(batter_rate: float, pitcher_rate: float, league_rate: float, *, shrink: float = 1.0) -> float:
    """Log5-like probability combination on the logit scale.

    When batter and pitcher both equal the league prior, the result equals the
    league prior. `shrink` controls how strongly the matchup departs from that
    prior when support is weak.
    """
    signal = _logit(batter_rate) + _logit(pitcher_rate) - _logit(league_rate)
    combined = _sigmoid(signal)
    return clip01(league_rate + _clip(shrink, 0.0, 1.0) * (combined - league_rate))


def _weighted_optional(a: float | None, b: float | None, wa: float, wb: float, default: float) -> float:
    values: list[tuple[float, float]] = []
    if a is not None and math.isfinite(float(a)):
        values.append((float(a), max(0.0, wa)))
    if b is not None and math.isfinite(float(b)):
        values.append((float(b), max(0.0, wb)))
    if not values:
        return float(default)
    denom = sum(weight for _, weight in values)
    if denom <= 0:
        return float(default)
    return sum(value * weight for value, weight in values) / denom


def expected_pa_distribution(
    *,
    batting_order: int | None,
    is_home: bool,
    team_expected_runs: float | None,
    pa_share: float | None = None,
) -> dict[int, float]:
    """Pregame PA distribution, separated from per-PA quality.

    This is intentionally conservative and transparent. Batting order drives
    the largest opportunity shift; team scoring environment adjusts only the
    opportunity count, never per-contact hit quality. A supplied recent PA
    share acts as a bounded secondary signal.
    """
    order = int(_clip(float(batting_order or 6), 1.0, 9.0))
    order_bonus = (5.0 - order) * 0.095
    run_env = float(team_expected_runs) if team_expected_runs is not None else 4.5
    run_bonus = _clip((run_env - 4.5) * 0.10, -0.30, 0.35)
    home_penalty = -0.08 if is_home else 0.0
    share_bonus = 0.0
    if pa_share is not None and math.isfinite(float(pa_share)):
        share_bonus = _clip((float(pa_share) - 0.111) * 10.0, -0.25, 0.25)
    mean = _clip(4.18 + order_bonus + run_bonus + home_penalty + share_bonus, 3.15, 5.25)

    values = np.arange(2, 7, dtype=float)
    sigma = 0.72
    weights = np.exp(-0.5 * np.square((values - mean) / sigma))
    weights /= weights.sum()
    return {int(pa): float(prob) for pa, prob in zip(values, weights)}


def pa_event_probabilities(
    batter: BatterProcessProfile,
    pitcher: PitcherProcessProfile,
    *,
    times_through_order: int,
    direct_matchup: DirectMatchupProcess | None = None,
    reliever_blend: float = 0.0,
) -> dict[str, float]:
    support = min(float(batter.support), float(pitcher.support))
    k = matchup_rate(batter.k_rate, pitcher.k_rate, LEAGUE_K_RATE, shrink=0.45 + 0.55 * support)
    bb = matchup_rate(batter.bb_rate, pitcher.bb_rate, LEAGUE_BB_RATE, shrink=0.45 + 0.55 * support)
    hbp = matchup_rate(batter.hbp_rate, pitcher.hbp_rate, LEAGUE_HBP_RATE, shrink=0.35 + 0.45 * support)
    hr = matchup_rate(batter.hr_rate, pitcher.hr_rate, LEAGUE_HR_RATE, shrink=0.40 + 0.55 * support)

    # Later trips through the order generally shift a small amount of mass away
    # from K and toward damage. This is a bounded state effect, not a claim of a
    # universal fixed TTO coefficient.
    tto = max(1, int(times_through_order))
    if tto >= 2:
        k -= 0.008 * (tto - 1)
        hr += 0.0025 * (tto - 1)

    if direct_matchup is not None and direct_matchup.pa > 0:
        w = _clip(direct_matchup.shrinkage_weight, 0.0, 0.45)
        direct_k = direct_matchup.strikeouts / direct_matchup.pa
        direct_bb = direct_matchup.walks / direct_matchup.pa
        direct_hbp = direct_matchup.hbp / direct_matchup.pa
        direct_hr = direct_matchup.home_runs / direct_matchup.pa
        k = (1.0 - w) * k + w * direct_k
        bb = (1.0 - w) * bb + w * direct_bb
        hbp = (1.0 - w) * hbp + w * direct_hbp
        hr = (1.0 - w) * hr + w * direct_hr

    # Once a reliever is likely, shrink pitcher-specific rates toward league
    # average rather than fabricating a named bullpen arm.
    r = _clip(reliever_blend, 0.0, 1.0)
    k = (1.0 - r) * k + r * LEAGUE_K_RATE
    bb = (1.0 - r) * bb + r * LEAGUE_BB_RATE
    hbp = (1.0 - r) * hbp + r * LEAGUE_HBP_RATE
    hr = (1.0 - r) * hr + r * LEAGUE_HR_RATE

    k = _clip(k, 0.02, 0.55)
    bb = _clip(bb, 0.01, 0.25)
    hbp = _clip(hbp, 0.0, 0.06)
    hr = _clip(hr, 0.002, 0.15)
    other = 0.0
    occupied = k + bb + hbp + hr + other
    if occupied >= 0.96:
        scale = 0.96 / occupied
        k, bb, hbp, hr = (value * scale for value in (k, bb, hbp, hr))
    non_hr_contact = 1.0 - (k + bb + hbp + hr + other)
    probs = {"K": k, "BB": bb, "HBP": hbp, "HR": hr, "NON_HR_CONTACT": non_hr_contact, "OTHER": other}
    total = sum(probs.values())
    return {key: value / total for key, value in probs.items()}


def contact_outcome_probabilities(
    batter: BatterProcessProfile,
    pitcher: PitcherProcessProfile,
    *,
    direct_matchup: DirectMatchupProcess | None,
    defense_residual: float,
    park_factor: float,
) -> dict[str, float]:
    bxba = batter.xba if batter.xba is not None else LEAGUE_CONTACT_XBA
    pxba = pitcher.xba_allowed if pitcher.xba_allowed is not None else LEAGUE_CONTACT_XBA
    p_hit = _weighted_optional(bxba, pxba, 0.58, 0.42, LEAGUE_CONTACT_XBA)

    if direct_matchup is not None and direct_matchup.xba_contact is not None:
        w = _clip(direct_matchup.shrinkage_weight, 0.0, 0.35)
        p_hit = (1.0 - w) * p_hit + w * float(direct_matchup.xba_contact)

    # xBA is an average-context expected result. Defense and park are only
    # zero-centered residuals around it, never a second full conversion model.
    park_delta = _clip((float(park_factor) - 1.0) * 0.018, -0.025, 0.025)
    p_hit = _clip(p_hit + _clip(defense_residual, -0.05, 0.05) + park_delta, 0.08, 0.72)

    shares = np.array(
        [
            max(0.01, batter.single_share_non_hr_hits),
            max(0.01, batter.double_share_non_hr_hits),
            max(0.005, batter.triple_share_non_hr_hits),
        ],
        dtype=float,
    )
    shares /= shares.sum()

    # Use xSLG disagreement to tilt the non-HR hit mix toward or away from
    # extra bases without changing the hit probability itself.
    bxslg = batter.xslg if batter.xslg is not None else LEAGUE_CONTACT_XSLG
    pxslg = pitcher.xslg_allowed if pitcher.xslg_allowed is not None else LEAGUE_CONTACT_XSLG
    xslg = _weighted_optional(bxslg, pxslg, 0.58, 0.42, LEAGUE_CONTACT_XSLG)
    power_tilt = _clip((xslg - LEAGUE_CONTACT_XSLG) * 0.35, -0.08, 0.12)
    shares[0] = max(0.01, shares[0] - power_tilt)
    shares[1] = max(0.01, shares[1] + power_tilt * 0.88)
    shares[2] = max(0.005, shares[2] + power_tilt * 0.12)
    shares /= shares.sum()

    p1, p2, p3 = p_hit * shares
    out = 1.0 - p_hit
    probs = {"OUT": float(out), "1B": float(p1), "2B": float(p2), "3B": float(p3), "ROE_OTHER": 0.0}
    total = sum(probs.values())
    return {key: value / total for key, value in probs.items()}


def _starter_reliever_blend(pitcher: PitcherProcessProfile, pa_index: int) -> tuple[int, float]:
    # Batter's first two trips are overwhelmingly starter-facing when the
    # probable starter has normal projected workload. After that, gradually
    # transfer probability mass to a league-average bullpen state instead of
    # pretending the same pitcher faces every PA.
    projected_ip = pitcher.projected_ip if pitcher.projected_ip is not None else 5.4
    tto = 1 if pa_index <= 1 else 2 if pa_index <= 2 else 3
    if pa_index <= 2:
        return tto, 0.0
    if pa_index == 3:
        return tto, _clip((5.2 - projected_ip) * 0.18 + 0.18, 0.08, 0.55)
    if pa_index == 4:
        return 3, _clip((5.8 - projected_ip) * 0.18 + 0.58, 0.35, 0.92)
    return 3, 0.90


def uncertainty_components(context: AdvancedCandidateContext, *, mc_standard_error: float) -> dict[str, float]:
    batter_sample = 1.0 - _clip(context.batter.support, 0.0, 1.0)
    pitcher_sample = 1.0 - _clip(context.pitcher.support, 0.0, 1.0)
    bvp = 1.0 if context.direct_matchup is None else 1.0 - _clip(context.direct_matchup.shrinkage_weight / 0.45, 0.0, 1.0)
    contact_missing = float(any(value is None for value in (context.batter.xba, context.batter.xslg, context.pitcher.xba_allowed, context.pitcher.xslg_allowed)))
    advanced_pitching_missing = float(context.pitcher.xfip is None or context.pitcher.siera is None)
    defense_missing = 1.0 if context.defense_status != "SPECIFIC_DEFENSE_AVAILABLE" else 0.0
    freshness = 0.0 if context.data_freshness_status == "FRESH" else 0.6 if context.data_freshness_status == "DEGRADED" else 1.0
    pa_uncertainty = 0.15 if context.batting_order is not None else 0.55
    mc = _clip(mc_standard_error / 0.01, 0.0, 1.0)
    return {
        "batter_sample": batter_sample,
        "pitcher_sample": pitcher_sample,
        "bvp_sample": bvp,
        "contact_quality_missing": contact_missing,
        "advanced_pitching_missing": advanced_pitching_missing,
        "defense_specificity_missing": defense_missing,
        "data_freshness": freshness,
        "expected_pa": pa_uncertainty,
        "monte_carlo": mc,
    }


def aggregate_uncertainty(parts: Mapping[str, float]) -> float:
    weights = {
        "batter_sample": 0.13,
        "pitcher_sample": 0.13,
        "bvp_sample": 0.05,
        "contact_quality_missing": 0.16,
        "advanced_pitching_missing": 0.10,
        "defense_specificity_missing": 0.07,
        "data_freshness": 0.20,
        "expected_pa": 0.10,
        "monte_carlo": 0.06,
    }
    return _clip(sum(weights.get(key, 0.0) * _clip(value, 0.0, 1.0) for key, value in parts.items()), 0.0, 1.0)


def _seed(context: AdvancedCandidateContext) -> int:
    payload = f"{MODEL_VERSION}|{context.run_date}|{context.game_id}|{context.batter.player_id}|{context.pitcher.player_id}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def _market_clear_from_totals(values: np.ndarray, line: float, side: str) -> float:
    side = str(side).upper()
    if side == "OVER":
        return float(np.mean(values > float(line)))
    return float(np.mean(values < float(line)))


def simulate_hitter_market(
    context: AdvancedCandidateContext,
    *,
    target: str,
    market_line: float,
    side: str = "OVER",
    trials: int = DEFAULT_TRIALS,
    pa_share: float | None = None,
) -> SequentialPAResult:
    trials = int(_clip(int(trials), MIN_TRIALS, MAX_TRIALS))
    rng = np.random.default_rng(_seed(context))
    pa_dist = expected_pa_distribution(
        batting_order=context.batting_order,
        is_home=context.is_home,
        team_expected_runs=context.team_expected_runs,
        pa_share=pa_share,
    )
    pa_values = np.array(sorted(pa_dist), dtype=int)
    pa_probs = np.array([pa_dist[value] for value in pa_values], dtype=float)
    pa_counts = rng.choice(pa_values, size=trials, p=pa_probs)

    hits = np.zeros(trials, dtype=int)
    total_bases = np.zeros(trials, dtype=int)
    home_runs = np.zeros(trials, dtype=int)
    at_bats = np.zeros(trials, dtype=int)
    walks = np.zeros(trials, dtype=int)
    hbp = np.zeros(trials, dtype=int)

    contact_probs = contact_outcome_probabilities(
        context.batter,
        context.pitcher,
        direct_matchup=context.direct_matchup,
        defense_residual=context.defense_residual,
        park_factor=context.park_factor,
    )
    contact_labels = np.array(list(contact_probs.keys()), dtype=object)
    contact_p = np.array(list(contact_probs.values()), dtype=float)

    max_pa = int(pa_counts.max())
    event_diagnostics: list[dict[str, Any]] = []
    for pa_index in range(1, max_pa + 1):
        active = np.flatnonzero(pa_counts >= pa_index)
        if not len(active):
            continue
        tto, reliever_blend = _starter_reliever_blend(context.pitcher, pa_index)
        event_probs = pa_event_probabilities(
            context.batter,
            context.pitcher,
            times_through_order=tto,
            direct_matchup=context.direct_matchup,
            reliever_blend=reliever_blend,
        )
        labels = np.array(list(event_probs.keys()), dtype=object)
        probs = np.array(list(event_probs.values()), dtype=float)
        draws = rng.choice(labels, size=len(active), p=probs)
        event_diagnostics.append({"pa_index": pa_index, "times_through_order": tto, "reliever_blend": reliever_blend, "probabilities": event_probs})

        k_mask = draws == "K"
        bb_mask = draws == "BB"
        hbp_mask = draws == "HBP"
        hr_mask = draws == "HR"
        contact_mask = draws == "NON_HR_CONTACT"
        other_mask = draws == "OTHER"

        at_bats[active[k_mask]] += 1
        walks[active[bb_mask]] += 1
        hbp[active[hbp_mask]] += 1
        at_bats[active[hr_mask]] += 1
        hits[active[hr_mask]] += 1
        home_runs[active[hr_mask]] += 1
        total_bases[active[hr_mask]] += 4
        at_bats[active[other_mask]] += 0

        contact_active = active[contact_mask]
        if len(contact_active):
            contact_draws = rng.choice(contact_labels, size=len(contact_active), p=contact_p)
            at_bats[contact_active] += 1
            one = contact_draws == "1B"
            two = contact_draws == "2B"
            three = contact_draws == "3B"
            hit = one | two | three
            hits[contact_active[hit]] += 1
            total_bases[contact_active[one]] += 1
            total_bases[contact_active[two]] += 2
            total_bases[contact_active[three]] += 3

    p_h0 = float(np.mean(hits == 0))
    p_h1 = float(np.mean(hits == 1))
    p_h2 = float(np.mean(hits >= 2))
    p_tb0 = float(np.mean(total_bases == 0))
    p_tb1 = float(np.mean(total_bases == 1))
    p_tb2 = float(np.mean(total_bases >= 2))
    p_hr = float(np.mean(home_runs >= 1))
    p_h_over = 1.0 - p_h0
    p_tb_over = p_tb2
    target_key = str(target).upper()
    raw = _market_clear_from_totals(hits if target_key == "H" else total_bases, market_line, side)
    mc_se = math.sqrt(max(1e-12, raw * (1.0 - raw) / trials))
    parts = uncertainty_components(context, mc_standard_error=mc_se)
    uncertainty = aggregate_uncertainty(parts)

    # v1 has no independently certified recalibrator yet. Keep calibrated equal
    # to raw and impose only a negative-authority uncertainty haircut. This
    # prevents the new model from acquiring false certainty before prospective
    # calibration evidence exists.
    calibrated = raw
    haircut = min(0.12, 0.16 * uncertainty)
    usable = _clip(calibrated - haircut, 0.01, 0.99)
    lcb = _clip(usable - 1.96 * mc_se, 0.01, 0.99)
    support = min(context.batter.support, context.pitcher.support)
    support_status = "SUPPORTED" if support >= 0.55 and context.data_freshness_status == "FRESH" else "WEAK"

    market_clear = {
        f"H|OVER|0.5": p_h_over,
        f"TB|OVER|1.5": p_tb_over,
        f"{target_key}|{str(side).upper()}|{float(market_line):.1f}": raw,
    }
    return SequentialPAResult(
        model_version=MODEL_VERSION, run_date=context.run_date, game_id=context.game_id,
        player_id=context.batter.player_id, pitcher_id=context.pitcher.player_id, trials=trials,
        expected_pa=float(pa_counts.mean()), expected_ab=float(at_bats.mean()), expected_hits=float(hits.mean()),
        expected_tb=float(total_bases.mean()), pa_distribution={str(key): float(value) for key, value in pa_dist.items()},
        p_h_0=p_h0, p_h_1=p_h1, p_h_ge_2=p_h2, p_tb_0=p_tb0, p_tb_1=p_tb1, p_tb_ge_2=p_tb2,
        p_hr_ge_1=p_hr, hit_over_0_5_probability=p_h_over, tb_over_1_5_probability=p_tb_over,
        market_clear_probabilities=market_clear, probability_standard_error=mc_se,
        raw_structural_probability=raw, calibrated_probability=calibrated, usable_probability=usable,
        probability_lcb=lcb, uncertainty=uncertainty, uncertainty_components=parts, support=float(support),
        support_status=support_status, calibration_status="UNCALIBRATED_NEGATIVE_AUTHORITY_ONLY",
        data_freshness_status=context.data_freshness_status,
        diagnostics={
            "event_tree": "K|BB|HBP|HR|NON_HR_CONTACT|OTHER",
            "contact_tree": "OUT|1B|2B|3B|ROE_OTHER",
            "event_probabilities_by_pa": event_diagnostics,
            "contact_probabilities": contact_probs,
            "walks_per_game": float(walks.mean()),
            "hbp_per_game": float(hbp.mean()),
            "defense_residual": context.defense_residual,
            "defense_status": context.defense_status,
            "missing_components": list(context.missing_components),
        },
    )

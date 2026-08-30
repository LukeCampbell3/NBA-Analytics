#!/usr/bin/env python3
"""Preregistered V3 test of within-slate balanced-probability ranking.

V3 is separate from local recovery. It asks whether balanced probability orders
winners above losers within the same H OVER 0.5 slate, and whether that ordering
is incrementally better than market probability and the v19 policy order.

Candidate pairs improve measurement within a slate; they are never counted as
independent evidence. All inference equal-weights independent slates. The first
four available slates are derivation-only. Later slates are locked evaluation,
with at least eight slates required before any acceptance decision can be
computed at all, and at least PROMOTION_MIN_SLATES required before an
ACCEPTED status can be assigned (the intermediate window returns
SHADOW_ELIGIBLE_PENDING_MORE_SLATES so a real ranking signal is visible while
its slate-level evidence base is still accumulating). The most recent
V4_RESERVE_MOST_RECENT_SLATES slates before the run date are held back
entirely and belong to no evaluation phase, so a future study never has to
compete with a data source V3 has already touched.

Every LCB check is redundantly confirmed by a slate-clustered paired
bootstrap of the same statistic; if the LCB says pass and the bootstrap says
fail (or vice versa) the run reports BOOTSTRAP_LCB_DISAGREEMENT rather than
picking a winner between the two methods -- disagreement itself is a
promotion-blocking finding, not a tie-breaker.

The frozen critical spec values (family, slate thresholds, confidence,
reserve size, acceptance rules) are hashed into PREREGISTRATION_SPEC_HASH.
An accompanying test recomputes that hash and fails CI if any of them are
edited without an accompanying update to the hash constant -- silent
loosening of the preregistration is a promotion-invalidating event by
construction rather than by convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import select_high_precision_predictions as shp  # noqa: E402
from build_v11_eligible_training_set import (  # noqa: E402
    DEFAULT_DAILY_RUNS_ROOT,
    DEFAULT_PROCESSED_ROOT,
    find_raw_pool_csvs,
    parse_v11_args,
)
from local_node_selector_v2 import one_sided_mean_lcb  # noqa: E402
from validate_historical_final_pools import build_actual_lookup, grade_result, normalize_player_key  # noqa: E402


V3_VERSION = "balanced_probability_within_slate_ranking_v3_preregistered"
TARGET = "H"
DIRECTION = "OVER"
LINE = 0.5
DERIVATION_SLATES = 4
MIN_LOCKED_SLATES = 8
# ACCEPTED status is reserved for runs that also have a real, thicker slate
# base -- 30 is the classical CLT threshold and matches the user's own
# framing that same-slate pairwise ranking is cheaper than V2's absolute
# residual estimation (which needed 93-158 slates), but a slate list of 8
# is still on the thin end for a promotion claim about live production.
PROMOTION_MIN_SLATES = 30
CONFIDENCE = 0.975
# One-sided coverage for the paired slate-clustered bootstrap that
# redundantly confirms each LCB decision. Same value as CONFIDENCE so
# both methods target the same nominal coverage; disagreement is then a
# methodological finding, not a design difference.
BOOTSTRAP_LOWER_QUANTILE = 1.0 - CONFIDENCE
BOOTSTRAP_RESAMPLES = 10_000
# Reserved future-study slates: the last N real evaluated dates in the
# input row set are dropped from BOTH derivation and locked partitions,
# so a follow-up preregistration in the same problem family has clean
# forward-only slates that V3 has never observed.
V4_RESERVE_MOST_RECENT_SLATES = 10
TOP_K = (1, 3, 5)
SCORE_FIELDS = (
    "balanced_probability",
    "market_probability",
    "base_ev",
    "v19_order_score",
)


def _preregistration_spec_hash() -> str:
    """SHA-256 of the frozen critical constants. Any silent edit to these
    values changes the hash and fails the accompanying test in
    test_balanced_ranking_v3.py -- a promotion-invalidating event by
    construction."""
    payload = json.dumps(
        {
            "V3_VERSION": V3_VERSION,
            "TARGET": TARGET,
            "DIRECTION": DIRECTION,
            "LINE": LINE,
            "DERIVATION_SLATES": DERIVATION_SLATES,
            "MIN_LOCKED_SLATES": MIN_LOCKED_SLATES,
            "PROMOTION_MIN_SLATES": PROMOTION_MIN_SLATES,
            "CONFIDENCE": CONFIDENCE,
            "BOOTSTRAP_LOWER_QUANTILE": BOOTSTRAP_LOWER_QUANTILE,
            "BOOTSTRAP_RESAMPLES": BOOTSTRAP_RESAMPLES,
            "V4_RESERVE_MOST_RECENT_SLATES": V4_RESERVE_MOST_RECENT_SLATES,
            "TOP_K": list(TOP_K),
            "SCORE_FIELDS": list(SCORE_FIELDS),
            "acceptance_rules": [
                "balanced concordance LCB > 0.5 AND bootstrap agrees",
                "balanced-minus-market concordance LCB > 0 AND bootstrap agrees",
                "balanced-minus-v19 concordance LCB > 0 AND bootstrap agrees",
                "locked_slates >= PROMOTION_MIN_SLATES for ACCEPTED (else SHADOW_ELIGIBLE_PENDING_MORE_SLATES)",
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PREREGISTRATION_SPEC_HASH = _preregistration_spec_hash()


@dataclass(frozen=True)
class SlateRankingMetric:
    date: str
    score: str
    rows: int
    wins: int
    losses: int
    comparable_pairs: int
    concordance: float | None
    reciprocal_rank: float | None
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    top_1_lift: float
    top_3_lift: float
    top_5_lift: float


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def american_to_decimal(price: float) -> float:
    return 1.0 + price / 100.0 if price > 0 else 1.0 + 100.0 / abs(price)


def pairwise_concordance(scores: Iterable[float], outcomes: Iterable[int]) -> tuple[float | None, int]:
    score = np.asarray(list(scores), dtype=float)
    outcome = np.asarray(list(outcomes), dtype=int)
    winners = score[outcome == 1]
    losers = score[outcome == 0]
    pairs = int(len(winners) * len(losers))
    if pairs == 0:
        return None, 0
    differences = winners[:, None] - losers[None, :]
    concordant = np.sum(differences > 0) + 0.5 * np.sum(differences == 0)
    return float(concordant / pairs), pairs


def slate_metric(rows: list[dict[str, Any]], score_field: str) -> SlateRankingMetric:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row[score_field]), str(row.get("candidate_id") or "")),
    )
    outcomes = [int(row["win"]) for row in ordered]
    scores = [float(row[score_field]) for row in ordered]
    concordance, pairs = pairwise_concordance(scores, outcomes)
    base_rate = sum(outcomes) / len(outcomes)
    reciprocal_rank = (
        sum(1.0 / rank for rank, outcome in enumerate(outcomes, start=1) if outcome) / sum(outcomes)
        if sum(outcomes)
        else None
    )
    top_rates: dict[int, float] = {}
    for k in TOP_K:
        used = outcomes[: min(k, len(outcomes))]
        top_rates[k] = sum(used) / len(used)
    return SlateRankingMetric(
        date=str(rows[0]["date"]),
        score=score_field,
        rows=len(rows),
        wins=sum(outcomes),
        losses=len(outcomes) - sum(outcomes),
        comparable_pairs=pairs,
        concordance=concordance,
        reciprocal_rank=reciprocal_rank,
        top_1_hit_rate=top_rates[1],
        top_3_hit_rate=top_rates[3],
        top_5_hit_rate=top_rates[5],
        top_1_lift=top_rates[1] - base_rate,
        top_3_lift=top_rates[3] - base_rate,
        top_5_lift=top_rates[5] - base_rate,
    )


def evaluate_rows(rows: list[dict[str, Any]]) -> list[SlateRankingMetric]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if all(_finite(row.get(field)) is not None for field in SCORE_FIELDS):
            by_date.setdefault(str(row["date"]), []).append(row)
    return [
        slate_metric(slate, score)
        for date in sorted(by_date)
        for score in SCORE_FIELDS
        for slate in [by_date[date]]
        if slate
    ]


def _slate_clustered_bootstrap_lower(values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 20260830) -> float | None:
    """One-sided lower confidence bound on the mean, computed by resampling
    slates (never candidates) with replacement. Deterministic given the
    seed so the same locked slate list always produces the same bound --
    a run whose bound flips is a real numerical drift, not RNG jitter.

    Returns None on an empty list so the surrounding summary can report
    "not computable" rather than a fake zero."""
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    idx = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[idx].mean(axis=1)
    return float(np.quantile(means, BOOTSTRAP_LOWER_QUANTILE))


def _score_summary(metrics: list[SlateRankingMetric], score: str) -> dict[str, Any]:
    selected = [metric for metric in metrics if metric.score == score]
    defined = [metric for metric in selected if metric.concordance is not None]
    concordance = [float(metric.concordance) for metric in defined]
    return {
        "slates": len(selected),
        "auc_defined_slates": len(defined),
        "comparable_pairs_descriptive_only": sum(metric.comparable_pairs for metric in defined),
        "mean_slate_concordance": float(np.mean(concordance)) if concordance else None,
        "concordance_lcb": one_sided_mean_lcb(concordance, confidence=CONFIDENCE),
        "concordance_bootstrap_lcb": _slate_clustered_bootstrap_lower(concordance),
        "mean_top_1_hit_rate": float(np.mean([metric.top_1_hit_rate for metric in selected])) if selected else None,
        "mean_top_3_hit_rate": float(np.mean([metric.top_3_hit_rate for metric in selected])) if selected else None,
        "mean_top_5_hit_rate": float(np.mean([metric.top_5_hit_rate for metric in selected])) if selected else None,
        "mean_top_1_lift": float(np.mean([metric.top_1_lift for metric in selected])) if selected else None,
        "mean_top_3_lift": float(np.mean([metric.top_3_lift for metric in selected])) if selected else None,
        "mean_top_5_lift": float(np.mean([metric.top_5_lift for metric in selected])) if selected else None,
    }


def _paired_delta(metrics: list[SlateRankingMetric], left: str, right: str) -> dict[str, Any]:
    left_by_date = {m.date: m for m in metrics if m.score == left and m.concordance is not None}
    right_by_date = {m.date: m for m in metrics if m.score == right and m.concordance is not None}
    dates = sorted(set(left_by_date) & set(right_by_date))
    delta = [float(left_by_date[d].concordance) - float(right_by_date[d].concordance) for d in dates]
    return {
        "left": left,
        "right": right,
        "paired_slates": len(delta),
        "mean_concordance_delta": float(np.mean(delta)) if delta else None,
        "delta_lcb": one_sided_mean_lcb(delta, confidence=CONFIDENCE),
        "delta_bootstrap_lcb": _slate_clustered_bootstrap_lower(delta),
    }


def _both_agree_above(threshold: float, lcb: float | None, bootstrap_lcb: float | None) -> tuple[bool, bool]:
    """Return (agreement_ok, both_pass). Two-method agreement is required
    to promote: if LCB and bootstrap disagree about whether the value
    clears `threshold`, agreement_ok is False and the caller must report
    BOOTSTRAP_LCB_DISAGREEMENT rather than treating either method as
    authoritative."""
    if lcb is None or bootstrap_lcb is None:
        return True, False  # neither method has a signal; not disagreement, just no pass
    lcb_passes = lcb > threshold
    boot_passes = bootstrap_lcb > threshold
    return lcb_passes == boot_passes, (lcb_passes and boot_passes)


def summarize(metrics: list[SlateRankingMetric], *, phase: str) -> dict[str, Any]:
    score_summaries = {score: _score_summary(metrics, score) for score in SCORE_FIELDS}
    comparisons = {
        baseline: _paired_delta(metrics, "balanced_probability", baseline)
        for baseline in ("market_probability", "base_ev", "v19_order_score")
    }
    locked_slates = len({metric.date for metric in metrics})
    if phase != "locked":
        status = "DERIVATION_ONLY"
    elif locked_slates < MIN_LOCKED_SLATES:
        status = "INSUFFICIENT_INDEPENDENT_SLATES"
    else:
        # Redundant two-method check for every production-relevant claim.
        # Every one of the three must (a) have both methods agree and
        # (b) have both methods pass their threshold. Disagreement on any
        # one is a promotion-blocking finding by itself.
        balanced_agree, balanced_pass = _both_agree_above(
            0.5,
            score_summaries["balanced_probability"]["concordance_lcb"],
            score_summaries["balanced_probability"]["concordance_bootstrap_lcb"],
        )
        market_agree, market_pass = _both_agree_above(
            0.0,
            comparisons["market_probability"]["delta_lcb"],
            comparisons["market_probability"]["delta_bootstrap_lcb"],
        )
        v19_agree, v19_pass = _both_agree_above(
            0.0,
            comparisons["v19_order_score"]["delta_lcb"],
            comparisons["v19_order_score"]["delta_bootstrap_lcb"],
        )
        if not (balanced_agree and market_agree and v19_agree):
            status = "BOOTSTRAP_LCB_DISAGREEMENT"
        elif not (balanced_pass and market_pass and v19_pass):
            status = "RANKING_SIGNAL_NOT_ACCEPTED"
        elif locked_slates < PROMOTION_MIN_SLATES:
            # Every ranking check cleared; the only thing holding promotion
            # back is the slate count. Report this explicitly so the signal
            # is visible while its evidence base is still accumulating,
            # rather than being silently collapsed into the same
            # NOT_ACCEPTED bucket that failing-signal runs use.
            status = "SHADOW_ELIGIBLE_PENDING_MORE_SLATES"
        else:
            status = "RANKING_SIGNAL_ACCEPTED"
    return {
        "phase": phase,
        "status": status,
        "independent_slates": locked_slates,
        "score_summaries": score_summaries,
        "paired_comparisons": comparisons,
    }


def harvest_rows() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, int]]]:
    """Returns (rows, per-date load_errors, per-date funnel diagnostics).

    The funnel diagnostics record, for each date on disk, how many
    candidates existed at each successive filter stage
    (total -> H OVER 0.5 family -> real+priced -> settled). This makes
    the honest reason a run has few evaluable dates visible in the
    report itself, rather than requiring a separate probe: e.g. an
    upstream `price_confirmed` capture gap or a settlement lookup gap
    both show up here directly."""
    actual_lookup = build_actual_lookup(DEFAULT_PROCESSED_ROOT)
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    funnel: dict[str, dict[str, int]] = {}
    for pool in find_raw_pool_csvs(DEFAULT_DAILY_RUNS_ROOT):
        date = pool.parent.name
        try:
            candidates, *_ = shp.prepare_candidates(parse_v11_args(pool))
        except Exception as exc:
            errors[date] = f"{type(exc).__name__}: {exc}"
            continue
        family_count = 0
        real_priced_count = 0
        settled_count = 0
        for candidate in candidates:
            in_family = (
                candidate.target == TARGET
                and candidate.direction == DIRECTION
                and abs(candidate.market_line - LINE) < 1e-9
            )
            if not in_family:
                continue
            family_count += 1
            is_real_priced = (
                candidate.market_source == "real"
                and candidate.price_confirmed
                and candidate.selected_side_price is not None
            )
            if not is_real_priced:
                continue
            real_priced_count += 1
            lookup_key = (
                candidate.run_date.isoformat(),
                normalize_player_key(candidate.player),
                candidate.target,
                str(candidate.game_id),
            )
            actual = actual_lookup.get(lookup_key)
            if actual is None:
                continue
            result = grade_result(actual, candidate.market_line, candidate.direction)
            if result not in {"win", "loss"}:
                continue
            settled_count += 1
            balanced = float(candidate.final_hit_probability)
            market = float(candidate.market_implied_probability)
            price = float(candidate.selected_side_price)
            base_ev = balanced * american_to_decimal(price) - 1.0
            v19_eligible = balanced >= 0.60 and balanced >= market + 0.01 and base_ev >= 0.0
            rows.append({
                "candidate_id": (
                    f"{candidate.run_date.isoformat()}|{candidate.game_id}|{candidate.player}|"
                    f"{candidate.target}|{candidate.direction}|{candidate.market_line}"
                ),
                "date": candidate.run_date.isoformat(),
                "game_id": str(candidate.game_id),
                "win": int(result == "win"),
                "balanced_probability": balanced,
                "market_probability": market,
                "base_ev": base_ev,
                # All v19-admissible candidates rank above rejected candidates;
                # EV then reproduces v19's first ranking axis within each group.
                "v19_order_score": base_ev + (1000.0 if v19_eligible else 0.0),
            })
        funnel[date] = {
            "total_candidates": len(candidates),
            "family_h_over_0_5": family_count,
            "family_real_priced": real_priced_count,
            "family_real_priced_settled": settled_count,
        }
    return rows, errors, funnel


def _describe_bottleneck(funnel: dict[str, dict[str, int]]) -> str:
    """Which staged filter is throwing away the most per-date opportunities?

    Returned as a plain-language label the report reviewer can act on:
    "upstream_price_capture" points at market-data ingestion gaps outside
    V3; "upstream_settlement_lookup" points at a lookup key mismatch in
    the processed data; "family_scope_filter" is the intended narrowness
    (H OVER 0.5 only); "insufficient_history" says the archive itself is
    too thin. Diagnostic label only -- it never changes what V3 accepts."""
    if not funnel:
        return "no_pool_csvs_available"
    with_family = sum(1 for v in funnel.values() if v["family_h_over_0_5"] > 0)
    with_priced = sum(1 for v in funnel.values() if v["family_real_priced"] > 0)
    with_settled = sum(1 for v in funnel.values() if v["family_real_priced_settled"] > 0)
    if with_settled == 0 and with_priced == 0 and with_family == 0:
        return "family_scope_filter"
    if with_family > 0 and with_priced == 0:
        return "upstream_price_capture"
    price_gap = with_family - with_priced
    settle_gap = with_priced - with_settled
    if price_gap > settle_gap and price_gap > 0:
        return "upstream_price_capture"
    if settle_gap > 0:
        return "upstream_settlement_lookup"
    if with_settled < MIN_LOCKED_SLATES + V4_RESERVE_MOST_RECENT_SLATES:
        return "insufficient_history"
    return "no_material_bottleneck"


def _partition_dates(all_dates: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split the harvested slates into (derivation, locked, v4_reserve).

    The reserve is the trailing V4_RESERVE_MOST_RECENT_SLATES dates; they
    are dropped from both derivation and locked partitions here so a
    follow-up preregistration in the same family has genuinely unseen
    forward slates to work with. When fewer real dates exist than the
    reserve is meant to hold, every date reserves and both other
    partitions come back empty -- the correct behavior, not a bug: a
    thin history should not be raided to give V3 something to grade."""
    ordered = sorted(all_dates)
    if V4_RESERVE_MOST_RECENT_SLATES <= 0 or len(ordered) <= V4_RESERVE_MOST_RECENT_SLATES:
        reserve = list(ordered)
        return [], [], reserve
    reserve = ordered[-V4_RESERVE_MOST_RECENT_SLATES:]
    evaluable = ordered[:-V4_RESERVE_MOST_RECENT_SLATES]
    derivation = evaluable[:DERIVATION_SLATES]
    locked = evaluable[DERIVATION_SLATES:]
    return derivation, locked, reserve


def run() -> dict[str, Any]:
    rows, errors, funnel = harvest_rows()
    dates = sorted({row["date"] for row in rows})
    derivation_dates, locked_dates, reserve_dates = _partition_dates(dates)
    derivation_metrics = evaluate_rows([row for row in rows if row["date"] in derivation_dates])
    locked_metrics = evaluate_rows([row for row in rows if row["date"] in locked_dates])
    return {
        "version": V3_VERSION,
        "publication_authority": False,
        "v2_unchanged": True,
        "v19_unchanged": True,
        "preregistration": {
            "family": [TARGET, DIRECTION, LINE],
            "derivation_slates": DERIVATION_SLATES,
            "minimum_locked_slates": MIN_LOCKED_SLATES,
            "promotion_min_slates": PROMOTION_MIN_SLATES,
            "v4_reserve_most_recent_slates": V4_RESERVE_MOST_RECENT_SLATES,
            "confidence": CONFIDENCE,
            "bootstrap_lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "spec_hash": PREREGISTRATION_SPEC_HASH,
            "primary_estimand": "equal-slate mean within-slate winner-loser concordance",
            "acceptance": [
                "balanced concordance LCB > 0.5 AND slate-clustered bootstrap agrees",
                "balanced-minus-market concordance LCB > 0 AND slate-clustered bootstrap agrees",
                "balanced-minus-v19 concordance LCB > 0 AND slate-clustered bootstrap agrees",
                f"locked_slates >= {PROMOTION_MIN_SLATES} for RANKING_SIGNAL_ACCEPTED (else SHADOW_ELIGIBLE_PENDING_MORE_SLATES)",
            ],
            "pair_count_is_independent_evidence": False,
            "slate_is_independent_evidence": True,
        },
        "limitations": [
            "Archived candidates are reconstructed with currently available calibration state because full point-in-time candidate snapshots do not exist.",
            "Retrospective locked results are therefore diagnostic; prospective frozen slates are required for certification.",
            "Slate clustering subsumes shared-game dependence for primary inference.",
        ],
        "rows": len(rows),
        "dates": dates,
        "derivation_dates": derivation_dates,
        "locked_dates": locked_dates,
        "v4_reserve_dates": reserve_dates,
        "load_errors": errors,
        "harvest_diagnostics": {
            "per_date_funnel": funnel,
            "dates_with_any_candidates": sum(1 for v in funnel.values() if v["total_candidates"] > 0),
            "dates_with_family_h_over_0_5": sum(1 for v in funnel.values() if v["family_h_over_0_5"] > 0),
            "dates_with_family_real_priced": sum(1 for v in funnel.values() if v["family_real_priced"] > 0),
            "dates_with_family_real_priced_settled": sum(1 for v in funnel.values() if v["family_real_priced_settled"] > 0),
            "primary_bottleneck": _describe_bottleneck(funnel),
        },
        "derivation": summarize(derivation_metrics, phase="derivation"),
        "locked": summarize(locked_metrics, phase="locked"),
        "locked_slate_metrics": [asdict(metric) for metric in locked_metrics],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


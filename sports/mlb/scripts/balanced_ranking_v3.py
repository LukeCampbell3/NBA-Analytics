#!/usr/bin/env python3
"""Preregistered V3 test of within-slate balanced-probability ranking.

V3 is separate from local recovery. It asks whether balanced probability orders
winners above losers within the same H OVER 0.5 slate, and whether that ordering
is incrementally better than market probability and the v19 policy order.

Candidate pairs improve measurement within a slate; they are never counted as
independent evidence. All inference equal-weights independent slates. The first
four available slates are derivation-only. Later slates are locked evaluation,
with at least eight required before any acceptance decision.
"""

from __future__ import annotations

import argparse
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
CONFIDENCE = 0.975
TOP_K = (1, 3, 5)
SCORE_FIELDS = (
    "balanced_probability",
    "market_probability",
    "base_ev",
    "v19_order_score",
)


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
    }


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
        balanced_lcb = score_summaries["balanced_probability"]["concordance_lcb"]
        market_lcb = comparisons["market_probability"]["delta_lcb"]
        v19_lcb = comparisons["v19_order_score"]["delta_lcb"]
        status = (
            "RANKING_SIGNAL_ACCEPTED"
            if balanced_lcb is not None and balanced_lcb > 0.5
            and market_lcb is not None and market_lcb > 0.0
            and v19_lcb is not None and v19_lcb > 0.0
            else "RANKING_SIGNAL_NOT_ACCEPTED"
        )
    return {
        "phase": phase,
        "status": status,
        "independent_slates": locked_slates,
        "score_summaries": score_summaries,
        "paired_comparisons": comparisons,
    }


def harvest_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    actual_lookup = build_actual_lookup(DEFAULT_PROCESSED_ROOT)
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for pool in find_raw_pool_csvs(DEFAULT_DAILY_RUNS_ROOT):
        date = pool.parent.name
        try:
            candidates, *_ = shp.prepare_candidates(parse_v11_args(pool))
        except Exception as exc:
            errors[date] = f"{type(exc).__name__}: {exc}"
            continue
        for candidate in candidates:
            if not (
                candidate.market_source == "real"
                and candidate.price_confirmed
                and candidate.target == TARGET
                and candidate.direction == DIRECTION
                and abs(candidate.market_line - LINE) < 1e-9
                and candidate.selected_side_price is not None
            ):
                continue
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
    return rows, errors


def run() -> dict[str, Any]:
    rows, errors = harvest_rows()
    dates = sorted({row["date"] for row in rows})
    derivation_dates = dates[:DERIVATION_SLATES]
    locked_dates = dates[DERIVATION_SLATES:]
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
            "confidence": CONFIDENCE,
            "primary_estimand": "equal-slate mean within-slate winner-loser concordance",
            "acceptance": [
                "balanced concordance LCB > 0.5",
                "balanced-minus-market concordance LCB > 0",
                "balanced-minus-v19 concordance LCB > 0",
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
        "load_errors": errors,
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


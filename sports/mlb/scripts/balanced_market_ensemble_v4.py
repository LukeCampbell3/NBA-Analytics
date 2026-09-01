#!/usr/bin/env python3
"""V4 optimized singles shadow selector with separate diagnostics and action.

V4 asks whether balanced probability adds incremental information to the exact
market probability. It has one fitted parameter, constrained to [0, 1]:

    logit(P_ensemble) = logit(P_market)
                       + w * (logit(P_balanced) - logit(P_market))

The market blend remains a calibration diagnostic. It failed to add actionable
value in the frozen functional replay, so it does not control singles. The
shadow singles action instead reuses the independently specified confidence /
exact-price frontier: balanced probability >= 60%, at least one percentage
point above exact break-even, and positive EV. Price is considered last and
there is no pick quota. This keeps V4 from being tuned into the failed market-
favorite ranking while still testing every supported straight bet.

Future V3 snapshots carry player/team/game identity metadata end-to-end. V4
copies those fields into its live-facing shadow rows; a separate publication-
time validator remains negative authority and verifies the identity against the
exact MLB game feed before a card may render.

This module is shadow-only and has no publication authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from local_node_selector_v2 import one_sided_mean_lcb


V4_VERSION = "balanced_value_frontier_v4_optimized_singles_shadow"
TARGET = "H"
DIRECTION = "OVER"
LINE = 0.5
WEIGHT_GRID_STEP = 0.01
MIN_TRAINING_SLATES = 4
PROMOTION_MIN_SLATES = 30
CONFIDENCE = 0.975
MIN_PROBABILITY_EDGE = 0.01
MIN_SAFE_EV = 0.0
MIN_BALANCED_PROBABILITY = 0.60


def _spec_hash() -> str:
    payload = {
        "version": V4_VERSION,
        "family": [TARGET, DIRECTION, LINE],
        "weight_range": [0.0, 1.0],
        "weight_grid_step": WEIGHT_GRID_STEP,
        "loss": "equal_slate_log_loss",
        "uncertainty": "leave_one_slate_out_residual_one_sided_lcb_negative_authority_only",
        "min_training_slates": MIN_TRAINING_SLATES,
        "promotion_min_slates": PROMOTION_MIN_SLATES,
        "confidence": CONFIDENCE,
        "minimum_probability_edge": MIN_PROBABILITY_EDGE,
        "minimum_safe_ev": MIN_SAFE_EV,
        "singles_action_probability": "balanced_probability",
        "singles_action_minimum_probability": MIN_BALANCED_PROBABILITY,
        "singles_action_gate": "balanced_probability >= market_probability + 0.01 AND decision_ev > 0",
        "market_ensemble_role": "diagnostic_only_after_functional_backtest_rejected_incremental_action_value",
        "pick_quota": None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# Frozen at V4 creation. CI recomputes the specification and fails if a
# critical constant/formula descriptor changes without an explicit new study.
PREREGISTRATION_SPEC_HASH = "e902fd8ba0628655528d9795a69c99b97a505a1ad1dcc9a217af9b50503fecad"


@dataclass(frozen=True)
class V4Fit:
    training_slates: int
    training_rows: int
    balanced_weight: float
    market_weight: float
    equal_slate_log_loss: float
    cross_fitted_slate_residuals: tuple[float, ...]
    residual_lcb: float | None
    safe_calibration_adjustment: float


@dataclass(frozen=True)
class V4Score:
    candidate_id: str
    balanced_probability: float
    ensemble_probability: float
    safe_probability: float
    market_probability: float
    price: float
    safe_ev: float
    probability_edge: float
    decision_probability: float
    decision_ev: float
    eligible: bool
    reasons: tuple[str, ...]


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _logit(value: float) -> float:
    p = _clip_probability(value)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def blend_probability(balanced: float, market: float, weight: float) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("balanced weight must be in [0, 1]")
    return _sigmoid(_logit(market) + weight * (_logit(balanced) - _logit(market)))


def american_to_decimal(price: float) -> float:
    if -100.0 < price < 100.0:
        raise ValueError("invalid American price")
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def _valid_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        try:
            balanced = float(row["balanced_probability"])
            market = float(row["market_probability"])
            outcome = int(row["win"])
            slate = str(row.get("date") or row.get("slate_date") or "")
        except (KeyError, TypeError, ValueError):
            continue
        if slate and outcome in {0, 1} and 0.0 < balanced < 1.0 and 0.0 < market < 1.0:
            valid.append({**row, "date": slate, "win": outcome})
    return valid


def equal_slate_log_loss(rows: Iterable[dict[str, Any]], weight: float) -> float:
    by_slate: dict[str, list[float]] = {}
    for row in _valid_rows(rows):
        p = blend_probability(float(row["balanced_probability"]), float(row["market_probability"]), weight)
        y = int(row["win"])
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        by_slate.setdefault(row["date"], []).append(loss)
    if not by_slate:
        return math.inf
    return statistics.fmean(statistics.fmean(losses) for losses in by_slate.values())


def select_weight(rows: Iterable[dict[str, Any]]) -> tuple[float, float]:
    materialized = list(rows)
    candidates = [round(index * WEIGHT_GRID_STEP, 10) for index in range(round(1.0 / WEIGHT_GRID_STEP) + 1)]
    scored = [(equal_slate_log_loss(materialized, weight), weight) for weight in candidates]
    # A tie goes to the market anchor (smaller balanced weight).
    loss, weight = min(scored, key=lambda item: (item[0], item[1]))
    return weight, loss


def fit(rows: Iterable[dict[str, Any]], *, before_date: str) -> V4Fit:
    prior = [row for row in _valid_rows(rows) if row["date"] < before_date]
    dates = sorted({row["date"] for row in prior})
    weight, loss = select_weight(prior)
    cross_fitted_residuals: list[float] = []
    for held_out in dates:
        training = [row for row in prior if row["date"] != held_out]
        held_rows = [row for row in prior if row["date"] == held_out]
        if not training or not held_rows:
            continue
        held_weight, _ = select_weight(training)
        cross_fitted_residuals.append(
            statistics.fmean(
                int(row["win"])
                - blend_probability(float(row["balanced_probability"]), float(row["market_probability"]), held_weight)
                for row in held_rows
            )
        )
    residual_lcb = one_sided_mean_lcb(cross_fitted_residuals, confidence=CONFIDENCE)
    # Positive retrospective residuals are not permission to inflate a new
    # candidate. Only demonstrated/uncertain overconfidence has authority.
    safe_adjustment = min(0.0, residual_lcb) if residual_lcb is not None else 0.0
    return V4Fit(
        training_slates=len(dates),
        training_rows=len(prior),
        balanced_weight=weight,
        market_weight=1.0 - weight,
        equal_slate_log_loss=loss,
        cross_fitted_slate_residuals=tuple(cross_fitted_residuals),
        residual_lcb=residual_lcb,
        safe_calibration_adjustment=safe_adjustment,
    )


def score(candidate: dict[str, Any], fitted: V4Fit) -> V4Score:
    balanced = float(candidate["balanced_probability"])
    market = float(candidate["market_probability"])
    price = float(candidate.get("price", candidate.get("selected_side_price")))
    ensemble = blend_probability(balanced, market, fitted.balanced_weight)
    safe = _clip_probability(ensemble + fitted.safe_calibration_adjustment)
    safe_ev = safe * american_to_decimal(price) - 1.0
    edge = safe - market
    decision_ev = balanced * american_to_decimal(price) - 1.0
    decision_edge = balanced - market
    reasons: list[str] = []
    if balanced < MIN_BALANCED_PROBABILITY:
        reasons.append("balanced_probability_below_60pct")
    if decision_edge < MIN_PROBABILITY_EDGE:
        reasons.append("balanced_probability_edge_below_1pct")
    if decision_ev <= MIN_SAFE_EV:
        reasons.append("decision_ev_not_positive")
    return V4Score(
        candidate_id=str(candidate.get("candidate_id") or ""),
        balanced_probability=balanced,
        ensemble_probability=ensemble,
        safe_probability=safe,
        market_probability=market,
        price=price,
        safe_ev=safe_ev,
        probability_edge=decision_edge,
        decision_probability=balanced,
        decision_ev=decision_ev,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def run_shadow(candidates: Iterable[dict[str, Any]], history: Iterable[dict[str, Any]], *, slate_date: str) -> dict[str, Any]:
    fitted = fit(history, before_date=slate_date)
    candidate_list = list(candidates)
    scores = [score(candidate, fitted) for candidate in candidate_list]
    eligible = sorted((item for item in scores if item.eligible), key=lambda item: (-item.decision_ev, -item.decision_probability, item.candidate_id))
    candidate_by_id = {str(candidate.get("candidate_id") or ""): candidate for candidate in candidate_list}
    frontend_plays = []
    for rank, item in enumerate(eligible, start=1):
        candidate = candidate_by_id.get(item.candidate_id, {})
        frontend_plays.append(
            {
                "rank": rank,
                "candidate_id": item.candidate_id,
                "player": str(candidate.get("player") or ""),
                "player_id": str(candidate.get("player_id") or ""),
                "team": str(candidate.get("team") or ""),
                "team_id": str(candidate.get("team_id") or ""),
                "opponent": str(candidate.get("opponent") or ""),
                "opponent_id": str(candidate.get("opponent_id") or ""),
                "is_home": str(candidate.get("is_home") or ""),
                "game_id": str(candidate.get("game_id") or ""),
                "commence_time_utc": str(candidate.get("commence_time_utc") or ""),
                "target": str(candidate.get("target") or TARGET),
                "direction": str(candidate.get("direction") or DIRECTION),
                "line": float(candidate.get("line", LINE)),
                "sportsbook": str(candidate.get("selected_sportsbook_key") or ""),
                "american_price": item.price,
                "balanced_probability": item.balanced_probability,
                "market_probability": item.market_probability,
                "probability_edge": item.probability_edge,
                "decision_ev": item.decision_ev,
                "authorization_status": "SHADOW_ONLY",
            }
        )
    status = "SHADOW_ONLY"
    if fitted.training_slates < MIN_TRAINING_SLATES:
        status = "INSUFFICIENT_PRIOR_SLATES"
    return {
        "version": V4_VERSION,
        "preregistration_spec_hash": PREREGISTRATION_SPEC_HASH,
        "status": status,
        "publication_authority": False,
        "market_ensemble_role": "diagnostic_only",
        "singles_action_probability": "balanced_probability",
        "dynamic_gate": "P_balanced >= max(0.60, exact_price_break_even + 0.01) and EV(P_balanced, price) > 0",
        "pick_count_constraint": "none",
        "slate_date": slate_date,
        "fit": asdict(fitted),
        "candidate_count": len(scores),
        "eligible_count": len(eligible),
        "eligible": [asdict(item) for item in eligible],
        "frontend_plays": frontend_plays,
        "scores": [asdict(item) for item in scores],
    }


def run_prospective_snapshot(snapshot_path: Path, evidence_root: Path, output_path: Path) -> dict[str, Any]:
    """Score one immutable snapshot using only earlier settled snapshots."""
    # Local import avoids making the pure scoring API depend on filesystem
    # evidence collection during normal unit-test imports.
    from prospective_balanced_ranking_v3 import load_settled_rows

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    slate_date = str(snapshot["slate_date"])
    history = [row for row in load_settled_rows(evidence_root) if str(row["date"]) < slate_date]
    candidates = [
        {**candidate, "price": candidate["selected_side_price"]}
        for candidate in snapshot["candidates"]
    ]
    report = run_shadow(candidates, history, slate_date=slate_date)
    report.update(
        {
            "record_type": "prospective_pregame_shadow_decision",
            "snapshot_identity_sha256": snapshot["identity_sha256"],
            "strictly_prior_settled_slates": len({row["date"] for row in history}),
            "strictly_prior_settled_rows": len(history),
            "v19_publication_unchanged": True,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        # A pregame decision for a frozen snapshot is immutable. Operational
        # retries may reproduce it, but may not rewrite its scientific state.
        if json.dumps(existing, sort_keys=True) != json.dumps(report, sort_keys=True):
            raise RuntimeError(f"immutable V4 shadow report conflict: {output_path}")
    else:
        output_path.write_text(encoded, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_prospective_snapshot(args.snapshot, args.evidence_root, args.output)
    print(
        f"V4 {report['status']}: {report['eligible_count']} eligible from "
        f"{report['candidate_count']} candidates; {report['strictly_prior_settled_slates']} prior slates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

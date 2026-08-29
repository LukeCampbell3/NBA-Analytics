#!/usr/bin/env python3
"""Apply the tighter MLB single-bet publication overlay.

The core v16 selector remains reproducible. This additive publication layer
uses evidence that already existed before the current slate:

* active isotonic hit-probability recalibration is negative-authority only;
* the repo's frozen historical floor sweep showed the 0.60 final-probability
  policy losing ROI, while 0.65 was the first tested floor with positive ROI;
* a card carrying ``lineup_unconfirmed`` is not suitable for the tight board.

The overlay therefore recalculates each published single's final probability
and derives a price-specific break-even gate from its exact confirmed odds. Each
play is evaluated independently; there is no fixed probability floor, top-N
quota, or board-size target. The public ``plays`` list and audit are then updated.
It never changes PARLAY_POLICY_V2, settlement records, or staking authority.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "hit_probability_isotonic_calibration_2026.json"
PAYLOAD_PATHS = (
    REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json",
    REPO_ROOT / "dist" / "mlb" / "data" / "daily_predictions.json",
    REPO_ROOT / "paywall" / "private-content" / "app" / "mlb" / "data" / "daily_predictions.json",
)

OVERLAY_VERSION = "premium_price_aware_quality_v18_shadow"
MIN_FINAL_EXPECTED_VALUE = 0.0
BLOCKING_RISK_FLAGS = frozenset({"lineup_unconfirmed", "lineup_role_mismatch"})


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
    if number > 0:
        return 1.0 + number / 100.0
    return 1.0 + 100.0 / abs(number)


def interpolate_breakpoints(probability: float, breakpoints: list[list[float]]) -> float:
    """Piecewise-linear equivalent of the deployed isotonic breakpoint map."""
    if not breakpoints:
        return probability
    points = sorted((float(x), float(y)) for x, y in breakpoints)
    xs = [point[0] for point in points]
    if probability <= xs[0]:
        return points[0][1]
    if probability >= xs[-1]:
        return points[-1][1]
    right = bisect_right(xs, probability)
    x0, y0 = points[right - 1]
    x1, y1 = points[right]
    if x1 <= x0:
        return min(y0, y1)
    weight = (probability - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def load_active_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "active" or not isinstance(payload.get("breakpoints"), list):
        return None
    return payload


def final_probability(play: dict[str, Any], calibration: dict[str, Any] | None) -> tuple[float | None, float | None]:
    current = _finite(play.get("final_hit_probability"))
    if current is None:
        current = _finite(play.get("estimated_hit_probability"))
    if current is None:
        return None, None

    historical = None
    model_probability = _finite(play.get("model_hit_probability"))
    if calibration is not None and model_probability is not None:
        historical = interpolate_breakpoints(model_probability, calibration["breakpoints"])
        current = min(current, historical)
    return max(0.0, min(1.0, current)), historical


def tighten_play(play: dict[str, Any], calibration: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    tightened = dict(play)
    reasons: list[str] = []
    probability, historical = final_probability(play, calibration)

    if probability is None:
        reasons.append("final_probability_unavailable")
        return tightened, reasons

    previous_probability = _finite(play.get("estimated_hit_probability"))
    tightened["pre_tight_hit_probability"] = previous_probability
    tightened["historically_calibrated_hit_probability"] = historical
    tightened["final_hit_probability"] = probability
    # Existing frontend consumes estimated_hit_probability. Make that public
    # contract conservative instead of continuing to display the superseded
    # pre-isotonic value.
    tightened["estimated_hit_probability"] = probability

    decimal = american_to_decimal(play.get("selected_side_price"))
    final_ev = probability * decimal - 1.0 if decimal is not None else None
    tightened["pre_tight_expected_value_per_unit"] = _finite(play.get("expected_value_per_unit"))
    tightened["expected_value_per_unit"] = final_ev

    implied = _finite(play.get("market_implied_probability"))
    if implied is None and decimal is not None:
        implied = 1.0 / decimal
    tightened["market_implied_probability"] = implied
    probability_margin = probability - implied if implied is not None else None
    tightened["probability_edge"] = probability_margin
    tightened["dynamic_break_even_probability"] = implied
    tightened["dynamic_probability_margin"] = probability_margin
    if final_ev is None or final_ev < MIN_FINAL_EXPECTED_VALUE:
        reasons.append("final_price_ev_negative")
    if not bool(play.get("price_confirmed", False)):
        reasons.append("price_unconfirmed")

    risk_flags = {str(flag).strip().lower() for flag in (play.get("risk_flags") or [])}
    blocking = sorted(risk_flags & BLOCKING_RISK_FLAGS)
    reasons.extend(blocking)
    return tightened, reasons


def apply_overlay(payload: dict[str, Any], calibration: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(payload)
    original = [dict(play) for play in payload.get("plays", []) if isinstance(play, dict)]
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for play in original:
        tightened, reasons = tighten_play(play, calibration)
        if reasons:
            rejected.append(
                {
                    "play_key": play.get("play_key"),
                    "player": play.get("player"),
                    "target": play.get("target"),
                    "line": play.get("market_line"),
                    "reasons": reasons,
                    "pre_tight_probability": play.get("estimated_hit_probability"),
                    "final_probability": tightened.get("final_hit_probability"),
                    "pre_tight_ev": play.get("expected_value_per_unit"),
                    "final_ev": tightened.get("expected_value_per_unit"),
                }
            )
            continue
        kept.append(tightened)

    # Within the evidence-supported probability-safe pool, price efficiency is
    # the first ranking axis. Final probability is the tie-break, then the
    # existing quality score. This avoids turning 65% into a new max-hit-only
    # selector while still refusing low-probability volume.
    kept.sort(
        key=lambda play: (
            _finite(play.get("expected_value_per_unit")) or -999.0,
            _finite(play.get("final_hit_probability")) or -999.0,
            _finite(play.get("final_pool_quality_score")) or -999.0,
        ),
        reverse=True,
    )
    for rank, play in enumerate(kept, start=1):
        play["rank"] = rank

    result["plays"] = kept
    result["base_policy_profile"] = payload.get("base_policy_profile") or payload.get("policy_profile")
    result["policy_profile"] = OVERLAY_VERSION
    result["tight_quality_overlay"] = {
        "version": OVERLAY_VERSION,
        "authority": "shadow_publication_filter",
        "base_policy_unchanged": True,
        "parlay_v2_unchanged": True,
        "dynamic_probability_gate": "final_probability >= exact_price_break_even_probability",
        "fixed_probability_floor": None,
        "minimum_final_expected_value": MIN_FINAL_EXPECTED_VALUE,
        "pick_count_constraint": "none; every play is evaluated independently",
        "blocking_risk_flags": sorted(BLOCKING_RISK_FLAGS),
        "probability_source": "min(existing_estimate, active_isotonic_model_probability_calibration)",
        "ranking_after_gates": "final_price_ev_desc_then_final_probability_desc; ranking_does_not_limit_publication",
        "input_plays": len(original),
        "published_plays": len(kept),
        "rejected_plays": len(rejected),
        "rejections": rejected,
        "evidence_note": (
            "The archived fixed-floor sweep is retained as historical evidence, but it does not justify "
            "applying one probability threshold across different prices. v18 instead requires each "
            "conservatively recalibrated probability to clear the break-even probability of its exact "
            "confirmed price. No relative rank, quota, or board-size target can reject an otherwise "
            "eligible play. This remains a shadow publication policy."
        ),
        "calibration_model_version": calibration.get("model_version") if calibration else None,
        "calibration_status": calibration.get("status") if calibration else "unavailable",
    }
    return result


def apply_to_file(path: Path, calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    tightened = apply_overlay(payload, calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tightened, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return tightened


def main() -> int:
    calibration = load_active_calibration()
    summaries = []
    for path in PAYLOAD_PATHS:
        payload = apply_to_file(path, calibration)
        if payload is not None:
            overlay = payload.get("tight_quality_overlay", {})
            summaries.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "input_plays": overlay.get("input_plays"),
                    "published_plays": overlay.get("published_plays"),
                    "rejected_plays": overlay.get("rejected_plays"),
                }
            )
    print(json.dumps({"overlay": OVERLAY_VERSION, "files": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

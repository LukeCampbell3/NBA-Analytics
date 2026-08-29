#!/usr/bin/env python3
"""Attach a tight public-quality audit to frozen PARLAY_POLICY_V2 output.

The frozen V2 action and prospective evidence must not be changed after seeing
results. This script therefore NEVER changes ``parlays.action``, the selected
wager, policy status, or settlement. It adds a separate public presentation
gate so a research-selected pair is not described as a high-hit choice when
its own displayed marginal/joint probabilities and price fail modern quality
standards.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_PATHS = (
    REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json",
    REPO_ROOT / "dist" / "mlb" / "data" / "daily_predictions.json",
    REPO_ROOT / "paywall" / "private-content" / "app" / "mlb" / "data" / "daily_predictions.json",
)

OVERLAY_VERSION = "PARLAY_V2_PUBLIC_QUALITY_OVERLAY_V1"
MIN_LEG_PROBABILITY = 0.70
MIN_JOINT_PROBABILITY = 0.50
MIN_COMBINED_DECIMAL_PRICE = 2.00
MIN_EXPECTED_VALUE = 0.05


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit_pair(pair: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pair, dict):
        return {
            "version": OVERLAY_VERSION,
            "action": "ABSTAIN",
            "reason": "no_pair_available",
            "frozen_policy_mutated": False,
        }

    legs = [pair.get("leg_1"), pair.get("leg_2")]
    probabilities: list[float] = []
    decimals: list[float] = []
    in_support = True
    reasons: list[str] = []
    for index, leg in enumerate(legs, start=1):
        if not isinstance(leg, dict):
            reasons.append(f"leg_{index}_missing")
            continue
        probability = _finite(leg.get("model_probability_estimate"))
        decimal = _finite(leg.get("decimal_price"))
        if probability is None:
            reasons.append(f"leg_{index}_probability_missing")
        else:
            probabilities.append(probability)
            if probability < MIN_LEG_PROBABILITY:
                reasons.append(f"leg_{index}_probability_below_70pct")
        if decimal is None or decimal <= 1.0:
            reasons.append(f"leg_{index}_price_missing")
        else:
            decimals.append(decimal)
        if not bool(leg.get("in_support", False)):
            in_support = False
            reasons.append(f"leg_{index}_out_of_support")

    joint = _finite(pair.get("joint_probability_estimate"))
    if joint is None and len(probabilities) == 2:
        joint = probabilities[0] * probabilities[1]
    if joint is None:
        reasons.append("joint_probability_missing")
    elif joint < MIN_JOINT_PROBABILITY:
        reasons.append("joint_probability_below_50pct")

    combined_decimal = decimals[0] * decimals[1] if len(decimals) == 2 else None
    if combined_decimal is None:
        reasons.append("combined_price_missing")
    elif combined_decimal < MIN_COMBINED_DECIMAL_PRICE:
        reasons.append("combined_price_below_plus_100")

    expected_value = joint * combined_decimal - 1.0 if joint is not None and combined_decimal is not None else None
    if expected_value is None:
        reasons.append("expected_value_unavailable")
    elif expected_value < MIN_EXPECTED_VALUE:
        reasons.append("expected_value_below_5pct")

    passed = not reasons and in_support
    return {
        "version": OVERLAY_VERSION,
        "action": "PASS" if passed else "ABSTAIN",
        "reason": None if passed else "tight_quality_gates_failed",
        "blocking_reasons": reasons,
        "leg_probabilities": probabilities,
        "joint_probability": joint,
        "combined_decimal_price": combined_decimal,
        "expected_value_per_unit": expected_value,
        "thresholds": {
            "min_leg_probability": MIN_LEG_PROBABILITY,
            "min_joint_probability": MIN_JOINT_PROBABILITY,
            "min_combined_decimal_price": MIN_COMBINED_DECIMAL_PRICE,
            "min_expected_value_per_unit": MIN_EXPECTED_VALUE,
            "in_support_required": True,
        },
        "authority": "public_presentation_only_shadow",
        "frozen_policy_mutated": False,
        "evidence_note": (
            "The frozen V2 research action remains authoritative for its prospective evidence stream. "
            "This overlay only controls whether that research action is presented as a tight-quality public candidate."
        ),
    }


def apply_overlay(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    parlays = dict(result.get("parlays") or {})
    pair = parlays.get("selected_parlay") or parlays.get("shadow_candidate")
    parlays["public_quality_overlay"] = audit_pair(pair)
    result["parlays"] = parlays
    return result


def main() -> int:
    summaries = []
    for path in PAYLOAD_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        output = apply_overlay(payload)
        path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        overlay = output.get("parlays", {}).get("public_quality_overlay", {})
        summaries.append({"path": str(path.relative_to(REPO_ROOT)), "action": overlay.get("action"), "reasons": overlay.get("blocking_reasons", [])})
    print(json.dumps({"overlay": OVERLAY_VERSION, "files": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

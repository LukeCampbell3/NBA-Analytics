"""NFL-only loss-aware meta-policy for base player-prop candidates."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


ARTIFACT_TYPE = "nfl_loss_aware_pick_meta_policy"
MODEL_VERSION = "nfl_pick_meta_policy_v1"
CONFIDENCE_CANDIDATES = (0.56, 0.58, 0.60, 0.62, 0.64, 0.65, 0.67)
ADVANTAGE_CANDIDATES = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
MINIMUM_PRICE_CANDIDATES = (-150, -140, -130, -120, -115)
MAXIMUM_PRICE_CANDIDATES = (-100, 100, 130)
WEEKLY_CAP_CANDIDATES = (4, 6, 8, 10, 12)


def apply_meta_policy(
    rows: pd.DataFrame,
    *,
    minimum_side_probability: float,
    minimum_no_vig_advantage: float,
    minimum_price: float,
    maximum_price: float,
    weekly_cap: int,
) -> pd.DataFrame:
    eligible = rows.loc[
        rows["target"].eq("passing")
        & rows["estimated_side_probability"].ge(minimum_side_probability)
        & rows["probability_advantage"].ge(minimum_no_vig_advantage)
        & rows["selected_price"].between(
            minimum_price, maximum_price, inclusive="both"
        )
    ].copy()
    ranked = eligible.sort_values(
        [
            "season",
            "week",
            "estimated_side_probability",
            "probability_advantage",
            "player_display_name",
        ],
        ascending=[True, True, False, False, True],
    )
    return (
        ranked.groupby(["season", "week"], group_keys=False, sort=True)
        .head(int(weekly_cap))
        .reset_index(drop=True)
    )


def wilson_interval(wins: int, losses: int) -> list[float] | None:
    decisions = wins + losses
    if not decisions:
        return None
    z = 1.959963984540054
    proportion = wins / decisions
    denominator = 1.0 + z * z / decisions
    midpoint = (proportion + z * z / (2.0 * decisions)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / decisions
            + z * z / (4.0 * decisions * decisions)
        )
        / denominator
    )
    return [round(midpoint - radius, 4), round(midpoint + radius, 4)]


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("sport") != "NFL" or artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(
            "The pick meta-policy must be the NFL loss-aware artifact; MLB artifacts "
            "and generic model files are not accepted."
        )
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Unsupported NFL pick meta-policy version: {artifact.get('model_version')}")
    required = {
        "minimum_side_probability",
        "minimum_no_vig_advantage",
        "minimum_price",
        "maximum_price",
        "weekly_cap",
    }
    if not required.issubset(artifact.get("policy", {})):
        raise ValueError("NFL pick meta-policy is missing its frozen rule contract.")
    calibration = artifact.get("confidence_calibration") or {}
    if calibration.get("method") != "identity" or calibration.get("status") != "passed":
        raise ValueError("NFL pick meta-policy lacks a validated confidence calibration.")
    if len(calibration.get("historical_support") or []) != 2:
        raise ValueError("NFL confidence calibration is missing historical support bounds.")


def score_with_artifact(rows: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Apply the learned gates without inventing an individual win probability."""

    validate_artifact(artifact)
    policy = artifact["policy"]
    calibration = artifact["confidence_calibration"]
    frame = rows.copy()
    frame["raw_model_probability"] = frame["estimated_side_probability"].astype(float)
    frame["calibrated_hit_probability"] = frame["raw_model_probability"].clip(0.0, 1.0)
    support_minimum, support_maximum = calibration["historical_support"]
    frame["confidence_in_support"] = frame["raw_model_probability"].between(
        support_minimum, support_maximum, inclusive="both"
    )
    frame["meta_policy_score"] = (
        frame["calibrated_hit_probability"] + frame["probability_advantage"]
    )
    frame["meta_eligible"] = (
        frame["target"].eq("passing")
        & frame["estimated_side_probability"].ge(policy["minimum_side_probability"])
        & frame["probability_advantage"].ge(policy["minimum_no_vig_advantage"])
        & frame["selected_price"].between(
            policy["minimum_price"], policy["maximum_price"], inclusive="both"
        )
        & frame["confidence_in_support"]
    )
    return frame

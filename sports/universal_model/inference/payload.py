"""Universal output payload (spec section 15/52): same shape for every
sport, used both for direct inference and for the shadow-comparison
artifact against existing per-sport predictors."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_payload(row, model_out: dict, checkpoint_name: str) -> dict[str, Any]:
    return {
        "observation_id": row["observation_id"],
        "sport": row["sport"],
        "event_id": row["event_id"],
        "entity_id": row["entity_id"],
        "entity_name": row["entity_name"],
        "target": row["target"],
        "line": None if row.get("line") != row.get("line") else row.get("line"),  # NaN-safe
        "expected_outcome_z": float(model_out["z_pred"].squeeze().item()),
        "prob_over": float(model_out["prob_over"].squeeze().item()),
        "calibrated_probability": float(model_out["prob_over"].squeeze().item()),
        "market_probability": row.get("no_vig_market_probability"),
        "checkpoint": checkpoint_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "SHADOW UNIVERSAL PREDICTOR output -- research/comparison only, not a certified betting signal (spec section 52/54).",
    }

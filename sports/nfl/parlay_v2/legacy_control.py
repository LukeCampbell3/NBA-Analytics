from __future__ import annotations

"""Read-only diagnostic access to the OLD parlay subsystem's output
(sports/nfl/predictions/daily_policy.py's build_shadow_parlay, embedded
under the "daily_parlay" key of sports/nfl/web/data/daily_predictions.json
-- unlike MLB's select_daily_parlay.py, NFL's old system writes no
separate daily_parlay_*.json file), for comparison purposes only.

Named explicitly `legacy_parlay_control` / `old_parlay_diagnostic`, mirroring
sports/mlb/parlay_v2/legacy_control.py -- never "parlay" unqualified, so
nothing here can be mistaken for the new V2 authority. This module NEVER
writes to the old system's output and NEVER feeds anything back into
parlay_certification_v2 -- it only reads what the old pipeline already
embedded in the published payload, for the comparison artifact in
comparison.py.

Reminder of why this old system is diagnostic-only, not a candidate to
replace: build_shadow_parlay's own "reason" field records that its
deterministic two-leg rule went 2-16 on the locked 2022 holdout, and it
already self-reports candidate_authorized=False / status="withheld" --
this module changes none of that, it only reads it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LegacyParlayControl:
    """Diagnostic snapshot of the OLD system's selection -- CONTROL only,
    never authorization. `old_control_pair` may legitimately be None (the
    old system found no cross-event/cross-player candidate, or produced no
    artifact for this date)."""

    available: bool
    old_control_pair: list[dict[str, Any]] | None
    old_control_probability: float | None
    old_control_quote: dict[str, Any] | None
    reason: str


def _control_from_daily_parlay(raw: dict[str, Any] | None) -> LegacyParlayControl:
    if not isinstance(raw, dict):
        return LegacyParlayControl(False, None, None, None, "old_parlay_diagnostic_no_daily_parlay_key")

    ticket = raw.get("selected_ticket")
    if not isinstance(ticket, dict) or not ticket.get("legs"):
        return LegacyParlayControl(True, None, None, None, "old_parlay_diagnostic_no_selected_ticket")

    legs = [
        {
            "player": leg.get("player"),
            "target": leg.get("market") or leg.get("target"),
            "line": leg.get("line"),
            "side": leg.get("direction"),
        }
        for leg in ticket.get("legs", [])
    ]
    quote = {
        "combined_decimal_price": ticket.get("combined_decimal_price"),
        "sportsbook": ticket.get("sportsbook_key"),
    }
    return LegacyParlayControl(
        available=True,
        old_control_pair=legs,
        old_control_probability=ticket.get("projected_probability"),
        old_control_quote=quote,
        reason="old_parlay_diagnostic_loaded",
    )


def load_legacy_parlay_control_from_payload(payload: dict[str, Any]) -> LegacyParlayControl:
    """Preferred entry point: NFL's old system's output lives inline in the
    already-published daily_predictions.json payload, under "daily_parlay"
    -- there is no separate legacy artifact file to read."""
    return _control_from_daily_parlay(payload.get("daily_parlay") if isinstance(payload, dict) else None)


def load_legacy_parlay_control(daily_predictions_json_path: Path) -> LegacyParlayControl:
    """File-path convenience wrapper, mirroring MLB's legacy_control.py
    signature, for callers that only have a path (e.g. comparison.py)."""
    path = Path(daily_predictions_json_path)
    if not path.exists():
        return LegacyParlayControl(False, None, None, None, "old_parlay_diagnostic_artifact_not_found")
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return LegacyParlayControl(False, None, None, None, f"old_parlay_diagnostic_artifact_unreadable: {exc}")
    if not isinstance(raw, dict):
        return LegacyParlayControl(False, None, None, None, "old_parlay_diagnostic_artifact_unreadable: not a JSON object")
    return _control_from_daily_parlay(raw.get("daily_parlay"))

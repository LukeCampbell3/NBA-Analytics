from __future__ import annotations

"""Read-only diagnostic access to the OLD parlay subsystem's output
(sports/mlb/scripts/select_daily_parlay.py's daily_parlay_*.json), for
comparison purposes only (mission section 4/15/16).

Named explicitly `legacy_parlay_control` / `old_parlay_diagnostic` per
section 15's instruction -- never "parlay" unqualified, so nothing here
can be mistaken for the new V2 authority. This module NEVER writes to the
old system's output and NEVER feeds anything back into
parlay_certification_v2 -- it only reads what the old CLI already
produced, for the comparison artifact in comparison.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LegacyParlayControl:
    """Diagnostic snapshot of the OLD system's selection -- CONTROL only,
    never authorization. `old_control_pair` may legitimately be None (the
    old system abstained, or produced no artifact for this date)."""

    available: bool
    old_control_pair: list[dict[str, Any]] | None
    old_control_probability: float | None
    old_control_quote: dict[str, Any] | None
    reason: str


def load_legacy_parlay_control(daily_parlay_json_path: Path) -> LegacyParlayControl:
    path = Path(daily_parlay_json_path)
    if not path.exists():
        return LegacyParlayControl(False, None, None, None, "old_parlay_diagnostic_artifact_not_found")
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return LegacyParlayControl(False, None, None, None, f"old_parlay_diagnostic_artifact_unreadable: {exc}")

    ticket = raw.get("selected_ticket") if isinstance(raw, dict) else None
    if not isinstance(ticket, dict) or not ticket.get("legs"):
        return LegacyParlayControl(True, None, None, None, "old_parlay_diagnostic_no_selected_ticket")

    legs = [
        {
            "player": leg.get("player_display_name") or leg.get("player"),
            "target": leg.get("target"),
            "line": leg.get("market_line"),
            "side": "OVER",  # the old system's consistency ticket is OVER-only by construction
        }
        for leg in ticket.get("legs", [])
    ]
    quote = {
        "combined_american_price": ticket.get("combined_american_price"),
        "sportsbook": ticket.get("sportsbook"),
    }
    return LegacyParlayControl(
        available=True,
        old_control_pair=legs,
        old_control_probability=ticket.get("projected_probability"),
        old_control_quote=quote,
        reason="old_parlay_diagnostic_loaded",
    )

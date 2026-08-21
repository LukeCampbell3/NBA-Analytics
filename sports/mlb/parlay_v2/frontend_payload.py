from __future__ import annotations

"""Additive embedding of the V2 parlay payload into the existing MLB web
export (mission section 9). Pure function, no side effects, importable in
isolation from the heavy export_web_prediction_payload.py module so it can
be unit tested directly.

Contract: `embed_parlays_v2` NEVER reads or writes any existing key in
`payload` (plays, parlay_summary, parlay_pairs, daily_parlay, summary,
policy_governance, ...) -- it only adds one new top-level key, `parlays`.
Old system's output is untouched; if the V2 JSON is missing/unreadable,
`parlays` reports a clear unavailable state rather than raising, so a
missing V2 artifact never breaks the singles export.
"""

import json
from pathlib import Path
from typing import Any

UNAVAILABLE_PARLAYS_V2 = {
    "system": "PARLAY_POLICY_V2",
    "policy_version": None,
    "policy_status": None,
    "eligible": None,
    "action": "ABSTAIN",
    "selected_parlay": None,
    "evidence_status": None,
    "abstain_reason": "PARLAY_V2_ARTIFACT_UNAVAILABLE",
}


def load_parlays_v2(parlay_v2_json_path: Path | None) -> dict[str, Any]:
    if parlay_v2_json_path is None or not Path(parlay_v2_json_path).exists():
        return dict(UNAVAILABLE_PARLAYS_V2)
    try:
        with open(parlay_v2_json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(UNAVAILABLE_PARLAYS_V2)
    if not isinstance(data, dict) or data.get("system") != "PARLAY_POLICY_V2":
        return dict(UNAVAILABLE_PARLAYS_V2)
    return data


def embed_parlays_v2(payload: dict[str, Any], parlay_v2_json_path: Path | None) -> dict[str, Any]:
    """Returns a NEW dict: `payload` plus a `parlays` key. Never mutates
    `payload` in place, and never touches any key already in it -- if
    `parlays` already exists (should never happen upstream), it is
    overwritten with the V2 value, everything else passes through
    identically."""
    result = dict(payload)
    result["parlays"] = load_parlays_v2(parlay_v2_json_path)
    return result

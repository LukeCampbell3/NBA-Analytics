#!/usr/bin/env python3
"""Real headshot enrichment for PARLAY_POLICY_V2 legs.

The main single-leg board already carries real headshot URLs (built in
export_web_prediction_payload.py from the same-day roster fetch that also
resolves each play's identity). The V2 parlay legs (parlays.selected_parlay,
parlays.shadow_candidate) are built by a separate pipeline
(sports/mlb/parlay_v2/) that only ever carries a player's display name and
a slug id -- no real MLB Stats API person_id, so no real headshot URL can
be built at selection time.

This is a small, additive, real enrichment step that runs after selection:
for every player name found in those leg dicts, it looks up the real
MLB Stats API person_id (reusing export_web_prediction_payload.py's own
roster-search machinery -- the same real, no-guessing lookup the main
board already trusts) and, only when a match is found, attaches the same
real player_headshot_url / player_headshot_fallback_url fields the main
board uses. A name that can't be resolved is left without a headshot
field -- the frontend's monogram fallback handles that honestly; nothing
is ever guessed.

Never touches any other key in daily_predictions.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
if str(MLB_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import export_web_prediction_payload as exporter  # noqa: E402

DEFAULT_DAILY_PREDICTIONS_PATH = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"

# Legs live at these dotted paths under payload["parlays"].
LEG_PATHS = (
    ("selected_parlay", "leg_1"),
    ("selected_parlay", "leg_2"),
    ("selected_parlay", "leg_3"),
    ("selected_parlay", "leg_4"),
    ("shadow_candidate", "leg_1"),
    ("shadow_candidate", "leg_2"),
)


def find_leg_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    parlays = payload.get("parlays")
    if isinstance(parlays, dict):
        for parent_key, leg_key in LEG_PATHS:
            parent = parlays.get(parent_key)
            if not isinstance(parent, dict):
                continue
            leg = parent.get(leg_key)
            if isinstance(leg, dict) and str(leg.get("player") or "").strip():
                legs.append(leg)

    # The legacy daily_parlay ticket's legs are no longer rendered on the
    # frontend at all (predictions.js's mergeLegacySoloBets was removed
    # 2026-08-29 -- its 62% leg floor was looser than the real singles
    # policy's 65% floor, so a merged leg could appear as a Solo Bet the
    # singles policy itself would have rejected). This enrichment is left
    # unchanged rather than removed here since that's a separate, real
    # decision about the enrichment pipeline this leg-visibility fix
    # didn't touch -- flagged as dead work worth trimming in a follow-up.
    ticket = (payload.get("daily_parlay") or {}).get("selected_ticket")
    if isinstance(ticket, dict) and isinstance(ticket.get("legs"), list):
        for leg in ticket["legs"]:
            if isinstance(leg, dict) and str(leg.get("player_display_name") or leg.get("player") or "").strip():
                legs.append(leg)

    return legs


def build_headshot_lookup(
    player_names: set[str], *, person_id_resolver=exporter.search_person_id_by_name
) -> dict[str, Optional[int]]:
    """Real MLB Stats API name search, one call per unique player name.
    Returns {player_name: person_id_or_None}. A None means the lookup
    genuinely found no match -- not a failure to guess around."""
    lookup: dict[str, Optional[int]] = {}
    for name in sorted(player_names):
        try:
            lookup[name] = person_id_resolver(name)
        except Exception:
            lookup[name] = None
    return lookup


def enrich_payload(
    payload: dict[str, Any], *, person_id_resolver=exporter.search_person_id_by_name
) -> dict[str, Any]:
    legs = find_leg_dicts(payload)
    if not legs:
        return payload
    names = {str(leg["player"]).strip() for leg in legs}
    lookup = build_headshot_lookup(names, person_id_resolver=person_id_resolver)
    for leg in legs:
        person_id = lookup.get(str(leg["player"]).strip())
        headshot_url = exporter.build_headshot_url(person_id)
        if not headshot_url:
            continue
        leg["player_headshot_url"] = headshot_url
        leg["player_headshot_fallback_url"] = exporter.build_headshot_fallback_url(person_id)
    return payload


def enrich_file(path: Path, *, person_id_resolver=exporter.search_person_id_by_name) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_payload(payload, person_id_resolver=person_id_resolver)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-predictions-path", type=Path, default=None, action="append",
        help=(
            "In-place enrich this daily_predictions.json's parlay legs "
            "with real headshot URLs. Repeatable -- pass once per real "
            "published copy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.daily_predictions_path or [DEFAULT_DAILY_PREDICTIONS_PATH]
    enriched_counts = {}
    for target in targets:
        if not target.exists():
            continue
        payload = enrich_file(target)
        legs = find_leg_dicts(payload)
        enriched_counts[str(target)] = sum(1 for leg in legs if leg.get("player_headshot_url"))
    print(json.dumps({"enriched": enriched_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

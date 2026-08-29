#!/usr/bin/env python3
"""Real, multi-region FanDuel "Add to Betslip" deep links for
already-published pitcher K-strikeout parlay legs
(pitcher_parlay_predictions.json).

run_mlb_pitcher_parlay_quality_daily.py's own generation flow already
threads a real single-region `sportsbook_deeplink` through every leg
(parlay.leg_a/leg_b, parlay.max_hit_control.leg_a/leg_b, and the flat
`legs` list) straight off FanduelPublicMlbProvider's real rows -- a
working link appears automatically at selection time for a viewer in
whichever single region the pipeline happened to fetch under.

Multi-region deep links (added 2026-08-29, real user report -- the exact
same class of bug enrich_parlay_leg_betslip.py fixed for PARLAY_POLICY_V2
pairs and the main single-bet board on 2026-08-27, and
enrich_same_game_betslip.py just fixed for same-game combos, never
extended to this product): FanDuel is a state-by-state licensed operator
-- each region is a genuinely separate sportsbook instance with its own
real marketId/selectionId for the identical player/market/line (see
fanduel_regions.py). The single-region deeplink this payload already
carries only actually adds to the betslip for a viewer whose real FanDuel
account is in whichever region the pipeline happened to fetch under (NJ
by default) -- every other real user got "Selection not added" on an
otherwise correctly formatted link. This live-fetches every real
FanDuel-licensed state (fanduel_regions.FANDUEL_LICENSED_STATES) and
attaches a real `deeplinks_by_region` map to every leg alongside the
original single-region `sportsbook_deeplink` (kept unchanged for backward
compatibility), reusing the exact same real multi-region player-prop
fetch machinery enrich_parlay_leg_betslip.py already built and validated
-- pitcher strikeouts are just another real player-prop market on the
same real FanDuel feed, so build_multi_region_odds_indexes() and
match_leg_to_regions() are imported directly here, not reimplemented.

Never touches any other field (probability, EV, pricing, authorization)
on any leg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
MLB_PARLAY_V2_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
for path in (MLB_SCRIPTS_ROOT, MLB_ODDS_PROVIDERS_ROOT, MLB_PARLAY_V2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from enrich_parlay_leg_betslip import build_multi_region_odds_indexes, match_leg_to_regions  # noqa: E402

DEFAULT_PITCHER_PARLAY_PREDICTIONS_PATH = (
    REPO_ROOT / "sports" / "mlb" / "web" / "data" / "pitcher_parlay_predictions.json"
)


def _as_matchable_leg(leg: dict[str, Any]) -> dict[str, Any]:
    """Adapts a pitcher-parlay leg's real fields (pitcher_name/line/side)
    to the shape match_leg_to_regions expects (player/target/line/side).
    Every leg in this product is a real pitcher-strikeouts prop -- unlike
    a PARLAY_POLICY_V2 leg, which can be any of several real targets --
    so `target` is always the real "K" market code."""
    return {
        "player": leg.get("pitcher_name"),
        "target": "K",
        "line": leg.get("line"),
        "side": leg.get("side"),
    }


def _iter_leg_dicts(payload: dict[str, Any]):
    """Every real leg dict this payload can carry a betslip link on --
    the published parlay's two legs, its max-hit-control comparison
    pair (same real shape, shown for transparency), and the flat `legs`
    list of every real candidate the selector considered."""
    parlay = payload.get("parlay")
    if isinstance(parlay, dict):
        for key in ("leg_a", "leg_b"):
            leg = parlay.get(key)
            if isinstance(leg, dict):
                yield leg
        control = parlay.get("max_hit_control")
        if isinstance(control, dict):
            for key in ("leg_a", "leg_b"):
                leg = control.get(key)
                if isinstance(leg, dict):
                    yield leg
    for leg in payload.get("legs") or []:
        if isinstance(leg, dict):
            yield leg


def enrich_payload(
    payload: dict[str, Any],
    *,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> dict[str, Any]:
    if not region_indexes:
        return payload
    for leg in _iter_leg_dicts(payload):
        leg["deeplinks_by_region"] = match_leg_to_regions(_as_matchable_leg(leg), region_indexes)
    return payload


def enrich_file(
    path: Path,
    *,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_payload(payload, region_indexes=region_indexes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitcher-parlay-predictions-path", type=Path, default=None, action="append",
        help="In-place enrich this pitcher_parlay_predictions.json's legs with real per-state FanDuel betslip URLs. Repeatable.",
    )
    parser.add_argument(
        "--disable-multi-region-betslip", action="store_true",
        help="Skip the real per-state (FANDUEL_LICENSED_STATES) fetch entirely -- for fast local iteration; the real "
        "pipeline leaves this enabled so every viewer's own state resolves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.pitcher_parlay_predictions_path or [DEFAULT_PITCHER_PARLAY_PREDICTIONS_PATH]
    region_indexes = None if args.disable_multi_region_betslip else build_multi_region_odds_indexes()
    region_coverage = {state: len(index) for state, index in (region_indexes or {}).items()}
    enriched_counts = {}
    for target in targets:
        if not target.exists():
            continue
        payload = enrich_file(target, region_indexes=region_indexes)
        enriched_counts[str(target)] = sum(1 for leg in _iter_leg_dicts(payload) if leg.get("deeplinks_by_region"))
    print(json.dumps({"legs_enriched": enriched_counts, "region_coverage": region_coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

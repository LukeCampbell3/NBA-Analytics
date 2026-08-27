#!/usr/bin/env python3
"""Real FanDuel "Add to Betslip" deep links for PARLAY_POLICY_V2 pairs.

The legacy daily_parlay pipeline (select_daily_parlay.py) already builds a
real, provider-issued multi-leg FanDuel deep link
(https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?marketId[i]=
...&selectionId[i]=...) from each leg's real sportsbook_deeplink -- the
same real per-selection deeplink FanduelPublicMlbProvider attaches to every
row it returns (see fanduel_public_mlb_provider.py's `sportsbook_deeplink`
field, itself just FanDuel's own documented marketId/selectionId query
format). That machinery already works; it was just never reused for the
newer PARLAY_POLICY_V2 pairs (parlays.selected_parlay /
parlays.shadow_candidate), which only ever carry a player name/target/
line/side -- no FanDuel marketId/selectionId of their own.

This is a small, additive, real enrichment step, structured exactly like
enrich_parlay_leg_headshots.py: for a V2 pair, live-fetch FanDuel's public
MLB player-prop odds (the same anonymous, no-auth feed the legacy pipeline
and the shadow/same-game boards already read), match each leg to a real
FanDuel row by (player name, market, line, side), and -- only when every
leg in the pair resolves to a real FanDuel selection -- attach the exact
same multi-leg betslip URL shape select_daily_parlay.py already builds and
validates. A pair where any leg can't be matched to a live FanDuel row is
left without a betslip URL; nothing is ever guessed or partially built.

Multi-region deep links (added 2026-08-27, real user report): FanDuel is a
state-by-state licensed operator -- fanduel_public_mlb_provider.py's own
`x-sportsbook-region` header changes which real marketId/selectionId
FanDuel's API returns for the identical player/market/line, because each
region is a genuinely separate sportsbook instance. A single-region link
(the site previously fetched NJ only) loads FanDuel's domain fine for any
viewer, but only actually adds to the betslip for a viewer whose real
account is in that same region -- this is the real, confirmed root cause
of "the link is there but doesn't add to my betslip". This module now
additionally live-fetches every real FanDuel-licensed state
(fanduel_regions.FANDUEL_LICENSED_STATES) and attaches a real
`deeplinks_by_region` map to every leg/play alongside the original
single-region `sportsbook_deeplink` (kept unchanged for backward
compatibility) -- see build_multi_region_odds_indexes().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
MLB_PARLAY_V2_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
for path in (MLB_SCRIPTS_ROOT, MLB_ODDS_PROVIDERS_ROOT, MLB_PARLAY_V2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_web_prediction_payload as exporter  # noqa: E402
from fanduel_public_mlb_provider import FanduelPublicMlbProvider  # noqa: E402
from fanduel_regions import DEFAULT_FALLBACK_REGION, FANDUEL_LICENSED_STATES  # noqa: E402
from fanduel_betslip import (  # noqa: E402
    FANDUEL_SPORTSBOOK_KEY,
    build_fanduel_betslip_url,
)

TARGET_MARKET_TYPES = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
}

PAIR_KEYS = ("selected_parlay", "shadow_candidate")
LEG_KEYS = ("leg_1", "leg_2", "leg_3", "leg_4")


def build_odds_index(odds_rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, str], str]:
    """Real FanDuel rows -> {(normalized_player, market_type, line, side): sportsbook_deeplink}."""
    index: dict[tuple[str, str, float, str], str] = {}
    for row in odds_rows:
        deeplink = str(row.get("sportsbook_deeplink") or "").strip()
        if not deeplink:
            continue
        try:
            line = float(row.get("line"))
        except (TypeError, ValueError):
            continue
        key = (
            exporter.normalize_player_name(row.get("player_name")),
            str(row.get("market_type") or "").strip(),
            line,
            str(row.get("side") or "").strip().lower(),
        )
        index[key] = deeplink
    return index


def build_multi_region_odds_indexes(
    states: tuple[str, ...] = FANDUEL_LICENSED_STATES,
    *,
    provider_factory: Callable[[str], Any] = None,
) -> dict[str, dict[tuple[str, str, float, str], str]]:
    """Real per-state FanDuel odds indexes -- one real live fetch per
    state (never per-leg; each state's full player-prop board is fetched
    once and matched against every leg locally, the same efficient
    pattern the single-region path already uses). A state whose real
    fetch fails or returns no odds is simply absent from the result --
    never a guessed/empty index standing in for a real one."""
    factory = provider_factory or (lambda region: FanduelPublicMlbProvider(region=region))
    indexes: dict[str, dict[tuple[str, str, float, str], str]] = {}
    for state in states:
        try:
            result = factory(state).collect_player_props()
        except Exception:
            continue
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        indexes[state] = build_odds_index(result.get("odds") or [])
    return indexes


def match_leg_to_regions(
    leg: dict[str, Any], region_indexes: dict[str, dict[tuple[str, str, float, str], str]]
) -> dict[str, str]:
    """{state: deeplink} for every real FanDuel-licensed state that has a
    live selection matching this leg -- a state with no match for this
    leg is simply absent, never filled with a guess or the wrong link."""
    deeplinks_by_region: dict[str, str] = {}
    for state, odds_index in region_indexes.items():
        deeplink = match_leg_to_deeplink(leg, odds_index)
        if deeplink:
            deeplinks_by_region[state] = deeplink
    return deeplinks_by_region


def match_leg_to_deeplink(leg: dict[str, Any], odds_index: dict[tuple[str, str, float, str], str]) -> Optional[str]:
    market_type = TARGET_MARKET_TYPES.get(str(leg.get("target") or "").strip())
    if not market_type:
        return None
    try:
        line = float(leg.get("line"))
    except (TypeError, ValueError):
        return None
    key = (
        exporter.normalize_player_name(leg.get("player")),
        market_type,
        line,
        str(leg.get("side") or "").strip().lower(),
    )
    return odds_index.get(key)


def enrich_pair(
    pair: dict[str, Any],
    odds_index: dict[tuple[str, str, float, str], str],
    *,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> None:
    legs = [pair.get(key) for key in LEG_KEYS if isinstance(pair.get(key), dict)]
    if not legs:
        return
    deeplinks: list[Optional[str]] = []
    for leg in legs:
        deeplink = match_leg_to_deeplink(leg, odds_index)
        if deeplink:
            leg["sportsbook_deeplink"] = deeplink
        deeplinks.append(deeplink)
        if region_indexes:
            leg["deeplinks_by_region"] = match_leg_to_regions(leg, region_indexes)

    if not all(deeplinks):
        pair["betslip"] = {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "unavailable",
            "reason": "one_or_more_legs_have_no_live_fanduel_selection",
        }
        return

    synthetic_legs = [
        {"selected_sportsbook_key": FANDUEL_SPORTSBOOK_KEY, "sportsbook_deeplink": deeplink}
        for deeplink in deeplinks
    ]
    url = build_fanduel_betslip_url(synthetic_legs)
    if url is None:
        pair["betslip"] = {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "unavailable",
            "reason": "provider_selection_links_did_not_validate",
        }
        return
    pair["betslip"] = {
        "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
        "sportsbook": "FanDuel",
        "status": "ready",
        "leg_count": len(legs),
        "url": url,
        "source": "direct_fanduel_public_market_ids",
    }
    pair["betslip_url"] = url


def enrich_single_play(
    play: dict[str, Any],
    odds_index: dict[tuple[str, str, float, str], str],
    *,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> None:
    """A real single-leg FanDuel deep link for one main-board pick --
    same real (player, market, line, side) match as a parlay leg, just
    attached directly rather than combined into a multi-leg URL. Fields
    differ from a parlay leg dict (player_display_name/market_line/
    direction vs. player/line/side), so this adapts rather than reusing
    match_leg_to_deeplink's leg shape directly."""
    normalized_leg = {
        "player": play.get("player_display_name") or play.get("player"),
        "target": play.get("target"),
        "line": play.get("market_line"),
        "side": play.get("direction"),
    }
    deeplink = match_leg_to_deeplink(normalized_leg, odds_index)
    if deeplink:
        play["sportsbook_deeplink"] = deeplink
    if region_indexes:
        play["deeplinks_by_region"] = match_leg_to_regions(normalized_leg, region_indexes)


def enrich_payload(
    payload: dict[str, Any],
    *,
    odds_fetcher: Callable[[], dict[str, Any]] = None,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> dict[str, Any]:
    parlays = payload.get("parlays")
    pairs = [parlays.get(key) for key in PAIR_KEYS if isinstance(parlays, dict) and isinstance(parlays.get(key), dict)]
    plays = [p for p in (payload.get("plays") or []) if isinstance(p, dict)]
    if not pairs and not plays:
        return payload

    fetch = odds_fetcher or (lambda: FanduelPublicMlbProvider().collect_player_props())
    result = fetch()
    if result.get("status") != "success":
        return payload
    odds_index = build_odds_index(result.get("odds") or [])
    for pair in pairs:
        enrich_pair(pair, odds_index, region_indexes=region_indexes)
    for play in plays:
        enrich_single_play(play, odds_index, region_indexes=region_indexes)
    return payload


def enrich_file(
    path: Path,
    *,
    odds_fetcher: Callable[[], dict[str, Any]] = None,
    region_indexes: dict[str, dict[tuple[str, str, float, str], str]] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_payload(payload, odds_fetcher=odds_fetcher, region_indexes=region_indexes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-predictions-path", type=Path, default=None, action="append",
        help="In-place enrich this daily_predictions.json's parlay pairs with a real FanDuel betslip URL. Repeatable.",
    )
    parser.add_argument(
        "--disable-multi-region-betslip", action="store_true",
        help="Skip the real per-state (FANDUEL_LICENSED_STATES) fetch and only attach the single-region sportsbook_deeplink -- "
        "for fast local iteration; the real pipeline leaves this enabled so every viewer's own state resolves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.daily_predictions_path or []
    region_indexes = None if args.disable_multi_region_betslip else build_multi_region_odds_indexes()
    ready_counts = {}
    region_coverage = {state: len(index) for state, index in (region_indexes or {}).items()}
    for target in targets:
        if not target.exists():
            continue
        payload = enrich_file(target, region_indexes=region_indexes)
        parlays = payload.get("parlays") or {}
        ready_counts[str(target)] = sum(
            1 for key in PAIR_KEYS
            if isinstance(parlays.get(key), dict) and (parlays[key].get("betslip") or {}).get("status") == "ready"
        )
    print(json.dumps({"betslip_ready": ready_counts, "region_coverage": region_coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

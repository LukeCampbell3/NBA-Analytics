#!/usr/bin/env python3
"""Real FanDuel "Add to Betslip" deep links for already-published
same-game combos (same_game_predictions.json).

select_mlb_same_game_bets.py's own generation flow already threads a
real `sportsbook_deeplink` through each SameGameLeg (see its
_build_legs_for_market, which reads home_moneyline_deeplink /
away_moneyline_deeplink / over_deeplink / under_deeplink straight off
FanduelPublicMlbTeamMarketProvider's real rows) and computes a real
combo-level `betslip` from SameGameComboCandidate.betslip -- for any
run that actually uses that provider, deep links appear automatically
at selection time, no extra step needed.

This script exists for the same reason enrich_parlay_leg_headshots.py /
enrich_parlay_leg_betslip.py do: a payload already published by an
older pipeline run (one that predates this deeplink support, or ran
before a code merge landed) has no deeplinks on it, and re-running the
full same-game selection pipeline would re-simulate and re-price every
game -- overkill, and risks silently drifting the already-published
probabilities/EV away from what was actually shown. Instead, this
fetches the SAME real, live, no-auth FanDuel team-market feed
select_mlb_same_game_bets.py itself reads, matches each already-
published leg to a live row by (real game, real market, real side,
real line), and -- only when a leg's real line still matches today's
live line exactly -- attaches the real deeplink and recomputes the
combo's betslip using the exact same fanduel_betslip.build_fanduel_
betslip_url helper every other MLB parlay product uses. A leg whose
line has moved, or that can't be matched to a live FanDuel row at all,
is left without a deeplink; nothing is ever guessed or partially built.

Never touches any other field (probability, EV, pricing, authorization)
on any combo.

Multi-region deep links (added 2026-08-29, real user report -- the exact
same class of bug enrich_parlay_leg_betslip.py fixed for PARLAY_POLICY_V2
pairs and the main single-bet board on 2026-08-27, never extended to this
product): FanDuel is a state-by-state licensed operator -- each region is
a genuinely separate sportsbook instance with its own real marketId/
selectionId for the identical player/market/line (see fanduel_regions.py).
The single-region deeplink this module already attached only actually
adds to the betslip for a viewer whose real FanDuel account is in
whichever region the pipeline happened to fetch under (NJ by default) --
every other real user got "Selection not added" on an otherwise correctly
formatted link. This now additionally live-fetches every real FanDuel-
licensed state (fanduel_regions.FANDUEL_LICENSED_STATES) and attaches a
real `deeplinks_by_region` map to each leg alongside the original
single-region `sportsbook_deeplink` (kept unchanged for backward
compatibility) -- see build_multi_region_deeplink_indexes(). Also now
covers `exploratory_ev_candidates`, not just `combo_candidates`: the
tight-quality headline gate (same_game_quality_selector.py) frequently
leaves combo_candidates empty while exploratory_ev_candidates carries the
real, priced, published-to-viewers combos -- those need real working
links exactly as much as a headline combo does.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
MLB_PARLAY_V2_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
for path in (MLB_SCRIPTS_ROOT, MLB_ODDS_PROVIDERS_ROOT, MLB_PARLAY_V2_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fanduel_betslip import FANDUEL_SPORTSBOOK_KEY, build_fanduel_betslip_url  # noqa: E402
from fanduel_public_mlb_team_market_provider import FanduelPublicMlbTeamMarketProvider  # noqa: E402
from fanduel_regions import FANDUEL_LICENSED_STATES  # noqa: E402
from run_mlb_same_game_daily import STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION  # noqa: E402

DEFAULT_SAME_GAME_PREDICTIONS_PATH = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "same_game_predictions.json"
COMBO_LIST_KEYS = ("combo_candidates", "exploratory_ev_candidates")
MULTI_REGION_FETCH_WORKERS = 10


def _index_key(home_team: str, away_team: str, market: str) -> tuple[str, str, str]:
    return (str(home_team or ""), str(away_team or ""), str(market or ""))


def match_leg_to_deeplink(leg: dict[str, Any], row: Optional[dict[str, Any]]) -> Optional[str]:
    if row is None:
        return None
    market = str(leg.get("market") or "")
    side = str(leg.get("side") or "").strip().lower()
    if market == "moneyline":
        if side == "home":
            return row.get("home_moneyline_deeplink")
        if side == "away":
            return row.get("away_moneyline_deeplink")
        return None
    # game_total / first_5_innings_total -- only trust a live row whose
    # line still matches exactly what was already published; a market
    # that has moved since publication is left unlinked, never relinked
    # to a different real line than the one shown.
    try:
        published_line = float(leg.get("line"))
        live_line = float(row.get("line"))
    except (TypeError, ValueError):
        return None
    if published_line != live_line:
        return None
    if side == "over":
        return row.get("over_deeplink")
    if side == "under":
        return row.get("under_deeplink")
    return None


def build_multi_region_deeplink_indexes(
    states: tuple[str, ...] = FANDUEL_LICENSED_STATES,
    *,
    provider_factory: Callable[[str], Any] = None,
    max_workers: int = MULTI_REGION_FETCH_WORKERS,
) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    """Real per-state {(home_espn, away_espn, market): row} indexes --
    one real live fetch per state (never per-leg), same real fail-open-
    per-state contract as enrich_parlay_leg_betslip.build_multi_region_
    odds_indexes(): a state whose real fetch fails or returns no odds is
    simply absent from the result, never a guessed/empty index. Fetches
    run concurrently since this is real network-bound work."""
    factory = provider_factory or (lambda region: FanduelPublicMlbTeamMarketProvider(region=region))
    indexes: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    if not states:
        return indexes

    def fetch_one(state: str) -> tuple[str, dict[tuple[str, str, str], dict[str, Any]] | None]:
        try:
            result = factory(state).collect_team_market_odds()
        except Exception:
            return state, None
        if not isinstance(result, dict) or result.get("status") != "success":
            return state, None
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in result.get("odds") or []:
            home_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(str(row.get("home_team") or ""))
            away_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(str(row.get("away_team") or ""))
            if not home_espn or not away_espn:
                continue
            index[_index_key(home_espn, away_espn, row.get("target"))] = row
        return state, index

    with ThreadPoolExecutor(max_workers=min(max_workers, len(states))) as pool:
        futures = [pool.submit(fetch_one, state) for state in states]
        for future in as_completed(futures):
            state, index = future.result()
            if index is not None:
                indexes[state] = index
    return indexes


def match_combo_leg_to_regions(
    leg: dict[str, Any],
    home_team: str,
    away_team: str,
    region_indexes: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
) -> dict[str, str]:
    """{state: deeplink} for every real FanDuel-licensed state whose live
    board has a selection matching this leg -- a state with no match is
    simply absent, never filled with a guess or the wrong link."""
    deeplinks_by_region: dict[str, str] = {}
    for state, index in region_indexes.items():
        row = index.get(_index_key(home_team, away_team, leg.get("market")))
        deeplink = match_leg_to_deeplink(leg, row)
        if deeplink:
            deeplinks_by_region[state] = deeplink
    return deeplinks_by_region


def enrich_combo(
    combo: dict[str, Any],
    deeplink_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    region_indexes: dict[str, dict[tuple[str, str, str], dict[str, Any]]] | None = None,
) -> bool:
    leg_a, leg_b = combo.get("leg_a"), combo.get("leg_b")
    if not isinstance(leg_a, dict) or not isinstance(leg_b, dict):
        return False

    home_team, away_team = combo.get("home_team"), combo.get("away_team")
    row_a = deeplink_index.get(_index_key(home_team, away_team, leg_a.get("market")))
    row_b = deeplink_index.get(_index_key(home_team, away_team, leg_b.get("market")))
    deeplink_a = match_leg_to_deeplink(leg_a, row_a)
    deeplink_b = match_leg_to_deeplink(leg_b, row_b)
    if deeplink_a:
        leg_a["sportsbook_deeplink"] = deeplink_a
    if deeplink_b:
        leg_b["sportsbook_deeplink"] = deeplink_b
    if region_indexes:
        leg_a["deeplinks_by_region"] = match_combo_leg_to_regions(leg_a, home_team, away_team, region_indexes)
        leg_b["deeplinks_by_region"] = match_combo_leg_to_regions(leg_b, home_team, away_team, region_indexes)

    if not (deeplink_a and deeplink_b):
        combo["betslip"] = {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "unavailable",
            "reason": "one_or_more_legs_have_no_live_fanduel_selection",
        }
        combo.pop("betslip_url", None)
        return False

    legs = [
        {"selected_sportsbook_key": leg_a.get("sportsbook"), "sportsbook_deeplink": deeplink_a},
        {"selected_sportsbook_key": leg_b.get("sportsbook"), "sportsbook_deeplink": deeplink_b},
    ]
    url = build_fanduel_betslip_url(legs)
    if url is None:
        combo["betslip"] = {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "unavailable",
            "reason": "provider_selection_links_did_not_validate",
        }
        combo.pop("betslip_url", None)
        return False

    combo["betslip"] = {
        "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
        "sportsbook": "FanDuel",
        "status": "ready",
        "leg_count": 2,
        "url": url,
        "source": "direct_fanduel_public_market_ids",
    }
    combo["betslip_url"] = url
    return True


def _all_combos(games: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        combo
        for game in (games or [])
        if isinstance(game, dict)
        for key in COMBO_LIST_KEYS
        for combo in (game.get(key) or [])
        if isinstance(combo, dict)
    ]


def enrich_payload(
    payload: dict[str, Any],
    *,
    odds_fetcher: Callable[[], dict[str, Any]] = None,
    region_indexes: dict[str, dict[tuple[str, str, str], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    combos = _all_combos(payload.get("games"))
    if not combos:
        return payload

    fetch = odds_fetcher or (lambda: FanduelPublicMlbTeamMarketProvider().collect_team_market_odds())
    result = fetch()
    if result.get("status") != "success":
        return payload
    deeplink_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in result.get("odds") or []:
        home_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(str(row.get("home_team") or ""))
        away_espn = STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION.get(str(row.get("away_team") or ""))
        if not home_espn or not away_espn:
            continue
        deeplink_index[_index_key(home_espn, away_espn, row.get("target"))] = row

    for combo in combos:
        enrich_combo(combo, deeplink_index, region_indexes=region_indexes)
    return payload


def enrich_file(
    path: Path,
    *,
    odds_fetcher: Callable[[], dict[str, Any]] = None,
    region_indexes: dict[str, dict[tuple[str, str, str], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_payload(payload, odds_fetcher=odds_fetcher, region_indexes=region_indexes)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--same-game-predictions-path", type=Path, default=None, action="append",
        help="In-place enrich this same_game_predictions.json's combos with a real FanDuel betslip URL. Repeatable.",
    )
    parser.add_argument(
        "--disable-multi-region-betslip", action="store_true",
        help="Skip the real per-state (FANDUEL_LICENSED_STATES) fetch and only attach the single-region sportsbook_deeplink -- "
        "for fast local iteration; the real pipeline leaves this enabled so every viewer's own state resolves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.same_game_predictions_path or [DEFAULT_SAME_GAME_PREDICTIONS_PATH]
    region_indexes = None if args.disable_multi_region_betslip else build_multi_region_deeplink_indexes()
    region_coverage = {state: len(index) for state, index in (region_indexes or {}).items()}
    ready_counts = {}
    for target in targets:
        if not target.exists():
            continue
        payload = enrich_file(target, region_indexes=region_indexes)
        ready_counts[str(target)] = sum(
            1 for combo in _all_combos(payload.get("games")) if (combo.get("betslip") or {}).get("status") == "ready"
        )
    print(json.dumps({"betslip_ready": ready_counts, "region_coverage": region_coverage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

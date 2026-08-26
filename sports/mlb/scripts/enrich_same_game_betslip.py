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

from fanduel_betslip import FANDUEL_SPORTSBOOK_KEY, build_fanduel_betslip_url  # noqa: E402
from fanduel_public_mlb_team_market_provider import FanduelPublicMlbTeamMarketProvider  # noqa: E402
from run_mlb_same_game_daily import STATSAPI_FULL_NAME_TO_ESPN_ABBREVIATION  # noqa: E402

DEFAULT_SAME_GAME_PREDICTIONS_PATH = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "same_game_predictions.json"


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


def enrich_combo(combo: dict[str, Any], deeplink_index: dict[tuple[str, str, str], dict[str, Any]]) -> bool:
    leg_a, leg_b = combo.get("leg_a"), combo.get("leg_b")
    if not isinstance(leg_a, dict) or not isinstance(leg_b, dict):
        return False

    row_a = deeplink_index.get(_index_key(combo.get("home_team"), combo.get("away_team"), leg_a.get("market")))
    row_b = deeplink_index.get(_index_key(combo.get("home_team"), combo.get("away_team"), leg_b.get("market")))
    deeplink_a = match_leg_to_deeplink(leg_a, row_a)
    deeplink_b = match_leg_to_deeplink(leg_b, row_b)
    if deeplink_a:
        leg_a["sportsbook_deeplink"] = deeplink_a
    if deeplink_b:
        leg_b["sportsbook_deeplink"] = deeplink_b

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


def enrich_payload(
    payload: dict[str, Any], *, odds_fetcher: Callable[[], dict[str, Any]] = None,
) -> dict[str, Any]:
    games = payload.get("games")
    combos = [
        combo
        for game in (games or [])
        if isinstance(game, dict)
        for combo in (game.get("combo_candidates") or [])
        if isinstance(combo, dict)
    ]
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
        enrich_combo(combo, deeplink_index)
    return payload


def enrich_file(path: Path, *, odds_fetcher: Callable[[], dict[str, Any]] = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_payload(payload, odds_fetcher=odds_fetcher)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--same-game-predictions-path", type=Path, default=None, action="append",
        help="In-place enrich this same_game_predictions.json's combos with a real FanDuel betslip URL. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.same_game_predictions_path or [DEFAULT_SAME_GAME_PREDICTIONS_PATH]
    ready_counts = {}
    for target in targets:
        if not target.exists():
            continue
        payload = enrich_file(target)
        combos = [
            combo
            for game in (payload.get("games") or [])
            if isinstance(game, dict)
            for combo in (game.get("combo_candidates") or [])
            if isinstance(combo, dict)
        ]
        ready_counts[str(target)] = sum(1 for combo in combos if (combo.get("betslip") or {}).get("status") == "ready")
    print(json.dumps({"betslip_ready": ready_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

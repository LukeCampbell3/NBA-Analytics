#!/usr/bin/env python3
"""Real FanDuel multi-leg "Add to Betslip" URL construction -- pure,
dependency-free (no sklearn/pandas/etc, safe to import from anywhere).

This is the one real, already-production-proven construction this repo
has for combining several real single-leg FanDuel deep links
(https://sportsbook.fanduel.com/addToBetslip?marketId=X&selectionId=Y,
themselves built straight from FanDuel's own public odds feed -- see
fanduel_public_mlb_provider.py / fanduel_public_mlb_team_market_provider.py)
into one real multi-leg deep link
(https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?
marketId[0]=...&selectionId[0]=...&marketId[1]=...&selectionId[1]=...).
Originally written for select_daily_parlay.py's legacy ticket; reused
identically (never re-derived) by enrich_parlay_leg_betslip.py for
PARLAY_POLICY_V2 pairs and by select_mlb_same_game_bets.py for same-game
combos -- every real MLB parlay product this repo builds a deep link for
goes through these same two functions.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

FANDUEL_SPORTSBOOK_KEY = "fanduel"
FANDUEL_BETSLIP_ENDPOINT = "https://account.sportsbook.fanduel.com/sportsbook/addToBetslip"
FANDUEL_DEEPLINK_HOSTS = {"account.sportsbook.fanduel.com", "sportsbook.fanduel.com"}
FANDUEL_ID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
MIN_LEGS = 2


def parse_fanduel_selection_deeplink(value: object) -> tuple[str, str] | None:
    """Extract provider-issued FanDuel IDs without accepting arbitrary redirect URLs."""
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return None
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() not in FANDUEL_DEEPLINK_HOSTS
        or not parsed.path.lower().endswith("/addtobetslip")
    ):
        return None
    query = parse_qs(parsed.query, keep_blank_values=True)
    market_ids = query.get("marketId") or query.get("marketId[0]") or []
    selection_ids = query.get("selectionId") or query.get("selectionId[0]") or []
    if len(market_ids) != 1 or len(selection_ids) != 1:
        return None
    market_id = str(market_ids[0]).strip()
    selection_id = str(selection_ids[0]).strip()
    if not FANDUEL_ID_PATTERN.fullmatch(market_id) or not FANDUEL_ID_PATTERN.fullmatch(selection_id):
        return None
    return market_id, selection_id


def build_fanduel_betslip_url(legs: list[dict[str, Any]]) -> str | None:
    """Each leg needs selected_sportsbook_key == "fanduel" and a real
    sportsbook_deeplink. Returns None (never a partial link) unless
    every leg resolves to a distinct real FanDuel selection and there
    are at least MIN_LEGS of them."""
    selections: list[tuple[str, str]] = []
    for leg in legs:
        if str(leg.get("selected_sportsbook_key") or "").strip().lower() != FANDUEL_SPORTSBOOK_KEY:
            return None
        selection = parse_fanduel_selection_deeplink(leg.get("sportsbook_deeplink"))
        if selection is None or selection in selections:
            return None
        selections.append(selection)
    if len(selections) < MIN_LEGS:
        return None
    params: list[tuple[str, str]] = []
    for index, (market_id, selection_id) in enumerate(selections):
        params.extend(
            [
                (f"marketId[{index}]", market_id),
                (f"selectionId[{index}]", selection_id),
            ]
        )
    return f"{FANDUEL_BETSLIP_ENDPOINT}?{urlencode(params)}"

#!/usr/bin/env python3
"""FanDuel multi-leg "Add to Betslip" URL construction -- pure,
dependency-free (no sklearn/pandas/etc, safe to import from anywhere).

Combines several real single-leg FanDuel deep links
(https://sportsbook.fanduel.com/addToBetslip?marketId=X&selectionId=Y,
themselves built straight from FanDuel's own public odds feed -- see
fanduel_public_mlb_provider.py / fanduel_public_mlb_team_market_provider.py,
and independently confirmed real -- this is the literal URL FanDuel's own
feed returns) into a hypothesized multi-leg deep link
(https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?
marketId[0]=...&selectionId[0]=...&marketId[1]=...&selectionId[1]=...).

CAUTION -- this combined-URL scheme has NEVER been confirmed real. It was
extrapolated from the single-leg format, not sourced from FanDuel's own
API or any public documentation, and a real, logged-in device test of a
same-game combo's build_fanduel_betslip_url() output failed with
FanDuel's "Selection not added ... network issue, or it is no longer
available" error (2026-08-26). No frontend renders this combined URL
any more for exactly that reason -- every leg gets its own real,
independently-verified single-leg link instead (see CardVault.
renderLegCard's betslipUrl param). This function and its callers
(select_daily_parlay.py's legacy ticket, enrich_parlay_leg_betslip.py,
select_mlb_same_game_bets.py) still compute a `betslip`/`betslip_url`
value for diagnostic/audit purposes, but that value should not be
surfaced to a user as a working link until the real multi-leg scheme
(if one exists) is actually found and confirmed.
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

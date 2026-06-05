from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any


MARKET_KEY_BY_TARGET = {
    "PTS": "player_points",
    "POINTS": "player_points",
    "TRB": "player_rebounds",
    "REB": "player_rebounds",
    "REBOUNDS": "player_rebounds",
    "AST": "player_assists",
    "ASSISTS": "player_assists",
}


LINK_QUALITY_RANK = {
    "outcome": 4,
    "market": 3,
    "event": 2,
    "bookmaker": 1,
    "missing": 0,
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _date_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return text[:10]


def choose_bet_link(
    outcome: dict[str, Any] | None,
    market: dict[str, Any] | None,
    bookmaker: dict[str, Any] | None,
) -> tuple[str, str]:
    outcome = outcome or {}
    market = market or {}
    bookmaker = bookmaker or {}
    for quality, payload in (("outcome", outcome), ("market", market), ("event", bookmaker), ("bookmaker", bookmaker)):
        link = payload.get("link") or payload.get("betslip_link") or payload.get("url")
        if link:
            return str(link), quality
    return "", "missing"


def _line_matches(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except Exception:
        return False


def match_play_to_catalog(
    play: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    sport: str = "",
) -> dict[str, Any] | None:
    target = str(play.get("target") or play.get("market_type") or "").upper()
    desired_market = MARKET_KEY_BY_TARGET.get(target, target.lower())
    desired_side = str(play.get("direction") or play.get("side") or "").upper()
    desired_player = _norm(play.get("player_display_name") or play.get("player") or play.get("player_name"))
    desired_date = _date_token(play.get("market_date") or play.get("game_date") or play.get("event_date"))
    desired_home = str(play.get("market_home_team") or play.get("home_team_code") or "").upper()
    desired_away = str(play.get("market_away_team") or play.get("away_team_code") or "").upper()

    best: tuple[int, dict[str, Any]] | None = None
    for row in catalog:
        if desired_market and str(row.get("market_key") or "").lower() != desired_market:
            continue
        if desired_side and str(row.get("outcome_side") or row.get("side") or "").upper() != desired_side:
            continue
        if desired_player and _norm(row.get("player") or row.get("player_name")) != desired_player:
            continue
        if play.get("market_line") is not None and not _line_matches(play.get("market_line"), row.get("line")):
            continue

        score = 0
        if desired_date and _date_token(row.get("event_date") or row.get("game_date")) == desired_date:
            score += 20
        if desired_home and str(row.get("home_team_code") or "").upper() == desired_home:
            score += 10
        if desired_away and str(row.get("away_team_code") or "").upper() == desired_away:
            score += 10
        score += LINK_QUALITY_RANK.get(str(row.get("link_quality") or "missing").lower(), 0)
        if sport and str(row.get("sport") or sport).lower() == sport.lower():
            score += 1
        if best is None or score > best[0]:
            best = (score, row)
    return dict(best[1]) if best is not None else None


def build_parlay_sportsbook_options(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_leg_keys = {str(leg.get("play_key") or leg.get("leg_id") or idx) for idx, leg in enumerate(legs)}
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, leg in enumerate(legs):
        bookmaker = str(leg.get("bookmaker") or "").strip()
        if not bookmaker:
            continue
        by_book[bookmaker].append({**leg, "_leg_key": str(leg.get("play_key") or leg.get("leg_id") or idx)})

    options: list[dict[str, Any]] = []
    for bookmaker, book_legs in by_book.items():
        covered = sorted({leg["_leg_key"] for leg in book_legs})
        title = next((str(leg.get("bookmaker_title")) for leg in book_legs if leg.get("bookmaker_title")), bookmaker)
        links = [
            {
                "play_key": leg["_leg_key"],
                "betslip_link": leg.get("betslip_link") or leg.get("link") or "",
                "link_quality": leg.get("link_quality", "outcome" if leg.get("betslip_link") else "missing"),
            }
            for leg in book_legs
        ]
        options.append(
            {
                "bookmaker": bookmaker,
                "bookmaker_title": title,
                "covered_leg_count": len(covered),
                "total_leg_count": len(unique_leg_keys),
                "complete": len(covered) == len(unique_leg_keys),
                "links": links,
            }
        )
    return sorted(options, key=lambda row: (not row["complete"], -int(row["covered_leg_count"]), str(row["bookmaker"])))


def enrich_parlay_payload_with_sportsbooks(parlay_payload: dict[str, Any], *, sport: str = "") -> dict[str, Any]:
    out = dict(parlay_payload)
    plays = [dict(play) for play in out.get("plays", [])]
    play_by_key = {str(play.get("play_key")): play for play in plays if play.get("play_key") is not None}
    parlays = []
    for pair in out.get("pairs", []):
        leg_rows = []
        for leg in pair.get("legs", []):
            key = str(leg.get("play_key") or "")
            if key in play_by_key:
                leg_rows.append(play_by_key[key])
        options = build_parlay_sportsbook_options(leg_rows)
        parlays.append(
            {
                **pair,
                "sportsbook_options": options,
                "recommended_sportsbook": options[0] if options else None,
                "sport": sport,
            }
        )
    out["parlay_board"] = {"parlays": parlays, "sport": sport}
    return out

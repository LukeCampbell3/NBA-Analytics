#!/usr/bin/env python3
"""Fail-closed publication-time identity/lineup gate for MLB batter picks.

This module is publication-only negative authority. It does not alter frozen
model evidence. Before a current batter pick is rendered, resolve the player
against the exact MLB game, require carried team/opponent/home-away identity to
agree with that game when present, require a confirmed starting batting-order
role, and require a live sportsbook selection/deeplink.

Both the primary ``plays`` surface and the V4 shadow surface pass through the
same gate. This prevents a valid V4 overlay from coexisting with stale or
incomplete primary player rows and keeps player/game/lineup identity consistent
across the whole published singles board.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
MIRROR_BOARDS = (
    REPO_ROOT / "dist/mlb/data/daily_predictions.json",
    REPO_ROOT / "paywall/private-content/app/mlb/data/daily_predictions.json",
)
MLB_GAME_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
BATTER_STARTER_REQUIRED_TARGETS = {"H", "TB", "R", "RBI", "HR"}


def _normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _play_name(play: dict[str, Any]) -> str:
    return str(play.get("player") or play.get("player_display_name") or "").strip()


def _has_live_selection(play: dict[str, Any]) -> bool:
    return bool(
        play.get("sportsbook_deeplink")
        or play.get("selected_sportsbook_deeplink")
        or play.get("deeplinks_by_region")
        or play.get("selected_deeplinks_by_region")
    )


def _fetch_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "NBA-Analytics/1.0 (+live-player-identity-gate)"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_game_context(feed: dict[str, Any]) -> dict[str, Any]:
    """Normalize the minimum MLB feed state needed for exact player identity."""
    game_data = feed.get("gameData") or {}
    boxscore = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    teams = game_data.get("teams") or {}
    context: dict[str, Any] = {"sides": {}, "players_by_name": {}}

    for side in ("away", "home"):
        team = teams.get(side) or {}
        box_team = boxscore.get(side) or {}
        abbreviation = str(team.get("abbreviation") or team.get("teamCode") or "").upper()
        batting_order = [int(value) for value in (box_team.get("battingOrder") or []) if str(value).isdigit()]
        starting_order = {player_id: index + 1 for index, player_id in enumerate(batting_order)}
        starting_ids = set(starting_order)
        roster_ids: set[int] = set()
        side_players: list[dict[str, Any]] = []
        for player_row in (box_team.get("players") or {}).values():
            person = player_row.get("person") or {}
            player_id = person.get("id")
            name = str(person.get("fullName") or "").strip()
            try:
                player_id = int(player_id)
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            roster_ids.add(player_id)
            entry = {
                "player_id": player_id,
                "player": name,
                "side": side,
                "team": abbreviation,
                "is_starter": player_id in starting_ids,
                "batting_order_position": starting_order.get(player_id),
            }
            side_players.append(entry)
            context["players_by_name"].setdefault(_normalize_name(name), []).append(entry)
        context["sides"][side] = {
            "team": abbreviation,
            "starting_ids": starting_ids,
            "starting_order": starting_order,
            "roster_ids": roster_ids,
            "players": side_players,
        }
    return context


def _validate_play(play: dict[str, Any], context: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if context is None:
        return None, "GAME_CONTEXT_UNAVAILABLE"

    player_name = _play_name(play)
    if not player_name:
        return None, "PLAYER_NAME_MISSING"
    matches = list((context.get("players_by_name") or {}).get(_normalize_name(player_name), []))
    if len(matches) != 1:
        return None, "PLAYER_GAME_IDENTITY_MISMATCH" if not matches else "PLAYER_IDENTITY_AMBIGUOUS"

    match = matches[0]
    side = str(match["side"])
    opposite = "home" if side == "away" else "away"
    side_state = (context.get("sides") or {}).get(side) or {}
    opposite_state = (context.get("sides") or {}).get(opposite) or {}
    actual_team = str(side_state.get("team") or match.get("team") or "").upper()
    actual_opponent = str(opposite_state.get("team") or "").upper()
    target = str(play.get("target") or "").upper()

    carried_team = str(play.get("team") or "").strip().upper()
    carried_opponent = str(play.get("opponent") or "").strip().upper()
    carried_is_home = str(play.get("is_home") or "").strip()
    carried_player_id = str(play.get("player_id") or "").strip()
    if carried_team and actual_team and carried_team != actual_team:
        return None, "TEAM_GAME_IDENTITY_MISMATCH"
    if carried_opponent and actual_opponent and carried_opponent != actual_opponent:
        return None, "OPPONENT_GAME_IDENTITY_MISMATCH"
    if carried_is_home in {"0", "1"}:
        expected_is_home = "1" if side == "home" else "0"
        if carried_is_home != expected_is_home:
            return None, "HOME_AWAY_IDENTITY_MISMATCH"
    if carried_player_id.isdigit() and int(carried_player_id) != int(match["player_id"]):
        return None, "PLAYER_IDENTITY_MISMATCH"

    if target in BATTER_STARTER_REQUIRED_TARGETS:
        starting_ids = set(side_state.get("starting_ids") or set())
        if not starting_ids:
            return None, "STARTING_LINEUP_UNCONFIRMED"
        if int(match["player_id"]) not in starting_ids:
            return None, "PLAYER_NOT_IN_STARTING_LINEUP"

    execution_status = str(play.get("execution_status") or "").upper()
    if execution_status and execution_status != "LIVE_SELECTION_AVAILABLE":
        return None, "LIVE_SELECTION_UNAVAILABLE"
    if not _has_live_selection(play):
        return None, "LIVE_SELECTION_UNAVAILABLE"

    enriched = dict(play)
    enriched.update(
        {
            "player": str(match.get("player") or player_name),
            "player_display_name": str(match.get("player") or player_name),
            "player_id": int(match["player_id"]),
            "team": actual_team,
            "opponent": actual_opponent,
            "is_home": "1" if side == "home" else "0",
            "batting_order_position": match.get("batting_order_position"),
            "identity_status": "VALIDATED",
            "lineup_status": "CONFIRMED_STARTER" if target in BATTER_STARTER_REQUIRED_TARGETS else "VALIDATED_ROSTER",
            "execution_status": "LIVE_SELECTION_AVAILABLE",
        }
    )
    return enriched, None


def _gate_plays(
    plays: list[dict[str, Any]],
    *,
    fetch_json: Callable[[str], dict[str, Any]],
    game_contexts: dict[str, dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for play in plays:
        game_id = str(play.get("game_id") or "").strip()
        player_name = _play_name(play)
        if not game_id:
            rejected.append({"player": player_name, "game_id": game_id, "target": play.get("target"), "reason": "GAME_ID_MISSING"})
            continue
        if game_id not in game_contexts:
            try:
                game_contexts[game_id] = build_game_context(fetch_json(MLB_GAME_FEED.format(game_id=game_id)))
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError):
                game_contexts[game_id] = None
        enriched, reason = _validate_play(play, game_contexts[game_id])
        if enriched is None:
            rejected.append({"player": player_name, "game_id": game_id, "target": play.get("target"), "reason": reason})
        else:
            valid.append(enriched)

    for rank, play in enumerate(valid, start=1):
        play["rank"] = rank
    return valid, rejected


def apply_live_identity_gate(
    payload: dict[str, Any],
    *,
    fetch_json: Callable[[str], dict[str, Any]] = _fetch_json,
) -> dict[str, Any]:
    updated = dict(payload)
    game_contexts: dict[str, dict[str, Any] | None] = {}

    primary_plays = list(payload.get("plays") or []) if isinstance(payload.get("plays"), list) else []
    if primary_plays:
        valid, rejected = _gate_plays(primary_plays, fetch_json=fetch_json, game_contexts=game_contexts)
        updated["model_eligible_primary_count"] = len(primary_plays)
        updated["plays"] = valid
        updated["live_identity_rejections"] = rejected
        updated["live_identity_rejection_count"] = len(rejected)
        updated["live_identity_gate"] = "REQUIRE_EXACT_GAME_PLAYER_TEAM_MATCH_CONFIRMED_STARTER_AND_LIVE_SELECTION"

    shadow = payload.get("v4_singles_shadow")
    if isinstance(shadow, dict):
        plays = list(shadow.get("plays") or [])
        if plays:
            valid, rejected = _gate_plays(plays, fetch_json=fetch_json, game_contexts=game_contexts)
            updated_shadow = dict(shadow)
            updated_shadow["model_eligible_count"] = int(shadow.get("eligible_count") or len(plays))
            updated_shadow["eligible_count"] = len(valid)
            updated_shadow["plays"] = valid
            updated_shadow["live_identity_gate"] = "REQUIRE_EXACT_GAME_PLAYER_TEAM_MATCH_CONFIRMED_STARTER_AND_LIVE_SELECTION"
            updated_shadow["identity_rejections"] = rejected
            updated_shadow["identity_rejection_count"] = len(rejected)
            updated["v4_singles_shadow"] = updated_shadow

    return updated


def main() -> int:
    payload = json.loads(DEFAULT_BOARD.read_text(encoding="utf-8"))
    updated = apply_live_identity_gate(payload)
    encoded = json.dumps(updated, indent=2, sort_keys=True) + "\n"
    for target in (DEFAULT_BOARD, *MIRROR_BOARDS):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    shadow = updated.get("v4_singles_shadow") or {}
    print(json.dumps({
        "status": "ok",
        "primary_live_candidates": len(updated.get("plays") or []),
        "primary_identity_rejections": int(updated.get("live_identity_rejection_count") or 0),
        "v4_live_candidates": len(shadow.get("plays") or []),
        "v4_identity_rejections": int(shadow.get("identity_rejection_count") or 0),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

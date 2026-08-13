#!/usr/bin/env python3
"""Build the market-independent NBA opening-night projection pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
NBA_ROOT = REPO_ROOT / "sports" / "nba"
DEFAULT_CARDS = NBA_ROOT / "web" / "data" / "player_simulation_cards.json"
DEFAULT_GATE = NBA_ROOT / "web" / "data" / "simulation_credibility_gate.json"
DEFAULT_OUTPUT = NBA_ROOT / "web" / "data" / "opening_night_pool.json"

SCHEDULE_SOURCE = "https://www.nba.com/news/2026-27-schedule-announced"
GAMES = (
    {
        "game_id": "2026-10-20_BOS_DET",
        "away_team": "BOS",
        "home_team": "DET",
        "tipoff_utc": "2026-10-20T19:00:00Z",
        "network": "NBC/Peacock",
    },
    {
        "game_id": "2026-10-20_PHI_NYK",
        "away_team": "PHI",
        "home_team": "NYK",
        "tipoff_utc": "2026-10-20T23:00:00Z",
        "network": "NBC/Peacock",
    },
    {
        "game_id": "2026-10-20_OKC_SAS",
        "away_team": "OKC",
        "home_team": "SAS",
        "tipoff_utc": "2026-10-21T01:30:00Z",
        "network": "NBC/Peacock",
    },
)

# A deliberately small headline pool. It avoids claiming that June simulation
# artifacts represent a finalized October depth chart. Brown and James are
# assigned to Philadelphia per the official opening-night announcement.
HEADLINE_PLAYERS = {
    "BOS": ("1628369", "1628401", "1630202"),
    "DET": ("1630595", "1631105", "1641709"),
    "PHI": ("1630178", "1627759", "2544", "203954"),
    "NYK": ("1628973", "1626157", "1628384"),
    "OKC": ("1628983", "1631096", "1631114"),
    "SAS": ("1641705", "1642264", "1628368"),
}

TARGETS = {
    "PTS": ("pts", "Points"),
    "REB": ("reb", "Rebounds"),
    "AST": ("ast", "Assists"),
    "PRA": ("pra", "Points + rebounds + assists"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--credibility-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_name(value: object) -> str:
    return str(value or "").replace("_", " ").strip()


def _game_index() -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for game in GAMES:
        away = str(game["away_team"])
        home = str(game["home_team"])
        index[away] = {**game, "opponent": home, "venue": "away"}
        index[home] = {**game, "opponent": away, "venue": "home"}
    return index


def _watchlist(
    name: str,
    target: str,
    note: str,
    projections: list[dict[str, object]],
) -> dict[str, object]:
    candidates = sorted(
        (row for row in projections if row["target"] == target),
        key=lambda row: (-float(row["projection"]), str(row["player"])),
    )
    legs: list[dict[str, object]] = []
    used_games: set[str] = set()
    for row in candidates:
        game_id = str(row["game_id"])
        if game_id in used_games:
            continue
        used_games.add(game_id)
        legs.append(
            {
                "player": row["player"],
                "team": row["team"],
                "opponent": row["opponent"],
                "game_id": game_id,
                "target": target,
                "target_label": row["target_label"],
                "projection": row["projection"],
                "market_line": None,
                "direction": None,
                "status": "awaiting_two_sided_line",
            }
        )
        if len(legs) == len(GAMES):
            break
    return {
        "name": name,
        "leg_count": len(legs),
        "status": "awaiting_lines",
        "candidate_authorized": False,
        "note": note,
        "legs": legs,
    }


def build_payload(
    cards: list[dict[str, Any]],
    credibility_gate: dict[str, Any],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    cards_by_id = {str(card.get("player_id")): card for card in cards}
    games_by_team = _game_index()
    missing_ids = [
        player_id
        for player_ids in HEADLINE_PLAYERS.values()
        for player_id in player_ids
        if player_id not in cards_by_id
    ]
    if missing_ids:
        raise ValueError(f"Opening-night players missing simulation cards: {', '.join(missing_ids)}")

    projections: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    cutoff_dates: set[str] = set()
    for team, player_ids in HEADLINE_PLAYERS.items():
        game = games_by_team[team]
        for player_id in player_ids:
            card = cards_by_id[player_id]
            player = _display_name(card.get("player"))
            cutoff = str(card.get("data_cutoff_date") or "")
            if cutoff:
                cutoff_dates.add(cutoff)
            players.append(
                {
                    "player_id": player_id,
                    "player": player,
                    "team": team,
                    "opponent": game["opponent"],
                    "game_id": game["game_id"],
                    "confidence_tier": card.get("confidence_tier"),
                    "projected_minutes_per_game": card.get("projected_minutes_per_game"),
                }
            )
            for target, (field, label) in TARGETS.items():
                distribution = card.get(field)
                if not isinstance(distribution, dict):
                    raise ValueError(f"{player} is missing the {target} simulation distribution.")
                projections.append(
                    {
                        "player_id": player_id,
                        "player": player,
                        "team": team,
                        "opponent": game["opponent"],
                        "venue": game["venue"],
                        "game_id": game["game_id"],
                        "tipoff_utc": game["tipoff_utc"],
                        "target": target,
                        "target_label": label,
                        "projection": round(float(distribution["mean"]), 1),
                        "median": round(float(distribution["p50"]), 1),
                        "p10": round(float(distribution["p10"]), 1),
                        "p90": round(float(distribution["p90"]), 1),
                        "confidence_tier": distribution.get("confidence", card.get("confidence_tier")),
                        "market_line": None,
                        "direction": None,
                        "market_status": "awaiting_two_sided_lines",
                        "candidate_authorized": False,
                    }
                )

    for target in TARGETS:
        ranked = sorted(
            (row for row in projections if row["target"] == target),
            key=lambda row: (-float(row["projection"]), str(row["player"])),
        )
        for rank, row in enumerate(ranked, start=1):
            row["projection_rank"] = rank
    projections.sort(
        key=lambda row: (str(row["target"]), int(row["projection_rank"]), str(row["player"]))
    )

    source_bytes = json.dumps(cards, sort_keys=True, separators=(",", ":")).encode("utf-8")
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    gate_labels = credibility_gate.get("labels", {}) if isinstance(credibility_gate, dict) else {}
    blocked_reasons = credibility_gate.get("blocked_reasons", []) if isinstance(credibility_gate, dict) else []
    if not blocked_reasons and not bool(credibility_gate.get("publish_as_calibrated")):
        blocked_reasons = ["preseason_backtest_not_passed"]

    payload: dict[str, object] = {
        "schema_version": 1,
        "league": "NBA",
        "season": "2026-27",
        "opening_date": "2026-10-20",
        "generated_at_utc": generated_at,
        "status": "projection_pool_ready",
        "publication_status": "research_only",
        "market_status": "awaiting_lines",
        "market_observations": 0,
        "games": list(GAMES),
        "game_count": len(GAMES),
        "player_count": len(players),
        "projection_count": len(projections),
        "target_counts": {
            target: sum(row["target"] == target for row in projections) for target in TARGETS
        },
        "scope": (
            "Curated opening-night headline-player PTS, REB, AST, and PRA projections; "
            "research watchlist only, not sportsbook picks."
        ),
        "schedule_source": {
            "provider": "NBA.com",
            "url": SCHEDULE_SOURCE,
            "status": "official_opening_night_confirmed",
        },
        "data_quality": {
            "simulation_cutoff_dates": sorted(cutoff_dates),
            "roster_scope": "curated_offseason_headliners",
            "roster_warning": (
                "This is not a final opening-day depth chart. Availability, roles, and rosters must be "
                "refreshed after training camp before any market decision."
            ),
            "source_cards_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "validation": {
            "status": "research_only",
            "frontend_label": gate_labels.get("frontend_label", "research projection / uncalibrated"),
            "publish_as_calibrated": False,
            "blocked_reasons": list(blocked_reasons),
        },
        "players": players,
        "pool": projections,
    }
    payload["watchlists"] = [
        _watchlist(
            "Opening Scorers",
            "PTS",
            "Highest points projection from each opening-night game.",
            projections,
        ),
        _watchlist(
            "Primary Creators",
            "AST",
            "Highest assists projection from each opening-night game.",
            projections,
        ),
        _watchlist(
            "All-Around Volume",
            "PRA",
            "Highest PRA projection from each opening-night game.",
            projections,
        ),
    ]
    payload["watchlist_policy"] = {
        "status": "withheld",
        "candidate_authorized": False,
        "reason": (
            "No authentic two-sided opener prop lines are attached, and the preseason simulation "
            "credibility gate remains research-only."
        ),
    }
    return payload


def main() -> int:
    args = parse_args()
    cards = _load_json(args.cards)
    gate = _load_json(args.credibility_gate)
    if not isinstance(cards, list):
        raise ValueError("Simulation-card input must be a JSON list.")
    if not isinstance(gate, dict):
        raise ValueError("Credibility-gate input must be a JSON object.")
    payload = build_payload(cards, gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "games": payload["game_count"],
                "players": payload["player_count"],
                "projections": payload["projection_count"],
                "publication_status": payload["publication_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_player_card(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_player(card: dict[str, Any]) -> dict[str, Any]:
    player = card.get("player", {})
    offense = card.get("offense", {})
    defense = card.get("defense", {})
    impact = card.get("impact", {})
    points = float(offense.get("points_per_game") or 0.0)
    usage = float(offense.get("usage_rate") or 0.0)
    stocks = float(defense.get("steals_per_game") or 0.0) + float(defense.get("blocks_per_game") or 0.0)
    plus_minus = float(impact.get("plus_minus") or 0.0)
    return {
        "player": player,
        "scouting_report": {
            "summary": f"{player.get('name', 'Player')} profiles as a {'primary' if points >= 20 else 'supporting'} scorer.",
            "offensive_role": "high-usage" if usage >= 0.27 else "balanced",
        },
        "breakout_potential": {"score": min(1.0, max(0.0, (points + plus_minus) / 35.0))},
        "defense_portability": {"score": min(1.0, stocks / 3.0)},
        "impact_sanity": {"plus_minus": plus_minus, "box_score_signal": impact.get("box_score_signal")},
    }

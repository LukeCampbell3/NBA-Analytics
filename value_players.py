from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PlayerValuator:
    def load_player_card(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def valuate_player(self, card: dict[str, Any]) -> dict[str, Any]:
        offense = card.get("offense", {})
        defense = card.get("defense", {})
        impact = card.get("impact", {})
        age = float(card.get("identity", {}).get("age") or 0.0)
        impact_score = (
            float(offense.get("points_per_game") or 0.0)
            + 0.6 * float(offense.get("assists_per_game") or 0.0)
            + 0.5 * float(defense.get("rebounds_per_game") or 0.0)
            + float(defense.get("steals_per_game") or 0.0)
            + float(defense.get("blocks_per_game") or 0.0)
            + float(impact.get("plus_minus") or 0.0)
        )
        age_multiplier = 1.08 if 23 <= age <= 28 else 0.95 if age >= 33 else 1.0
        market_value = round(max(0.0, impact_score * age_multiplier * 1.25), 2)
        return {"card": card, "impact_score": round(impact_score, 3), "market_value_m": market_value}

    def generate_report(self, valuation: dict[str, Any]) -> dict[str, Any]:
        card = valuation["card"]
        player = card.get("player", {})
        value = float(valuation.get("market_value_m") or 0.0)
        return {
            "player": player,
            "impact": {"score": valuation.get("impact_score"), "role": "starter" if value >= 25 else "rotation"},
            "market_value": {"annual_value_m": value},
            "contract": {"suggested_years": 4 if value >= 25 else 2},
            "trade_value": {"tier": "positive" if value >= 15 else "neutral"},
            "aging": {"age": card.get("identity", {}).get("age"), "risk": "normal"},
        }

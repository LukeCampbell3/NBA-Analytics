#!/usr/bin/env python3
"""Join a Week projection pool to captured RotoWire lines and build shadow pools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
NFL_ROOT = REPO_ROOT / "sports/nfl"
TARGET_BY_POSITION = {"QB": "passing", "RB": "rushing", "WR": "receiving", "TE": "receiving"}
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX"}


def _name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _team(value: Any) -> str:
    raw = str(value or "").strip().upper()
    return TEAM_ALIASES.get(raw, raw)


def _implied(price: float) -> float:
    return 100 / (price + 100) if price > 0 else abs(price) / (abs(price) + 100)


def _decimal(price: float) -> float:
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def _normal_over(mean: float, p10: float, p90: float, line: float) -> float:
    sd = max(1.0, (p90 - p10) / (2 * 1.2815515655446004))
    z = (line - mean) / sd
    return 0.5 * math.erfc(z / math.sqrt(2))


def build_board(pool_payload: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    projection_by_key = {
        (_name(row["player"]), _team(row.get("team")), TARGET_BY_POSITION.get(str(row.get("position")))): row
        for row in pool_payload.get("pool", [])
    }
    candidates = []
    for offer in snapshot.get("observations", []):
        projection = projection_by_key.get(
            (_name(offer.get("player")), _team(offer.get("provider_team")), offer.get("target"))
        )
        if projection is None:
            continue
        over_probability = _normal_over(
            float(projection["projection"]), float(projection["p10"]),
            float(projection["p90"]), float(offer["line"]),
        )
        over_implied = _implied(float(offer["over_price"]))
        under_implied = _implied(float(offer["under_price"]))
        no_vig_over = over_implied / (over_implied + under_implied)
        side = "OVER" if over_probability >= 0.5 else "UNDER"
        probability = over_probability if side == "OVER" else 1 - over_probability
        market_probability = no_vig_over if side == "OVER" else 1 - no_vig_over
        price = float(offer["over_price"] if side == "OVER" else offer["under_price"])
        disagreement = abs(probability - market_probability)
        support_status = "OUT_OF_SUPPORT" if disagreement > 0.15 else "IN_SUPPORT"
        candidates.append({
            "player": projection["player"], "player_id": projection["player_id"],
            "position": projection["position"], "team": projection["team"],
            "opponent": projection["opponent"], "game_id": projection["game_id"],
            "kickoff_utc": projection["kickoff_utc"], "market": offer["market"],
            "side": side, "line": float(offer["line"]), "bookmaker": offer["bookmaker"],
            "price": price, "projection": float(projection["projection"]),
            "raw_model_probability": round(probability, 6),
            "no_vig_market_probability": round(market_probability, 6),
            "raw_probability_edge": round(probability - market_probability, 6),
            "raw_model_ev": round(probability * _decimal(price) - 1, 6),
            "absolute_model_market_disagreement": round(disagreement, 6),
            "support_status": support_status,
            "snapshot_time_utc": offer["snapshot_time_utc"],
            "source": offer["source"],
            "selection_status": "RESEARCH_ONLY_UNCALIBRATED",
            "candidate_authorized": False,
        })
    candidates.sort(key=lambda row: (-row["raw_model_probability"], -row["raw_model_ev"], row["player"]))

    def best_unique(position_set: set[str], *, same_game: bool = False, limit: int = 2) -> list[dict[str, Any]]:
        eligible = [
            row for row in candidates
            if row["position"] in position_set
            and row["raw_model_ev"] > 0
            and row["support_status"] == "IN_SUPPORT"
        ]
        if same_game:
            game_counts: dict[str, int] = {}
            for row in eligible:
                game_counts[row["game_id"]] = game_counts.get(row["game_id"], 0) + 1
            eligible = [row for row in eligible if game_counts[row["game_id"]] >= limit]
            if not eligible:
                return []
            game = max(game_counts, key=lambda key: max((r["raw_model_probability"] for r in eligible if r["game_id"] == key), default=0))
            eligible = [row for row in eligible if row["game_id"] == game]
        selected, players, games = [], set(), set()
        for row in eligible:
            if row["player_id"] in players or (not same_game and row["game_id"] in games):
                continue
            selected.append(row); players.add(row["player_id"]); games.add(row["game_id"])
            if len(selected) == limit:
                break
        return selected

    categories = {
        "passer_parlay": best_unique({"QB"}),
        "rusher_parlay": best_unique({"RB"}),
        "receiver_parlay": best_unique({"WR", "TE"}),
        "same_game_parlay": best_unique({"QB", "RB", "WR", "TE"}, same_game=True),
    }
    first_td = snapshot.get("audit", {}).get("first_td_best_prices", [])
    return {
        "schema_version": 1,
        "league": "NFL", "season": pool_payload.get("season"), "week": pool_payload.get("week"),
        "generated_at_utc": snapshot.get("audit", {}).get("fetched_at_utc"),
        "source": snapshot.get("audit", {}).get("source_url"),
        "status": "SHADOW_RESEARCH_ONLY", "candidate_authorized": False,
        "methodology": {
            "probability": "Normal approximation from the frozen projection pool P10/P90 interval.",
            "warning": "These probabilities are not market-calibrated or certification-authorized.",
            "support_rule": "Absolute model-market disagreement above 15pp is retained but excluded as OUT_OF_SUPPORT.",
            "parlay_probability": "WITHHELD_NO_DEPENDENCY_MODEL_OR_EXECUTABLE_COMBINED_QUOTE",
        },
        "candidate_count": len(candidates), "candidates": candidates,
        "pools": categories,
        "first_touchdown_odds_only": first_td,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=NFL_ROOT / "web/data/week_1_pool.json")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=NFL_ROOT / "web/data/week_1_market_board.json")
    args = parser.parse_args()
    payload = build_board(json.loads(args.pool.read_text()), json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": payload["candidate_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

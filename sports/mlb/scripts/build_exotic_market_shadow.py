#!/usr/bin/env python3
"""Build a fail-closed, non-authoritative MLB exotic-market research board.

Only one actionable research side is surfaced for a canonical game/market/line.
Opposite sides and non-positive-EV rows remain auditable in diagnostics instead
of appearing as duplicate game picks with betslip calls to action.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "sports/mlb/web/data/same_game_predictions.json"
DEFAULT_OUTPUT = REPO_ROOT / "sports/mlb/web/data/exotic_market_predictions.json"
SUPPORTED = {"game_total", "first_5_innings_total"}
MARKET_REGISTRY = [
    ("game_total_runs", "game_total", "SCORABLE_SHADOW", "Existing joint game simulation and live total price."),
    ("first_5_total_runs", "first_5_innings_total", "SCORABLE_SHADOW", "Existing first-five simulation and live total price."),
    ("team_total_runs", None, "DISCOVERY", "Requires live team-total parsing and a team-specific run distribution."),
    ("team_hits", None, "MODEL_REQUIRED", "Requires a calibrated team-hit count model and box-score settlement."),
    ("pitcher_strikeouts_inning", None, "EVENT_MODEL_REQUIRED", "Requires pitch/play sequence data and inning settlement."),
    ("pitcher_pitches_inning", None, "EVENT_MODEL_REQUIRED", "Requires pitch counts by inning and pitcher-removal rules."),
    ("plate_appearance_pitch_count", None, "EVENT_MODEL_REQUIRED", "Requires an exact plate-appearance identifier and pitch-event model."),
]


def american_to_decimal(price: int | float) -> float:
    price = float(price)
    return 1.0 + (100.0 / abs(price) if price < 0 else price / 100.0)


def _canonical_market_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """One sportsbook proposition regardless of side/price.

    A total's OVER and UNDER are two sides of the same proposition. Showing
    both as separate research picks is confusing and can create duplicate
    NYY/LAA-style entries on the board. Keep one winning side per proposition
    after scoring, while preserving every rejected side in diagnostics.
    """
    return (row.get("game_id"), row.get("market"), row.get("line"))


def build_payload(source: dict[str, Any]) -> dict[str, Any]:
    raw_candidates: list[dict[str, Any]] = []
    seen_exact: set[tuple[Any, ...]] = set()
    for game in source.get("games", []):
        combos = list(game.get("combo_candidates") or []) + list(game.get("exploratory_ev_candidates") or [])
        for combo in combos:
            for leg in (combo.get("leg_a"), combo.get("leg_b")):
                if not isinstance(leg, dict) or leg.get("market") not in SUPPORTED:
                    continue
                if not leg.get("price_confirmed") or leg.get("price_american") is None:
                    continue
                exact_key = (
                    game.get("game_id"), leg.get("market"), leg.get("side"),
                    leg.get("line"), leg.get("price_american"),
                )
                if exact_key in seen_exact:
                    continue
                seen_exact.add(exact_key)
                probability = float(leg["model_probability"])
                raw_candidates.append({
                    "game_id": game.get("game_id"), "event_date": game.get("event_date") or source.get("run_date"),
                    "away_team": game.get("away_team"), "home_team": game.get("home_team"),
                    "market": leg.get("market"), "side": leg.get("side"), "line": leg.get("line"),
                    "model_probability": probability, "price_american": leg.get("price_american"),
                    "expected_value_per_unit": probability * american_to_decimal(leg["price_american"]) - 1.0,
                    "sportsbook": leg.get("sportsbook"), "sportsbook_deeplink": leg.get("sportsbook_deeplink"),
                    "deeplinks_by_region": leg.get("deeplinks_by_region"), "authorization_status": "SHADOW_ONLY",
                    "publication_authority": False,
                    "support_blocking_dimensions": leg.get("support_blocking_dimensions") or [],
                })

    by_market: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_candidates:
        by_market.setdefault(_canonical_market_key(row), []).append(row)

    candidates: list[dict[str, Any]] = []
    diagnostic_rejections: list[dict[str, Any]] = []
    for rows in by_market.values():
        rows.sort(
            key=lambda row: (row["expected_value_per_unit"], row["model_probability"]),
            reverse=True,
        )
        best = rows[0]
        if best["expected_value_per_unit"] > 0.0:
            candidates.append(best)
            rejected = rows[1:]
            for row in rejected:
                diagnostic_rejections.append({
                    **row,
                    "rejection_reason": "DOMINATED_OPPOSITE_SIDE_OR_DUPLICATE_MARKET",
                })
        else:
            rejected = rows
            for row in rejected:
                diagnostic_rejections.append({
                    **row,
                    "rejection_reason": "NON_POSITIVE_MODEL_EV",
                })

    candidates.sort(key=lambda row: (row["expected_value_per_unit"], row["model_probability"]), reverse=True)
    diagnostic_rejections.sort(
        key=lambda row: (str(row.get("game_id") or ""), str(row.get("market") or ""), float(row.get("line") or 0.0), str(row.get("side") or ""))
    )
    return {
        "status": "ok" if source.get("status") == "ok" else "source_unavailable",
        "policy": "EXOTIC_MARKETS_V1_SHADOW", "authorization_status": "SHADOW_ONLY",
        "publication_authority": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": source.get("run_date"), "candidates": candidates, "candidate_count": len(candidates),
        "diagnostic_rejections": diagnostic_rejections,
        "diagnostic_rejection_count": len(diagnostic_rejections),
        "dedupe_policy": "one_positive_ev_side_per_game_market_line",
        "market_registry": [
            {"market": market, "source_market": source_market, "readiness": readiness, "reason": reason}
            for market, source_market, readiness, reason in MARKET_REGISTRY
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_payload(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": payload["candidate_count"],
        "diagnostic_rejection_count": payload["diagnostic_rejection_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

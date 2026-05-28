from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFE_STATE_REQUIRED = [
    "player",
    "market_type",
    "side",
    "line",
    "safe_state_tier",
    "edge_defendability_tier",
    "recommended_action",
    "settlement_status",
    "explanation",
    "warning_badges",
]
SIMULATION_REQUIRED = [
    "data_cutoff_date",
    "confidence_tier",
    "volatility_score",
    "missing_data_warnings",
]
SIMULATION_STATS = ["pts", "reb", "ast", "pra"]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return [] if path.name.endswith("cards.json") else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [] if path.name.endswith("cards.json") else {}


def _as_cards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [card for card in payload if isinstance(card, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
        return [card for card in payload["cards"] if isinstance(card, dict)]
    return []


def _missing(card: dict[str, Any], fields: list[str]) -> list[str]:
    missing: list[str] = []
    for field in fields:
        value = card.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def validate_site_cards(*, site_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or site_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_cards = _as_cards(_read_json(site_dir / "safe_state_cards.json"))
    simulation_cards = _as_cards(_read_json(site_dir / "player_simulation_cards.json"))

    safe_issues: list[dict[str, Any]] = []
    for card in safe_cards:
        missing = _missing(card, SAFE_STATE_REQUIRED)
        if card.get("shadow_only") is not True:
            missing.append("shadow_only")
        if card.get("staking_enabled") is True or card.get("auto_bet_enabled") is True:
            missing.append("no_staking_or_autobet")
        if missing:
            safe_issues.append({"candidate_id": card.get("candidate_id", ""), "missing_or_invalid": missing})

    sim_issues: list[dict[str, Any]] = []
    for card in simulation_cards:
        missing = _missing(card, SIMULATION_REQUIRED)
        for stat in SIMULATION_STATS:
            stat_payload = card.get(stat)
            if not isinstance(stat_payload, dict):
                missing.append(f"{stat}.range")
                continue
            for field in ["p10", "p50", "p90"]:
                if stat_payload.get(field) is None:
                    missing.append(f"{stat}.{field}")
        if "projected_minutes_per_game" in card and not any(
            isinstance(card.get(stat), dict) and card[stat].get("p10") is not None for stat in SIMULATION_STATS
        ):
            missing.append("single_point_only_projection")
        if missing:
            sim_issues.append({"player": card.get("player", ""), "missing_or_invalid": missing})

    report = {
        "safe_state_card_count": int(len(safe_cards)),
        "simulation_card_count": int(len(simulation_cards)),
        "safe_state_cards_valid": not safe_issues,
        "simulation_cards_valid": not sim_issues,
        "all_cards_shadow_or_research_only": all(card.get("shadow_only") is True for card in safe_cards),
        "staking_or_autobet_visible": any(
            card.get("staking_enabled") is True or card.get("auto_bet_enabled") is True for card in safe_cards + simulation_cards
        ),
        "validation_passed": not safe_issues and not sim_issues,
        "issues": {
            "safe_state": safe_issues[:100],
            "simulation": sim_issues[:100],
        },
    }
    (output_dir / "site_card_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "site_card_validation_report.md").write_text(_format_markdown(report), encoding="utf-8")
    return report


def _format_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Site Card Validation Report",
            "",
            f"- safe_state_card_count: {report.get('safe_state_card_count')}",
            f"- simulation_card_count: {report.get('simulation_card_count')}",
            f"- safe_state_cards_valid: {report.get('safe_state_cards_valid')}",
            f"- simulation_cards_valid: {report.get('simulation_cards_valid')}",
            f"- staking_or_autobet_visible: {report.get('staking_or_autobet_visible')}",
            f"- validation_passed: {report.get('validation_passed')}",
            "",
            "Safe-state labels remain shadow-only. Simulation cards remain research projections until credibility backtests pass.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate safe-state and player simulation site cards.")
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_site_cards(site_dir=args.site_dir, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Load local site export artifacts — no cloud inference."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any


def get_artifact_data_dir() -> Path:
    configured = os.environ.get("ARTIFACT_DATA_DIR", "").strip()
    if configured:
        path = Path(configured)
        if path.is_dir():
            return path
    repo_root = Path(__file__).resolve().parents[4]
    default = repo_root / "sports" / "nba" / "web" / "data"
    return default


def read_json(name: str, fallback: Any) -> Any:
    path = get_artifact_data_dir() / name
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def load_safe_state_latest() -> dict[str, Any]:
    payload = read_json("safe_state_latest.json", {})
    if isinstance(payload, dict):
        return payload
    return {}


def load_safe_state_cards() -> list[dict[str, Any]]:
    latest = load_safe_state_latest()
    cards = latest.get("cards") if isinstance(latest, dict) else None
    if isinstance(cards, list):
        return [card for card in cards if isinstance(card, dict)]
    flat = read_json("safe_state_cards.json", [])
    return [card for card in flat if isinstance(card, dict)] if isinstance(flat, list) else []


def load_simulation_cards() -> list[dict[str, Any]]:
    payload = read_json("player_simulation_cards.json", [])
    return [card for card in payload if isinstance(card, dict)] if isinstance(payload, list) else []


def load_site_manifest() -> dict[str, Any]:
    payload = read_json("site_manifest.json", {})
    return payload if isinstance(payload, dict) else {}


def load_model_status() -> dict[str, Any]:
    manifest = load_site_manifest()
    production = read_json("site_production_status.json", {})
    credibility = read_json("simulation_credibility_gate.json", {})
    latest = load_safe_state_latest()
    return {
        "run_id": latest.get("run_id") or manifest.get("run_id"),
        "run_date": latest.get("run_date") or manifest.get("run_date"),
        "data_cutoff_date": latest.get("data_cutoff_date") or manifest.get("data_cutoff_date"),
        "shadow_only": True,
        "promotion_ready": bool(latest.get("promotion_ready") or manifest.get("promotion_ready") or False),
        "production_behavior_changed": False,
        "staking_enabled": False,
        "auto_bet_enabled": False,
        "simulation_credibility_status": credibility.get("status"),
        "production_status": production.get("status") if isinstance(production, dict) else None,
        "disclaimer": "Analytics and research only. Shadow labels are not production promotion.",
    }


def load_par_model() -> dict[str, Any]:
    payload = read_json("par_model.json", {})
    return payload if isinstance(payload, dict) else {}


def load_par_players() -> list[dict[str, Any]]:
    payload = read_json("player_par_components.json", [])
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def load_par_atoms() -> list[dict[str, Any]]:
    summary = read_json("player_par_atom_summary.json", [])
    if isinstance(summary, list) and summary:
        return [row for row in summary if isinstance(row, dict)]
    path = get_artifact_data_dir() / "player_par_atoms.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_par_forecasts() -> list[dict[str, Any]]:
    payload = read_json("player_par_forecasts.json", [])
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def load_par_leaderboard() -> list[dict[str, Any]]:
    payload = read_json("par_leaderboard.json", [])
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def load_par_validation() -> dict[str, Any]:
    payload = read_json("par_validation.json", {})
    return payload if isinstance(payload, dict) else {}


def load_par_manifest() -> dict[str, Any]:
    payload = read_json("par_build_manifest.json", {})
    return payload if isinstance(payload, dict) else {}


def response_meta() -> dict[str, Any]:
    latest = load_safe_state_latest()
    manifest = load_site_manifest()
    return {
        "shadow_only": True,
        "promotion_ready": bool(latest.get("promotion_ready") or manifest.get("promotion_ready") or False),
        "production_behavior_changed": False,
        "data_cutoff_date": latest.get("data_cutoff_date") or manifest.get("data_cutoff_date"),
        "run_id": latest.get("run_id") or manifest.get("run_id"),
        "run_date": latest.get("run_date") or manifest.get("run_date"),
    }


def settlement_history_rows() -> list[dict[str, Any]]:
    cards = load_safe_state_cards()
    rows = []
    for card in cards:
        rows.append(
            {
                "player": card.get("player"),
                "market_type": card.get("market_type"),
                "side": card.get("side"),
                "line": card.get("line"),
                "settlement_status": card.get("settlement_status"),
                "recommended_action": card.get("recommended_action"),
                "safe_state_tier": card.get("safe_state_tier"),
                "shadow_only": True,
            }
        )
    return rows


def safe_state_csv(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "player,market_type,side,line,settlement_status,recommended_action\n"
    fieldnames = [
        "player",
        "market_type",
        "side",
        "line",
        "settlement_status",
        "recommended_action",
        "safe_state_tier",
        "edge_defendability_tier",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for card in cards:
        writer.writerow({key: card.get(key) for key in fieldnames})
    return buffer.getvalue()


def filter_cards_for_entitlements(
    cards: list[dict[str, Any]],
    *,
    max_cards_per_day: int | None,
    can_view_candidate_pool: bool,
) -> list[dict[str, Any]]:
    visible = cards
    if not can_view_candidate_pool:
        visible = [
            card
            for card in cards
            if str(card.get("recommended_action") or "").upper() != "CANDIDATE_POOL_ONLY"
        ]
    if max_cards_per_day is not None:
        visible = visible[: max(0, int(max_cards_per_day))]
    return visible


def filter_simulations_for_entitlements(
    cards: list[dict[str, Any]],
    *,
    max_cards_per_day: int | None,
    can_view_simulation_filters: bool,
    preview_limit: int = 3,
) -> list[dict[str, Any]]:
    if can_view_simulation_filters:
        limit = max_cards_per_day if max_cards_per_day is not None else len(cards)
        return cards[:limit]
    return cards[:preview_limit]

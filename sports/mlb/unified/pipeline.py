from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import adapt_legacy_play, adapt_pitcher_leg, adapt_team_leg
from .decision import DecisionPolicy, select
from .market_registry import capability_payload
from .policy_manifest import build_policy_manifest
from .parlay import construct_all_ticket_classes
from .production_state import atomic_write_json
from .schemas import BetCandidate


@dataclass
class UnifiedRun:
    candidates: list[BetCandidate]
    singles: list[BetCandidate]
    rejected: list[BetCandidate]
    tickets: dict
    source_status: dict[str, str]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_candidates(data_dir: Path) -> tuple[list[BetCandidate], dict[str, str]]:
    candidates: list[BetCandidate] = []
    status: dict[str, str] = {}
    daily = _load(data_dir / "daily_predictions.json")
    status["daily_predictions"] = "LOADED" if daily else "UNAVAILABLE"
    for play in daily.get("plays", []):
        candidates.append(adapt_legacy_play(play))
    for play in (daily.get("v4_singles_shadow") or {}).get("plays", []):
        candidates.append(adapt_legacy_play(play))

    same_game = _load(data_dir / "same_game_predictions.json")
    status["same_game_predictions"] = "LOADED" if same_game else "UNAVAILABLE"
    for game in same_game.get("games", []):
        for combo in game.get("combo_candidates", []):
            for leg in combo.get("legs", []):
                candidates.append(adapt_team_leg(leg, game))

    pitcher = _load(data_dir / "pitcher_parlay_predictions.json")
    status["pitcher_parlay_predictions"] = "LOADED" if pitcher else "UNAVAILABLE"
    for leg in pitcher.get("legs", []):
        candidates.append(adapt_pitcher_leg(leg))
    return candidates, status


def _deduplicate(candidates: list[BetCandidate]) -> list[BetCandidate]:
    unique: dict[tuple, BetCandidate] = {}
    for candidate in candidates:
        key = (candidate.game_id, candidate.subject_id, candidate.market_type, candidate.period, candidate.side, candidate.line, candidate.sportsbook)
        prior = unique.get(key)
        score = candidate.calibrated_probability if candidate.calibrated_probability is not None else -1
        prior_score = prior.calibrated_probability if prior and prior.calibrated_probability is not None else -1
        if prior is None or score > prior_score:
            unique[key] = candidate
    return list(unique.values())


def run(data_dir: Path, *, policy: DecisionPolicy | None = None) -> UnifiedRun:
    candidates, source_status = collect_candidates(data_dir)
    candidates = _deduplicate(candidates)
    singles, rejected = select(candidates, policy or DecisionPolicy())
    # Only universal single survivors feed tickets. Compatibility adapters can
    # never bypass the gate, and ticket legs never flow back into singles.
    tickets = construct_all_ticket_classes(singles)
    return UnifiedRun(candidates, singles, rejected, tickets, source_status)


def export_payload(result: UnifiedRun, *, run_date: str | None = None, repo_root: Path | None = None,
                   engine_state: str = "DEVELOPMENT") -> dict[str, Any]:
    rejected = [{"candidate": candidate.to_dict(), "reasons": candidate.rejection_reasons} for candidate in result.rejected]
    parlays = {}
    pruning = {}
    for leg_count, (tickets, counts) in result.tickets.items():
        key = {2: "two_leg", 3: "three_leg", 4: "four_leg"}[leg_count]
        parlays[key] = [ticket.to_dict() for ticket in tickets[:1]]
        pruning[key] = counts
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    policy_hash = build_policy_manifest(repo_root)["policy_hash"] if repo_root else None
    generation_seed = f"{run_date}|{generated_at}|{policy_hash}|unified_mlb_v1"
    return {
        "schema_version": "unified_mlb_v1",
        "generated_at_utc": generated_at,
        "generation_id": hashlib.sha256(generation_seed.encode()).hexdigest()[:24],
        "run_date": run_date,
        "policy_hash": policy_hash,
        "model_version": "compatibility_models_at_source_artifact",
        "engine_state": engine_state,
        "evidence": {"state": "DEVELOPMENT", "publication_authority": False},
        "singles": [candidate.to_dict() for candidate in result.singles],
        "parlays": parlays,
        "same_game_parlays": [],
        "exotic": [],
        "diagnostics": {"rejected": rejected, "pruning": pruning, "sources": result.source_status},
        "capabilities": capability_payload(),
    }


def write_payload(payload: dict[str, Any], output: Path) -> None:
    validate_payload(payload)
    atomic_write_json(output, payload)


def validate_payload(payload: dict[str, Any]) -> None:
    required = {"schema_version", "generated_at_utc", "generation_id", "run_date", "policy_hash", "engine_state", "singles", "parlays", "same_game_parlays", "exotic", "diagnostics", "capabilities", "evidence"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"unified artifact missing fields: {sorted(missing)}")
    if payload["schema_version"] != "unified_mlb_v1":
        raise ValueError("unified artifact schema mismatch")
    if not payload["run_date"] or not payload["policy_hash"]:
        raise ValueError("run date and policy hash are required")
    for candidate in payload["singles"]:
        if candidate.get("rejection_reasons"):
            raise ValueError("rejected candidate cannot appear in singles")
        for field in ("usable_probability", "decimal_price", "market_break_even_probability", "probability_edge", "conservative_expected_value"):
            if candidate.get(field) is None:
                raise ValueError(f"selected single missing {field}")
        if not 0 <= float(candidate["usable_probability"]) <= 1:
            raise ValueError("impossible selected probability")
        if float(candidate["conservative_expected_value"]) <= 0:
            raise ValueError("non-positive EV selected single")
    expected_classes = {"two_leg": 2, "three_leg": 3, "four_leg": 4}
    if set(payload["parlays"]) != set(expected_classes):
        raise ValueError("parlay classes missing or unexpected")
    for class_name, count in expected_classes.items():
        for ticket in payload["parlays"][class_name]:
            if ticket.get("leg_count") != count or len(ticket.get("legs") or []) != count:
                raise ValueError("ticket leg-count mismatch")
            if ticket.get("rejection_reasons"):
                raise ValueError("rejected ticket cannot be selected")
            joint = float(ticket["joint_probability"])
            marginals = [float(leg["usable_probability"]) for leg in ticket["legs"]]
            if joint < 0 or joint > min(marginals):
                raise ValueError("invalid ticket joint probability")
            if any(float(leg["conservative_expected_value"]) <= 0 for leg in ticket["legs"]):
                raise ValueError("negative-EV leg in ticket")

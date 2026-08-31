from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import adapt_legacy_play, adapt_pitcher_leg, adapt_team_leg
from .decision import DecisionPolicy, select
from .market_registry import capability_payload
from .parlay import construct_all_ticket_classes
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


def export_payload(result: UnifiedRun, *, run_date: str | None = None) -> dict[str, Any]:
    rejected = [{"candidate": candidate.to_dict(), "reasons": candidate.rejection_reasons} for candidate in result.rejected]
    parlays = {}
    pruning = {}
    for leg_count, (tickets, counts) in result.tickets.items():
        key = {2: "two_leg", 3: "three_leg", 4: "four_leg"}[leg_count]
        parlays[key] = [ticket.to_dict() for ticket in tickets[:1]]
        pruning[key] = counts
    return {
        "schema_version": "unified_mlb_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_date": run_date,
        "evidence": {"state": "DEVELOPMENT", "publication_authority": False},
        "singles": [candidate.to_dict() for candidate in result.singles],
        "parlays": parlays,
        "same_game_parlays": [],
        "exotic": [],
        "diagnostics": {"rejected": rejected, "pruning": pruning, "sources": result.source_status},
        "capabilities": capability_payload(),
    }


def write_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

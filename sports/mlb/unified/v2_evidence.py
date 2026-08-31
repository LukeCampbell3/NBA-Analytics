from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence_ledger import append_generation, append_revision, read_ledger


REQUIRED_CANDIDATE_FIELDS = {
    "slate_id", "event_id", "player_id", "market_id", "line", "sportsbook",
    "quoted_odds", "quote_timestamp", "prediction_timestamp", "lineup_status",
    "player_status", "model_version", "calibrator_version", "policy_hash",
    "raw_probability", "calibrated_probability", "usable_probability",
    "market_implied_probability", "uncertainty", "uncertainty_components",
    "support_score", "ood_status", "edge", "raw_ev", "calibrated_ev",
    "conservative_ev", "admissible", "rejection_reasons", "ranking_position",
    "final_selection_decision",
}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_point_in_time_candidate(candidate: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(candidate))
    if missing:
        raise ValueError(f"point-in-time candidate missing fields: {missing}")
    if not isinstance(candidate["rejection_reasons"], list):
        raise ValueError("rejection_reasons must be a list")
    if candidate["admissible"] and candidate["rejection_reasons"]:
        raise ValueError("admissible candidate has rejection reasons")
    prediction = datetime.fromisoformat(str(candidate["prediction_timestamp"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    if candidate["quote_timestamp"] is not None:
        quote = datetime.fromisoformat(str(candidate["quote_timestamp"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        if quote > prediction:
            raise ValueError("quote timestamp occurs after decision")
    elif candidate["admissible"]:
        raise ValueError("admissible candidate lacks quote timestamp")


def capture_policy_generation(path: Path, *, generation_id: str, generated_at_utc: str,
                              run_date: str, baseline_policy_hash: str,
                              challenger_policy_hash: str,
                              baseline_candidates: list[dict[str, Any]],
                              challenger_candidates: list[dict[str, Any]],
                              disagreements: list[dict[str, Any]]) -> bool:
    for candidate in baseline_candidates + challenger_candidates:
        validate_point_in_time_candidate(candidate)
    scientific_payload = {
        "run_date": run_date, "baseline_policy_hash": baseline_policy_hash,
        "challenger_policy_hash": challenger_policy_hash,
        "baseline_candidates": baseline_candidates, "challenger_candidates": challenger_candidates,
        "disagreements": disagreements,
    }
    record = {
        "schema_version": "mlb_v2_1_evidence_v1", "generation_id": generation_id,
        "generated_at_utc": generated_at_utc, **scientific_payload,
        "prediction_payload_sha256": canonical_hash(scientific_payload),
        "settlement": None, "revision": 1,
    }
    return append_generation(path, record)


def append_hash_linked_settlement(path: Path, *, generation_id: str,
                                  candidate_id: str, official_outcome: float,
                                  settlement: str, realized_return: float,
                                  source_identity: str, source_response_sha256: str,
                                  settled_at_utc: str) -> bool:
    rows = read_ledger(path)
    generation = next((row for row in rows if row.get("generation_id") == generation_id and row.get("revision") == 1), None)
    if generation is None:
        raise ValueError("settlement references unknown prediction generation")
    if len(source_response_sha256) != 64:
        raise ValueError("invalid settlement source hash")
    known = {
        str(candidate.get("candidate_id"))
        for key in ("baseline_candidates", "challenger_candidates")
        for candidate in generation.get(key, [])
    }
    if candidate_id not in known:
        raise ValueError("settlement references unknown candidate")
    prior_revisions = [row for row in rows if row.get("generation_id") == generation_id]
    revision = max(int(row.get("revision", 1)) for row in prior_revisions) + 1
    payload = {
        "candidate_id": candidate_id, "official_outcome": official_outcome,
        "settlement": settlement, "realized_return": realized_return,
        "source_identity": source_identity, "source_response_sha256": source_response_sha256,
        "settled_at_utc": settled_at_utc,
        "prediction_payload_sha256": generation["prediction_payload_sha256"],
    }
    record = {
        "schema_version": "mlb_v2_1_settlement_v1", "generation_id": generation_id,
        "generated_at_utc": settled_at_utc, "revision": revision,
        "supersedes_revision": revision - 1, "settlement_payload": payload,
        "settlement_payload_sha256": canonical_hash(payload),
    }
    return append_revision(path, record)

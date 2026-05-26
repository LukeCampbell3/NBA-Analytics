from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "failure_mode_schema.yaml"
REGISTRY_PATH = ROOT / "failure_mode_registry.yaml"


@dataclass(frozen=True)
class FailureModeDefinition:
    failure_mode_id: str
    market_families: tuple[str, ...]
    candidate_symptoms: tuple[str, ...]
    required_pre_event_features: tuple[str, ...]
    postgame_attribution_signals: tuple[str, ...]
    likely_causal_pathway: str
    candidate_interventions: tuple[dict[str, Any], ...]
    allowed_penalties_gates: tuple[str, ...]
    opposite_side_discovery_rules: tuple[str, ...]
    validation_segments: tuple[str, ...]
    promotion_requirements: tuple[str, ...]
    known_risks_overfit_traps: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}")
    return payload


def load_failure_mode_schema(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(path or SCHEMA_PATH)


def load_failure_mode_registry_payload(path: Path | None = None) -> dict[str, Any]:
    payload = _load_yaml(path or REGISTRY_PATH)
    validate_failure_mode_registry(payload, load_failure_mode_schema())
    return payload


def validate_failure_mode_registry(
    payload: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> None:
    active_schema = schema or load_failure_mode_schema()
    required_fields = list(active_schema.get("required_fields", []))
    allowed_interventions = set(active_schema.get("allowed_intervention_types", []))
    required_promotion = set(active_schema.get("required_promotion_requirements", []))

    failure_modes = payload.get("failure_modes")
    if not isinstance(failure_modes, list) or not failure_modes:
        raise ValueError("failure_mode_registry.yaml must define a non-empty failure_modes list.")

    seen_ids: set[str] = set()
    for row in failure_modes:
        if not isinstance(row, dict):
            raise ValueError("Each failure mode entry must be a mapping.")
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Failure mode missing required fields: {missing}")
        failure_mode_id = str(row.get("failure_mode_id", "")).strip()
        if not failure_mode_id:
            raise ValueError("failure_mode_id must be non-empty.")
        if failure_mode_id in seen_ids:
            raise ValueError(f"Duplicate failure_mode_id: {failure_mode_id}")
        seen_ids.add(failure_mode_id)
        interventions = row.get("candidate_interventions", [])
        if not isinstance(interventions, list) or not interventions:
            raise ValueError(f"{failure_mode_id} must define at least one candidate intervention.")
        for intervention in interventions:
            if not isinstance(intervention, dict):
                raise ValueError(f"{failure_mode_id} intervention entries must be mappings.")
            intervention_type = str(intervention.get("intervention_type", "")).strip()
            if intervention_type not in allowed_interventions:
                raise ValueError(f"{failure_mode_id} uses unsupported intervention_type={intervention_type}")
        promotion_requirements = {str(token).strip() for token in row.get("promotion_requirements", [])}
        missing_promotion = sorted(required_promotion - promotion_requirements)
        if missing_promotion:
            raise ValueError(f"{failure_mode_id} is missing promotion requirements: {missing_promotion}")


def load_failure_mode_registry(path: Path | None = None) -> dict[str, FailureModeDefinition]:
    payload = load_failure_mode_registry_payload(path)
    out: dict[str, FailureModeDefinition] = {}
    for row in payload.get("failure_modes", []):
        definition = FailureModeDefinition(
            failure_mode_id=str(row["failure_mode_id"]).strip(),
            market_families=tuple(str(token).strip().upper() for token in row.get("market_families", [])),
            candidate_symptoms=tuple(str(token).strip() for token in row.get("candidate_symptoms", [])),
            required_pre_event_features=tuple(str(token).strip() for token in row.get("required_pre_event_features", [])),
            postgame_attribution_signals=tuple(str(token).strip() for token in row.get("postgame_attribution_signals", [])),
            likely_causal_pathway=str(row.get("likely_causal_pathway", "")).strip(),
            candidate_interventions=tuple(dict(item) for item in row.get("candidate_interventions", [])),
            allowed_penalties_gates=tuple(str(token).strip() for token in row.get("allowed_penalties_gates", [])),
            opposite_side_discovery_rules=tuple(str(token).strip() for token in row.get("opposite_side_discovery_rules", [])),
            validation_segments=tuple(str(token).strip() for token in row.get("validation_segments", [])),
            promotion_requirements=tuple(str(token).strip() for token in row.get("promotion_requirements", [])),
            known_risks_overfit_traps=tuple(str(token).strip() for token in row.get("known_risks_overfit_traps", [])),
        )
        out[definition.failure_mode_id] = definition
    return out


def registry_market_families(registry: dict[str, FailureModeDefinition] | None = None) -> dict[str, tuple[str, ...]]:
    active = registry or load_failure_mode_registry()
    return {mode_id: definition.market_families for mode_id, definition in active.items()}


def failure_mode_exists(failure_mode_id: str, registry: dict[str, FailureModeDefinition] | None = None) -> bool:
    active = registry or load_failure_mode_registry()
    return str(failure_mode_id).strip() in active


def get_failure_mode(
    failure_mode_id: str,
    registry: dict[str, FailureModeDefinition] | None = None,
) -> FailureModeDefinition | None:
    active = registry or load_failure_mode_registry()
    return active.get(str(failure_mode_id).strip())

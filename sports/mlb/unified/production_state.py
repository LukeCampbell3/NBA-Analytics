from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class EngineState(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    LOCKED_HISTORICAL_VALIDATION = "LOCKED_HISTORICAL_VALIDATION"
    LOCKED_HISTORICAL_VALIDATION_FAILED = "LOCKED_HISTORICAL_VALIDATION_FAILED"
    PRODUCTION_CANDIDATE = "PRODUCTION_CANDIDATE"
    PRODUCTION_DEPLOYED_DARK = "PRODUCTION_DEPLOYED_DARK"
    LIVE_OPERATIONAL_CANARY = "LIVE_OPERATIONAL_CANARY"
    CANARY_FAILED = "CANARY_FAILED"
    PRODUCTION_ACTIVE = "PRODUCTION_ACTIVE"
    PRODUCTION_DEMOTED = "PRODUCTION_DEMOTED"
    ROLLBACK_ACTIVE = "ROLLBACK_ACTIVE"


class CapabilityAuthority(StrEnum):
    BLOCKED = "BLOCKED"
    SHADOW = "SHADOW"
    VALIDATION_ONLY = "VALIDATION_ONLY"
    CANARY = "CANARY"
    CERTIFIED = "CERTIFIED"
    PRODUCTION_ACTIVE = "PRODUCTION_ACTIVE"


ALLOWED_TRANSITIONS = {
    EngineState.DEVELOPMENT: {EngineState.LOCKED_HISTORICAL_VALIDATION},
    EngineState.LOCKED_HISTORICAL_VALIDATION: {
        EngineState.LOCKED_HISTORICAL_VALIDATION_FAILED,
        EngineState.PRODUCTION_CANDIDATE,
    },
    EngineState.LOCKED_HISTORICAL_VALIDATION_FAILED: {EngineState.LOCKED_HISTORICAL_VALIDATION},
    EngineState.PRODUCTION_CANDIDATE: {EngineState.PRODUCTION_DEPLOYED_DARK},
    EngineState.PRODUCTION_DEPLOYED_DARK: {EngineState.LIVE_OPERATIONAL_CANARY, EngineState.ROLLBACK_ACTIVE},
    EngineState.LIVE_OPERATIONAL_CANARY: {EngineState.PRODUCTION_ACTIVE, EngineState.CANARY_FAILED},
    EngineState.CANARY_FAILED: {EngineState.PRODUCTION_DEPLOYED_DARK, EngineState.ROLLBACK_ACTIVE},
    EngineState.PRODUCTION_ACTIVE: {EngineState.PRODUCTION_DEMOTED, EngineState.ROLLBACK_ACTIVE},
    EngineState.PRODUCTION_DEMOTED: {EngineState.LIVE_OPERATIONAL_CANARY, EngineState.ROLLBACK_ACTIVE},
    EngineState.ROLLBACK_ACTIVE: {EngineState.PRODUCTION_DEPLOYED_DARK},
}


def assert_transition(previous: EngineState, current: EngineState) -> None:
    if current not in ALLOWED_TRANSITIONS.get(previous, set()):
        raise ValueError(f"invalid engine transition: {previous.value} -> {current.value}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_engine_manifest(*, policy_hash: str, implementation_commit: str, state: EngineState,
                          capabilities: dict[str, str], rollback_reference: str | None = None) -> dict[str, Any]:
    active = "unified" if any(value == CapabilityAuthority.PRODUCTION_ACTIVE.value for value in capabilities.values()) else "legacy"
    return {
        "schema_version": 1,
        "active_engine": active,
        "fallback_engine": "legacy",
        "unified_available": True,
        "unified_policy_commit": "7c56729a9914eb9f903edffe9ca58b1a0a749ad4",
        "implementation_commit": implementation_commit,
        "policy_hash": policy_hash,
        "generated_at": utc_now(),
        "artifact_schema_version": "unified_mlb_v1",
        "certified_capabilities": sorted(name for name, value in capabilities.items() if value in {"CERTIFIED", "PRODUCTION_ACTIVE"}),
        "capabilities": capabilities,
        "production_state": state.value,
        "rollback_reference": rollback_reference,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {"schema_version", "active_engine", "fallback_engine", "policy_hash", "production_state", "capabilities"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["active_engine"] not in {"legacy", "unified"}:
        raise ValueError("invalid active engine")
    active_caps = [name for name, value in manifest["capabilities"].items() if value == "PRODUCTION_ACTIVE"]
    if manifest["active_engine"] == "unified" and not active_caps:
        raise ValueError("unified cannot be active without an active capability")
    if manifest["active_engine"] == "legacy" and active_caps:
        raise ValueError("legacy cannot be authoritative while unified capabilities are active")

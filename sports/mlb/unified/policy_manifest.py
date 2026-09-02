from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .decision import DecisionPolicy
from .market_registry import capability_payload
from .parlay import DEFAULT_TICKET_POLICIES


FROZEN_POLICY_COMMIT = "7c56729a9914eb9f903edffe9ca58b1a0a749ad4"
POLICY_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_values(policy: DecisionPolicy) -> dict[str, Any]:
    return {
        "minimum_usable_probability": policy.minimum_usable_probability,
        "minimum_probability_edge": policy.minimum_probability_edge,
        "minimum_conservative_ev": policy.minimum_conservative_ev,
        "uncertainty_multiplier": policy.uncertainty_multiplier,
        "valid_support_states": sorted(policy.valid_support_states),
        "valid_lineup_states": sorted(policy.valid_lineup_states),
        "valid_role_states": sorted(policy.valid_role_states),
        "valid_identity_states": sorted(policy.valid_identity_states),
        "require_exact_selection_ids": policy.require_exact_selection_ids,
    }


def build_policy_manifest(repo_root: Path) -> dict[str, Any]:
    policy_files = [
        "sports/mlb/unified/adapters.py",
        "sports/mlb/unified/decision.py",
        "sports/mlb/unified/market_conditioning.py",
        "sports/mlb/unified/market_registry.py",
        "sports/mlb/unified/parlay.py",
        "sports/mlb/unified/schemas.py",
    ]
    manifest = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_commit": FROZEN_POLICY_COMMIT,
        "decision_policy": _policy_values(DecisionPolicy()),
        "ticket_policies": {
            str(count): {
                key: value
                for key, value in vars(policy).items()
            }
            for count, policy in sorted(DEFAULT_TICKET_POLICIES.items())
        },
        "capabilities": capability_payload(),
        "staking": {
            "production_staking_authorized": False,
            "martingale": False,
            "automatic_scaling": False,
        },
        "compatibility_probability": {
            "field": "final_hit_probability",
            "semantic": "legacy post-calibration conservative probability",
            "missing_behavior": "FAIL_CLOSED",
        },
        "market_conditioning": {
            "minimum_identification_level": 2,
            "weight_clip": [0.25, 4.0],
            "minimum_effective_sample_fraction": 0.5,
            "authority": "DEVELOPMENT",
        },
        "policy_file_sha256": {
            name: sha256_file(repo_root / name)
            for name in policy_files
        },
        "model_artifacts": [],
        "calibrator_artifacts": [],
        "model_artifact_note": "Frozen unified code consumes compatibility probabilities; no separate unified model/calibrator artifact is loaded.",
    }
    manifest["policy_hash"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return manifest


def verify_policy_manifest(manifest: dict[str, Any]) -> bool:
    expected = manifest.get("policy_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "policy_hash"}
    actual = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    return bool(expected) and expected == actual

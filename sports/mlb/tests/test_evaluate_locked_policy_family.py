from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports.mlb.governance.evaluate_locked_policy_family import evaluate_family
from sports.mlb.governance.policy_governance import load_policy_registry


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "governance" / "policies" / "mlb_policy_family_v1.json"


def test_locked_validation_refuses_unfrozen_development_family() -> None:
    registry = load_policy_registry(REGISTRY_PATH)
    evidence = pd.DataFrame(
        {
            "slate_id": ["MLB_20260806"],
            "snapshot_id": ["snapshot-1"],
            "policy_version": [registry["policies"][0]["policy_version"]],
            "policy_digest": [registry["policies"][0]["policy_digest"]],
            "evidence_partition": ["LOCKED_VALIDATION"],
            "capture_label": ["FULL_SLATE_SNAPSHOT"],
            "decision_frozen_at_utc": ["2026-08-06T16:00:00Z"],
            "slate_date": ["2026-08-06"],
            "eligible_slate": [True],
            "action_taken": [True],
            "resolved": [True],
            "selection_count": [1],
            "eligible_candidate_count": [10],
            "selected_candidate_count": [1],
            "daily_unit_return": [0.5],
        }
    )

    report = evaluate_family(registry, evidence)

    first = report["policies"][0]
    assert first["status"] == "REJECTED"
    assert "POLICY_NOT_IN_LOCKED_VALIDATION" in first["blocking_reasons"]
    assert "POLICY_FAMILY_NOT_FROZEN" in first["blocking_reasons"]
    assert report["method"] == "LEARN_THEN_TEST_BONFERRONI_HOEFFDING_V1"

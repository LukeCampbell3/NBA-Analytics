from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sports.mlb.governance.policy_governance import canonical_json_hash, load_policy_registry
from sports.mlb.governance.replay_frozen_policy import replay_single_policy


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "governance" / "policies" / "mlb_policy_family_v1.json"


def _policy(*, frozen: bool) -> dict:
    policy = json.loads(json.dumps(load_policy_registry(REGISTRY_PATH)["policies"][0]))
    policy.pop("policy_digest")
    policy["decision_rule"]["family_is_frozen"] = frozen
    policy["decision_rule"]["minimum_model_score"] = 0.1
    policy["policy_digest"] = canonical_json_hash(policy)
    return policy


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eligible_by_input_rules": True,
                "market": "H",
                "side": "OVER",
                "book": "draftkings",
                "line": 0.5,
                "price_decimal": 1.8,
                "model_score": 0.5,
                "event_id": "g1",
                "player_id": "best",
            },
            {
                "eligible_by_input_rules": True,
                "market": "H",
                "side": "OVER",
                "book": "draftkings",
                "line": 0.5,
                "price_decimal": 2.0,
                "model_score": 0.4,
                "event_id": "g1",
                "player_id": "same_game",
            },
            {
                "eligible_by_input_rules": True,
                "market": "TB",
                "side": "OVER",
                "book": "fanduel",
                "line": 1.5,
                "price_decimal": 2.1,
                "model_score": 0.3,
                "event_id": "g2",
                "player_id": "other_game",
            },
        ]
    )


def test_replay_refuses_unfrozen_policy() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        replay_single_policy(_universe(), _policy(frozen=False))


def test_replay_retains_rejections_and_applies_exposure_rules() -> None:
    result = replay_single_policy(_universe(), _policy(frozen=True))

    assert len(result) == 3
    assert set(result.loc[result["selected_by_policy"], "player_id"]) == {"best", "other_game"}
    rejected = result.loc[result["player_id"] == "same_game"].iloc[0]
    assert rejected["replay_rejection_reason"] == "GAME_EXPOSURE_LIMIT"

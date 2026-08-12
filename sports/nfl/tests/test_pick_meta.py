from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import pytest

from sports.nfl.predictions.pick_meta import score_with_artifact, validate_artifact


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "sports/nfl/scripts/train_nfl_pick_meta_selector.py"
SPEC = importlib.util.spec_from_file_location("train_nfl_pick_meta_selector", SCRIPT_PATH)
assert SPEC and SPEC.loader
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def test_recent_meta_policy_reproduces_locked_weeks() -> None:
    recent = pd.read_csv(
        REPO_ROOT / "sports/nfl/data/evaluation/recent_selector_pool_2025.csv"
    )
    development = recent.loc[recent["week"].le(12)]
    locked = recent.loc[recent["week"].ge(13)]

    policy, _ = TRAINER.select_policy(development)
    result = TRAINER.evaluate(locked, policy)

    assert policy == {
        "minimum_side_probability": 0.58,
        "minimum_no_vig_advantage": 0.1,
        "minimum_price": -130,
        "maximum_price": 130,
        "weekly_cap": 6,
    }
    assert result["graded_decisions"] == 36
    assert result["wins"] == 26
    assert result["hit_rate"] == 0.7222
    assert result["roi"] == 0.3486


def test_meta_artifact_is_nfl_only_and_applies_frozen_gates() -> None:
    artifact = joblib.load(
        REPO_ROOT / "sports/nfl/model/nfl_pick_meta_selector.joblib"
    )
    rows = pd.DataFrame(
        {
            "target": ["passing", "passing"],
            "estimated_side_probability": [0.64, 0.57],
            "probability_advantage": [0.12, 0.12],
            "selected_price": [-120, -120],
        }
    )

    scored = score_with_artifact(rows, artifact)

    assert artifact["sport"] == "NFL"
    assert scored["meta_eligible"].tolist() == [True, False]
    with pytest.raises(ValueError, match="NFL loss-aware artifact"):
        validate_artifact(
            {
                "sport": "MLB",
                "artifact_type": artifact["artifact_type"],
                "model_version": artifact["model_version"],
                "policy": artifact["policy"],
            }
        )

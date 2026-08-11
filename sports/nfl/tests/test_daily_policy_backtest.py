from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "sports/nfl/scripts/backtest_nfl_daily_policy.py"
SPEC = importlib.util.spec_from_file_location("backtest_nfl_daily_policy", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_locked_daily_policy_reproduces_committed_holdout() -> None:
    pool = pd.read_csv(
        REPO_ROOT / "sports/nfl/data/evaluation/market_selector_validated_pool_2022.csv"
    )

    result = MODULE.evaluate(pool)

    assert result["singles"]["graded_decisions"] == 210
    assert result["singles"]["wins"] == 127
    assert result["singles"]["hit_rate"] == 0.6048
    assert result["singles"]["roi"] == 0.13
    assert result["parlay"]["wins"] == 2
    assert result["parlay"]["losses"] == 16

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import optimize_walk_forward_policy as optimizer


def test_split_dates_keeps_holdout_last_and_disjoint() -> None:
    dates = [f"2026-06-{day:02d}" for day in range(1, 31)]

    train, validation, holdout = optimizer.split_dates(dates, validation_days=7, holdout_days=7)

    assert holdout == dates[-7:]
    assert validation == dates[-14:-7]
    assert not (set(train) & set(validation) or set(train) & set(holdout) or set(validation) & set(holdout))


def test_score_rows_rejects_invalid_prices_from_priced_roi() -> None:
    rows = pd.DataFrame(
        [
            {"result": "win", "probability": 0.7, "price_confirmed": True, "units": 0.9},
            {"result": "loss", "probability": 0.7, "price_confirmed": False, "units": -1.0},
        ]
    )

    stats = optimizer.score_rows(rows, date_count=1)

    assert stats["priced_plays"] == 1
    assert stats["priced_net_units"] == 0.9
    assert stats["proxy_net_units"] < 0.0


def _ledger_row(**overrides) -> dict:
    row = {
        "date": "2026-06-18",
        "player": "Test Player",
        "player_id": "test_player",
        "game_id": "game_1",
        "team": "AAA",
        "market_bucket": "TB|UNDER|1.5",
        "selection_score": 1.0,
        "hit_probability": 0.8,
        "probability": 0.85,
        "push_probability": 0.0,
        "abs_edge": 1.0,
        "historical_bet_profile_support": 20,
        "historical_bet_profile_win_rate": 0.6,
        "historical_market_availability_support": 5,
        "historical_market_availability_rate": 0.8,
        "market_source": "real",
    }
    row.update(overrides)
    return row


def test_select_config_requires_prior_market_placeability() -> None:
    ledger = pd.DataFrame([_ledger_row()])
    config = optimizer.policy_grid()[0]

    selected = optimizer.select_config(ledger, {"2026-06-18"}, config)

    assert selected.empty


def test_select_config_excludes_synthetic_market_source() -> None:
    """Real bug found investigating an implausible 96-100% holdout hit
    rate: this filter never checked market_source, and the tool's own
    broad_candidate_policy() ledger is ~97.5% synthetic -- a config
    "optimized" against unfiltered rows is mostly measuring the model
    beating its own estimated lines, never a real, executable price.
    The real live selector always runs with --require-real-market-
    source, so this tool must too, unconditionally."""
    real_row = _ledger_row(historical_bet_profile_support=20, historical_market_availability_support=5)
    synthetic_row = _ledger_row(market_source="synthetic", player_id="synthetic_player")
    ledger = pd.DataFrame([real_row, synthetic_row])
    config = {
        **optimizer.policy_grid()[0],
        "min_historical_bet_profile_support": 0,
        "min_historical_market_availability_support": 0,
    }

    selected = optimizer.select_config(ledger, {"2026-06-18"}, config)

    assert list(selected["market_source"]) == ["real"]

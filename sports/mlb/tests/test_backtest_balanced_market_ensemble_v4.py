from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import backtest_balanced_market_ensemble_v4 as backtest  # noqa: E402


def _slate(date: str, winner_balanced: float, loser_balanced: float) -> list[dict]:
    return [
        {"date": date, "candidate_id": f"{date}-w", "win": 1, "balanced_probability": winner_balanced, "market_probability": 0.6, "base_ev": 0.1, "v19_order_score": 0.1},
        {"date": date, "candidate_id": f"{date}-l", "win": 0, "balanced_probability": loser_balanced, "market_probability": 0.6, "base_ev": -0.1, "v19_order_score": -0.1},
    ]


def test_walk_forward_scores_only_after_four_strictly_prior_slates() -> None:
    rows = [row for day in range(1, 7) for row in _slate(f"2026-08-{day:02d}", 0.7, 0.5)]
    scored, fits = backtest.score_walk_forward(rows)
    assert sorted({row["date"] for row in scored}) == ["2026-08-05", "2026-08-06"]
    assert [fit["training_slates"] for fit in fits] == [4, 5]


def test_backtest_reports_zero_coverage_without_inventing_accuracy() -> None:
    rows = [row for day in range(1, 6) for row in _slate(f"2026-08-{day:02d}", 0.59, 0.58)]
    report = backtest.build_report(rows)
    assert report["selection"]["eligible_rows"] == 0
    assert report["selection"]["coverage"] == 0.0
    assert report["selection"]["hit_rate"] is None
    assert report["selection"]["roi"] is None


def test_ranking_inference_equal_weights_slates() -> None:
    rows = []
    rows += _slate("2026-08-05", 0.9, 0.1) * 50
    rows += _slate("2026-08-06", 0.1, 0.9)
    # Add four prior dates so these two are evaluated.
    rows += [row for day in range(1, 5) for row in _slate(f"2026-08-{day:02d}", 0.6, 0.6)]
    report = backtest.build_report(rows)
    summary = report["ranking"]["balanced_probability"]
    assert summary["independent_slates"] == 2
    assert summary["mean_slate_auc"] == pytest.approx(0.5)


def test_implied_price_round_trip_probability() -> None:
    assert backtest.implied_price(0.6) == pytest.approx(-150.0)
    assert backtest.implied_price(0.4) == pytest.approx(150.0)


def test_exact_price_is_recovered_from_recorded_ev_not_consensus_probability() -> None:
    row = {"balanced_probability": 0.60, "market_probability": 0.55, "base_ev": -0.04}
    # D=(.96/.60)=1.60 -> exact price -166.666..., not the -122.22
    # that the separate 55% market diagnostic would imply.
    assert backtest.exact_price_from_row(row) == pytest.approx(-166.6666667)

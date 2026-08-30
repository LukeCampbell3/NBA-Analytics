from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import balanced_ranking_v3 as ranking  # noqa: E402


def _rows(date: str, outcomes: list[int], balanced: list[float], market: list[float]) -> list[dict]:
    return [
        {
            "candidate_id": f"{date}_{index}",
            "date": date,
            "game_id": f"g{index // 2}",
            "win": outcome,
            "balanced_probability": balanced[index],
            "market_probability": market[index],
            "base_ev": balanced[index] - market[index],
            "v19_order_score": balanced[index] - market[index],
        }
        for index, outcome in enumerate(outcomes)
    ]


def test_pairwise_concordance_counts_ties_as_half() -> None:
    value, pairs = ranking.pairwise_concordance([0.8, 0.6, 0.6, 0.4], [1, 1, 0, 0])
    assert pairs == 4
    assert value == pytest.approx(0.875)


def test_all_win_or_all_loss_slate_has_undefined_concordance() -> None:
    assert ranking.pairwise_concordance([0.8, 0.7], [1, 1]) == (None, 0)
    assert ranking.pairwise_concordance([0.8, 0.7], [0, 0]) == (None, 0)


def test_top_k_and_concordance_follow_score_order() -> None:
    rows = _rows("2026-08-01", [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], [0.5] * 4)
    metric = ranking.slate_metric(rows, "balanced_probability")
    assert metric.concordance == pytest.approx(0.75)
    assert metric.top_1_hit_rate == 1.0
    assert metric.top_3_hit_rate == pytest.approx(2 / 3)
    assert metric.top_1_lift == 0.5


def test_pair_count_never_becomes_independent_slate_count() -> None:
    rows = []
    rows += _rows("2026-08-01", [1] * 50 + [0] * 50, list(reversed(range(100))), list(range(100)))
    rows += _rows("2026-08-02", [1, 0], [0.9, 0.1], [0.1, 0.9])
    metrics = ranking.evaluate_rows(rows)
    summary = ranking.summarize(metrics, phase="locked")
    assert summary["independent_slates"] == 2
    assert summary["score_summaries"]["balanced_probability"]["comparable_pairs_descriptive_only"] == 2501
    assert summary["status"] == "INSUFFICIENT_INDEPENDENT_SLATES"


def test_locked_acceptance_requires_eight_slates_even_with_perfect_ranking() -> None:
    rows = []
    for day in range(1, 8):
        rows += _rows(f"2026-08-{day:02d}", [1, 0], [0.9, 0.1], [0.1, 0.9])
    summary = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
    assert summary["status"] == "INSUFFICIENT_INDEPENDENT_SLATES"


def test_acceptance_requires_incremental_market_and_v19_ranking() -> None:
    rows = []
    for day in range(1, 9):
        # Balanced and market are equally perfect; balanced is therefore useful
        # but has no incremental advantage and must not be accepted.
        rows += _rows(f"2026-08-{day:02d}", [1, 0], [0.9, 0.1], [0.9, 0.1])
    summary = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
    assert summary["score_summaries"]["balanced_probability"]["mean_slate_concordance"] == 1.0
    assert summary["status"] == "RANKING_SIGNAL_NOT_ACCEPTED"


from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import local_node_selector_v2 as selector  # noqa: E402


def _row(
    slate: int,
    index: int,
    *,
    balanced: float = 0.52,
    market: float = 0.60,
    win: int = 1,
    target: str = "H",
    direction: str = "OVER",
    line: float = 0.5,
) -> dict:
    return {
        "candidate_id": f"s{slate}_{index}",
        "slate_date": f"2026-08-{slate:02d}",
        "game_id": f"g{slate}_{index}",
        "target": target,
        "direction": direction,
        "line": line,
        "balanced_probability": balanced,
        "market_probability": market,
        "win": win,
    }


def _candidate(**overrides) -> dict:
    row = {
        "candidate_id": "candidate",
        "slate_date": "2026-08-20",
        "game_id": "candidate_game",
        "target": "H",
        "direction": "OVER",
        "line": 0.5,
        "balanced_probability": 0.52,
        "market_probability": 0.60,
        "price": -120,
    }
    row.update(overrides)
    return row


def test_v2_uses_only_strictly_prior_same_family_analogues() -> None:
    history = [_row(slate, 0) for slate in range(1, 16)]
    history += [
        _row(9, 0, target="HR"),
        _row(10, 0, direction="UNDER"),
        _row(11, 0, line=1.5),
        {**_row(20, 0), "slate_date": "2026-08-20"},
        {**_row(21, 0), "slate_date": "2026-08-21"},
    ]

    score = selector.score_candidate(_candidate(), history)

    assert score.independent_slates == 15
    assert score.neighbor_rows == 15
    assert score.family == ("H", "OVER", 0.5, "")


def test_high_hit_region_with_negative_residual_is_rejected() -> None:
    # Seven wins in ten rows is a good absolute hit rate, but 70% outcomes
    # against 72% balanced probability are negative calibration evidence.
    history = []
    for slate in range(1, 16):
        history.append(_row(slate, 0, balanced=0.72, market=0.74, win=1 if slate <= 10 else 0))

    score = selector.score_candidate(
        _candidate(balanced_probability=0.72, market_probability=0.74, price=-130),
        history,
    )

    assert score.mean_slate_hit_rate == pytest.approx(10 / 15)
    assert score.mean_slate_residual == pytest.approx(10 / 15 - 0.72)
    assert score.eligible is False
    assert "residual_lcb_not_meaningful" in score.reasons


def test_positive_slate_clustered_residual_produces_only_lcb_correction() -> None:
    # Fifteen independent slates, each 4/5, create a stable +28pp residual over
    # balanced P=.52. The usable correction is the lower bound, not +.28.
    history = []
    for slate in range(1, 16):
        history.extend(_row(slate, index, win=1 if index < 4 else 0) for index in range(5))

    score = selector.score_candidate(_candidate(price=110), history)

    assert score.independent_slates == 15
    assert score.neighbor_rows == 75
    assert score.mean_slate_residual == pytest.approx(0.28)
    assert score.residual_lcb == pytest.approx(0.28)
    assert score.safe_correction == pytest.approx(0.28)
    assert score.support_probability == pytest.approx(0.8)
    assert score.recovered_probability == pytest.approx(0.8)
    assert score.eligible is True


def test_slates_are_equal_weighted_despite_different_row_counts() -> None:
    # One five-row losing slate must have the same residual weight as each
    # one-row winning slate, rather than five times the influence.
    history = []
    history.extend(_row(1, index, win=0) for index in range(5))
    history.extend(_row(slate, 0, win=1) for slate in range(2, 16))

    score = selector.score_candidate(_candidate(price=110), history, min_residual_lcb=-1.0)

    expected = ((0.0 - 0.52) + 14 * (1.0 - 0.52)) / 15
    assert score.independent_slates == 15
    assert score.neighbor_rows == 19
    assert score.mean_slate_residual == pytest.approx(expected)


def test_minimum_independent_slates_forces_abstention() -> None:
    history = [_row(slate, 0) for slate in range(1, 15)]
    score = selector.score_candidate(_candidate(price=110), history)

    assert score.independent_slates == 14
    assert score.eligible is False
    assert "insufficient_independent_slates" in score.reasons


def test_support_lcb_caps_recovered_probability() -> None:
    history = []
    # Vary slate hit rates so both confidence bounds carry uncertainty.
    for slate in range(1, 13):
        wins = 4 if slate <= 9 else 3
        history.extend(_row(slate, index, balanced=0.50, win=1 if index < wins else 0) for index in range(5))

    score = selector.score_candidate(
        _candidate(balanced_probability=0.54, price=120),
        history,
        min_residual_lcb=0.0,
    )

    assert score.safe_correction is not None
    assert score.support_probability is not None
    assert score.recovered_probability == pytest.approx(score.support_probability)
    assert score.recovered_probability < 0.54 + score.safe_correction


def test_ev_ranks_only_nodes_that_clear_residual_evidence() -> None:
    history = []
    for slate in range(1, 16):
        history.extend(_row(slate, index, win=1 if index < 4 else 0) for index in range(5))
    candidates = [
        _candidate(candidate_id="short_price", price=-140),
        _candidate(candidate_id="better_price", price=110),
        _candidate(
            candidate_id="wrong_family_high_ev",
            target="TB",
            price=300,
        ),
    ]

    selected, scores = selector.select_node(candidates, history)
    by_id = {score.candidate_id: score for score in scores}

    assert selected is not None
    assert selected["candidate_id"] == "better_price"
    assert by_id["wrong_family_high_ev"].eligible is False
    assert "insufficient_independent_slates" in by_id["wrong_family_high_ev"].reasons


def test_student_t_critical_matches_known_small_sample_value() -> None:
    assert selector._student_t_critical(0.975, 14) == pytest.approx(2.1448, abs=0.006)


def test_selector_applies_simultaneous_candidate_scan_confidence() -> None:
    history = []
    for slate in range(1, 16):
        history.extend(_row(slate, index, win=1 if index < 4 else 0) for index in range(5))
    candidates = [_candidate(candidate_id=f"candidate_{index}", price=110) for index in range(4)]

    _, scores = selector.select_node(candidates, history)

    assert all(score.confidence_level == pytest.approx(1.0 - 0.05 / 4) for score in scores)

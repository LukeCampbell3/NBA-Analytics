from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import local_node_selector as selector  # noqa: E402


def _history(
    *,
    start_day: int,
    rows: int,
    balanced: float,
    market: float,
    wins: int,
    prefix: str,
) -> list[dict]:
    output = []
    for index in range(rows):
        day = start_day + index // 8
        output.append(
            {
                "candidate_id": f"{prefix}_{index}",
                "slate_date": f"2026-08-{day:02d}",
                "game_id": f"{prefix}_g{index}",
                "balanced_probability": balanced + (index % 3 - 1) * 0.002,
                "market_probability": market + (index % 3 - 1) * 0.002,
                "win": 1 if index < wins else 0,
            }
        )
    return output


def test_local_score_uses_only_strictly_prior_settled_slates() -> None:
    candidate = {
        "candidate_id": "target",
        "slate_date": "2026-08-20",
        "game_id": "target_game",
        "balanced_probability": 0.55,
        "market_probability": 0.64,
        "price": -145,
    }
    history = _history(start_day=10, rows=24, balanced=0.55, market=0.64, wins=18, prefix="prior")
    # These would make the region look perfect if outcome leakage were allowed.
    history.extend(
        {
            "candidate_id": f"future_{index}",
            "slate_date": "2026-08-20" if index < 10 else "2026-08-21",
            "game_id": f"future_g{index}",
            "balanced_probability": 0.55,
            "market_probability": 0.64,
            "win": 1,
        }
        for index in range(20)
    )

    score = selector.score_candidate(
        candidate,
        history,
        k_neighbors=40,
        min_neighbors=20,
        min_lcb=0.50,
    )

    assert score.neighbor_count == 24
    assert score.local_hit_rate == pytest.approx(18 / 24)


def test_selector_recovers_supported_lower_p_node_instead_of_highest_balanced_p() -> None:
    # Region A: balanced P systematically underestimates outcomes and the market
    # points in the same direction.  Region B has the higher displayed balanced
    # P but historical analogues do not support an upward correction.
    history = []
    history += _history(start_day=1, rows=40, balanced=0.55, market=0.64, wins=30, prefix="recovery")
    history += _history(start_day=1, rows=40, balanced=0.69, market=0.70, wins=24, prefix="high_p")

    candidates = [
        {
            "candidate_id": "supported_low_p",
            "slate_date": "2026-08-15",
            "game_id": "a",
            "balanced_probability": 0.55,
            "market_probability": 0.64,
            "price": -135,
        },
        {
            "candidate_id": "unsupported_high_p",
            "slate_date": "2026-08-15",
            "game_id": "b",
            "balanced_probability": 0.69,
            "market_probability": 0.70,
            "price": -180,
        },
    ]

    selected, scores = selector.select_node(
        candidates,
        history,
        k_neighbors=40,
        min_neighbors=20,
        min_market_disagreement=0.03,
        min_local_correction=0.02,
        min_lcb=0.55,
    )
    by_id = {score.candidate_id: score for score in scores}

    assert selected is not None
    assert selected["candidate_id"] == "supported_low_p"
    assert by_id["supported_low_p"].eligible is True
    assert by_id["supported_low_p"].local_correction > 0.10
    assert by_id["unsupported_high_p"].eligible is False
    assert "market_disagreement_too_small" in by_id["unsupported_high_p"].reasons


def test_selector_rejects_apparent_recovery_when_sample_is_too_small() -> None:
    candidate = {
        "candidate_id": "tiny_sample",
        "slate_date": "2026-08-15",
        "game_id": "a",
        "balanced_probability": 0.54,
        "market_probability": 0.65,
        "price": -130,
    }
    # 9/10 looks excellent but is deliberately below the evidence requirement.
    history = _history(start_day=1, rows=10, balanced=0.54, market=0.65, wins=9, prefix="tiny")
    score = selector.score_candidate(candidate, history, k_neighbors=40, min_neighbors=20)

    assert score.local_hit_rate == pytest.approx(0.9)
    assert score.eligible is False
    assert "insufficient_local_support" in score.reasons


def test_selector_ranks_ev_only_after_reliability_gates() -> None:
    history = []
    history += _history(start_day=1, rows=40, balanced=0.55, market=0.64, wins=30, prefix="a")
    history += _history(start_day=1, rows=40, balanced=0.57, market=0.66, wins=30, prefix="b")
    candidates = [
        {
            "candidate_id": "shorter_price",
            "slate_date": "2026-08-15",
            "game_id": "a",
            "balanced_probability": 0.55,
            "market_probability": 0.64,
            "price": -165,
        },
        {
            "candidate_id": "better_price",
            "slate_date": "2026-08-15",
            "game_id": "b",
            "balanced_probability": 0.57,
            "market_probability": 0.66,
            "price": -125,
        },
    ]

    selected, scores = selector.select_node(
        candidates,
        history,
        k_neighbors=30,
        min_neighbors=20,
        min_lcb=0.55,
    )
    assert all(score.eligible for score in scores)
    assert selected is not None
    assert selected["candidate_id"] == "better_price"


def test_wilson_lcb_penalizes_uncertain_node_even_with_high_point_hit_rate() -> None:
    tiny = selector.wilson_lower_bound(9, 10)
    supported = selector.wilson_lower_bound(30, 40)
    assert tiny is not None and supported is not None
    assert tiny < 0.75
    assert supported > 0.60

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "parlay_v2"))

import select_mlb_pitcher_parlay as legacy_pitcher  # noqa: E402
from pitcher_alt_line_frontier import (  # noqa: E402
    MAX_TWO_SIDED_MODEL_MARKET_GAP,
    build_pitcher_parlay_frontier,
)


def _leg(
    pitcher_id: int,
    *,
    probability: float,
    no_vig: float | None,
    price: int,
    game_id: str,
) -> legacy_pitcher.PitcherKLeg:
    return legacy_pitcher.PitcherKLeg(
        pitcher_id=pitcher_id,
        pitcher_name=f"Pitcher {pitcher_id}",
        team=f"T{pitcher_id}",
        opponent=f"O{pitcher_id}",
        game_id=game_id,
        line=4.5,
        side="over",
        model_probability=probability,
        no_vig_market_probability=no_vig,
        price_american=price,
        sportsbook="fanduel",
        market_books=1,
        price_confirmed=True,
        leg_authorized=False,
    )


def test_frontier_rejects_large_two_sided_model_market_conflict() -> None:
    # Mirrors the failure class seen on the August 31 board: a season-Poisson
    # probability can look enormously +EV only because it is 15-30 points
    # away from a real two-sided sportsbook market. That is unresolved model
    # reliability, not automatic actionable edge.
    extreme_a = _leg(1, probability=.781, no_vig=.510, price=-118, game_id="g1")
    extreme_b = _leg(2, probability=.668, no_vig=.504, price=-115, game_id="g2")

    assert abs(extreme_a.model_probability - extreme_a.no_vig_market_probability) > MAX_TWO_SIDED_MODEL_MARKET_GAP
    assert abs(extreme_b.model_probability - extreme_b.no_vig_market_probability) > MAX_TWO_SIDED_MODEL_MARKET_GAP
    assert build_pitcher_parlay_frontier([extreme_a, extreme_b]) is None


def test_frontier_keeps_reasonable_positive_ev_two_sided_disagreement() -> None:
    leg_a = _leg(1, probability=.72, no_vig=.62, price=-120, game_id="g1")
    leg_b = _leg(2, probability=.74, no_vig=.64, price=-115, game_id="g2")

    candidate = build_pitcher_parlay_frontier([leg_a, leg_b])

    assert candidate is not None
    assert abs(leg_a.model_probability - leg_a.no_vig_market_probability) <= MAX_TWO_SIDED_MODEL_MARKET_GAP
    assert abs(leg_b.model_probability - leg_b.no_vig_market_probability) <= MAX_TWO_SIDED_MODEL_MARKET_GAP
    assert candidate.expected_value_per_unit is not None and candidate.expected_value_per_unit >= .05


def test_one_sided_alt_line_retains_existing_shadow_frontier_behavior() -> None:
    # No exact opposite-side quote means no no-vig disagreement can be
    # measured. Preserve the existing alt-line behavior; authorization remains
    # controlled elsewhere by its evidence/support path.
    leg_a = _leg(1, probability=.75, no_vig=None, price=-105, game_id="g1")
    leg_b = _leg(2, probability=.72, no_vig=None, price=100, game_id="g2")

    candidate = build_pitcher_parlay_frontier([leg_a, leg_b])
    assert candidate is not None

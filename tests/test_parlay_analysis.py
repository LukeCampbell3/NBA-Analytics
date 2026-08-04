from __future__ import annotations

import pytest

from sports.parlay_analysis import annotate_parlay_board, score_candidate_parlays


def _play(
    *,
    player: str,
    game_id: str,
    team: str,
    direction: str,
    probability: float,
    price: float,
    sportsbook_key: str,
) -> dict:
    return {
        "player": player,
        "player_display_name": player,
        "game_id": game_id,
        "team": team,
        "target": "TB",
        "direction": direction,
        "market_bucket": f"TB|{direction}|1.5",
        "estimated_graded_hit_rate": probability,
        "final_pool_quality_score": 0.80,
        "expected_value_per_unit": 0.12,
        "selected_side_price": price,
        "selected_sportsbook_key": sportsbook_key,
        "selected_sportsbook": "Caesars" if sportsbook_key == "caesars" else "BetMGM",
        "parlay_precision_eligible": True,
    }


def test_mlb_parlay_requires_one_confirmed_sportsbook_and_reports_return() -> None:
    plays = [
        _play(
            player="High Confidence Under",
            game_id="game-1",
            team="ATH",
            direction="UNDER",
            probability=0.766,
            price=-205,
            sportsbook_key="caesars",
        ),
        _play(
            player="Positive EV Over",
            game_id="game-2",
            team="SD",
            direction="OVER",
            probability=0.535,
            price=117,
            sportsbook_key="caesars",
        ),
    ]

    parlays = score_candidate_parlays(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
    )

    assert len(parlays) == 1
    parlay = parlays[0]
    assert parlay["same_sportsbook_confirmed"] is True
    assert parlay["sportsbook_key"] == "caesars"
    assert parlay["combined_decimal_price"] == pytest.approx((1 + 100 / 205) * 2.17)
    assert parlay["combined_american_price"] > 200
    assert parlay["expected_return_per_unit"] > 0.0


def test_mlb_parlay_rejects_legs_priced_at_different_sportsbooks() -> None:
    plays = [
        _play(
            player="First Player",
            game_id="game-1",
            team="ATH",
            direction="UNDER",
            probability=0.766,
            price=-205,
            sportsbook_key="caesars",
        ),
        _play(
            player="Second Player",
            game_id="game-2",
            team="SD",
            direction="OVER",
            probability=0.535,
            price=117,
            sportsbook_key="mgm",
        ),
    ]

    parlays = score_candidate_parlays(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
    )

    assert parlays == []


def test_mlb_board_selects_only_the_best_executable_ticket() -> None:
    plays = [
        _play(
            player="Anchor",
            game_id="game-1",
            team="ATH",
            direction="UNDER",
            probability=0.78,
            price=-180,
            sportsbook_key="caesars",
        ),
        _play(
            player="Over One",
            game_id="game-2",
            team="SD",
            direction="OVER",
            probability=0.54,
            price=120,
            sportsbook_key="caesars",
        ),
        _play(
            player="Over Two",
            game_id="game-3",
            team="COL",
            direction="OVER",
            probability=0.52,
            price=125,
            sportsbook_key="caesars",
        ),
    ]

    payload = annotate_parlay_board(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
    )

    assert payload["summary"]["selected_parlay_count"] == 1
    assert payload["summary"]["same_sportsbook_required"] is True
    assert payload["pairs"][0]["leg_names"] == ["Anchor", "Over One"]
    assert payload["summary"]["best_expected_return_per_unit"] > 0.0

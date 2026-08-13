from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import select_daily_parlay as parlay_selector


def _candidate(*, player: str, game_id: str, probability: float, price: float, direction: str = "OVER") -> SimpleNamespace:
    return SimpleNamespace(
        player=player,
        player_id=player.lower().replace(" ", "_"),
        team=player[:3].upper(),
        game_id=game_id,
        target="H",
        direction=direction,
        prediction=1.2,
        market_line=0.5,
        market_source="real",
        market_bucket="H|OVER|0.5",
        historical_bucket_key="H|OVER|0.5",
        calibrated_graded_hit_rate=probability,
        model_hit_probability=probability,
        selection_score=0.85,
        expected_value_per_unit=0.08,
        selected_side_price=price,
        selected_sportsbook_key="draftkings",
        selected_sportsbook="DraftKings",
        price_confirmed=True,
        market_books=6,
        market_common_books=3,
        market_book_keys="draftkings|fanduel|caesars|betmgm|espnbet|bovada",
        market_common_book_keys="draftkings|fanduel|caesars",
        history_rows=80,
        days_since_history=1,
        model_selected="et",
        game_status_code="P",
        run_date=date(2026, 8, 6),
        raw={
            "Player_Type": "hitter",
            "Opponent": "OPP",
            "Game_Date": "2026-08-06",
            "Commence_Time_UTC": "2026-08-06T23:00:00Z",
            "Game_Status_Detail": "Scheduled",
        },
    )


def test_adaptive_ticket_can_use_three_legs_when_two_legs_do_not_reach_even_money() -> None:
    candidates = [
        _candidate(player="One", game_id="g1", probability=0.75, price=-250),
        _candidate(player="Two", game_id="g2", probability=0.75, price=-250),
        _candidate(player="Three", game_id="g3", probability=0.75, price=-250),
    ]

    ticket, _ = parlay_selector.select_ticket(
        candidates,
        min_legs=2,
        max_legs=4,
        min_leg_probability=0.62,
        min_ticket_probability=0.40,
        min_combined_decimal_price=2.0,
        min_expected_return=0.0,
    )

    assert ticket is not None
    assert ticket["leg_count"] == 3
    assert ticket["projected_probability"] >= 0.40


def test_anchor_filter_is_over_only_and_requires_playable_market_support() -> None:
    eligible = _candidate(player="Eligible", game_id="g1", probability=0.68, price=-180)
    under = _candidate(player="Under", game_id="g2", probability=0.75, price=-180, direction="UNDER")
    thin = _candidate(player="Thin", game_id="g3", probability=0.70, price=-180)
    thin.market_books = 2

    kept, rejected = parlay_selector.filter_anchor_candidates(
        [eligible, under, thin],
        min_leg_probability=0.62,
    )

    assert kept == [eligible]
    assert rejected["not_over"] == 1
    assert rejected["insufficient_book_coverage"] == 1


def test_ticket_ladder_keeps_best_two_leg_ticket_and_adds_longer_options() -> None:
    candidates = [
        _candidate(player=f"Player {index}", game_id=f"g{index}", probability=0.75, price=-150)
        for index in range(1, 5)
    ]

    ladder, considered = parlay_selector.select_ticket_ladder(
        candidates,
        min_legs=2,
        max_legs=4,
        min_leg_probability=0.62,
        base_min_ticket_probability=0.40,
        min_combined_decimal_price=2.0,
        min_expected_return=0.0,
    )

    assert len(considered) == 4
    assert [ticket["leg_count"] for ticket in ladder] == [2, 3, 4]
    assert [ticket["ticket_tier"] for ticket in ladder] == ["consistency", "balanced", "extended"]
    assert ladder[0]["projected_probability"] > ladder[1]["projected_probability"] > ladder[2]["projected_probability"]
    assert all(ticket["same_sportsbook_confirmed"] for ticket in ladder)
    assert all(not ticket["same_game"] for ticket in ladder)


def test_profit_boost_uses_only_confirmed_higher_lines_and_recalculates_probability(tmp_path: Path) -> None:
    candidates = [
        _candidate(player="First Player", game_id="g1", probability=0.72, price=-180),
        _candidate(player="Second Player", game_id="g2", probability=0.70, price=-170),
    ]
    for candidate in candidates:
        candidate.prediction = 1.5
    observations = tmp_path / "provider.csv"
    pd.DataFrame(
        [
            {
                "source": "sportsgameodds",
                "source_market_id": f"alt-{index}",
                "player_name": candidate.player,
                "market_type": "batter_hits",
                "side": "over",
                "line": 1.5,
                "price_american": 200,
                "sportsbook": "draftkings",
                "home_team": candidate.team,
                "away_team": "OPP",
                "game_start_utc": "2026-08-06T23:00:00Z",
                "canonical_selected": True,
                "validation_status": "VALID",
                "observed_at_utc": "2026-08-06T14:00:00Z",
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
    ).to_csv(observations, index=False)

    plays = parlay_selector.build_alternate_line_plays(candidates, observations)
    ticket = parlay_selector.select_profit_boost_ticket(plays)

    assert len(plays) == 2
    assert all(play["line_variant"] == "alternate" for play in plays)
    assert all(play["base_market_line"] == 0.5 for play in plays)
    assert all(play["market_line"] == 1.5 for play in plays)
    assert all(play["selected_side_price"] == 200 for play in plays)
    assert all(play["alternate_line_books"] == 1 for play in plays)
    assert all(play["estimated_graded_hit_rate"] < 0.72 for play in plays)
    assert ticket is not None
    assert ticket["ticket_tier"] == "profit_boost"
    assert ticket["evidence_status"] == "SHADOW_ALT_LINE_PRICE_CAPTURE"
    assert ticket["combined_decimal_price"] == 9.0

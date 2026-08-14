from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import select_daily_parlay as parlay_selector


def _candidate(
    *,
    player: str,
    game_id: str,
    probability: float,
    price: float,
    direction: str = "OVER",
    target: str = "H",
) -> SimpleNamespace:
    return SimpleNamespace(
        player=player,
        player_id=player.lower().replace(" ", "_"),
        team=player[:3].upper(),
        game_id=game_id,
        target=target,
        direction=direction,
        prediction=1.2,
        market_line=0.5,
        market_source="real",
        market_bucket=f"{target}|OVER|0.5",
        historical_bucket_key=f"{target}|OVER|0.5",
        calibrated_graded_hit_rate=probability,
        model_hit_probability=probability,
        selection_score=0.85,
        expected_value_per_unit=0.08,
        market_implied_probability=0.67,
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


def test_consistency_ticket_can_accept_safer_sub_even_money_two_leg_price() -> None:
    candidates = [
        _candidate(player="One", game_id="g1", probability=0.75, price=-250),
        _candidate(player="Two", game_id="g2", probability=0.75, price=-250, target="TB"),
    ]

    ticket, _ = parlay_selector.select_ticket(
        candidates,
        min_legs=2,
        max_legs=4,
        min_leg_probability=0.62,
        min_ticket_probability=0.40,
        min_combined_decimal_price=parlay_selector.MIN_COMBINED_DECIMAL_PRICE,
        min_expected_return=0.0,
    )

    assert ticket is not None
    assert ticket["leg_count"] == 2
    assert ticket["combined_decimal_price"] < 2.0
    assert ticket["reliability_profile"]["status"] == "pass"


def test_longer_tickets_are_withheld_when_sweep_probability_is_not_reliability_grade() -> None:
    candidates = [
        _candidate(player="Player 1", game_id="g1", probability=0.65, price=-150, target="H"),
        _candidate(player="Player 2", game_id="g2", probability=0.65, price=-150, target="TB"),
        _candidate(player="Player 3", game_id="g3", probability=0.65, price=-150, target="RBI"),
        _candidate(player="Player 4", game_id="g4", probability=0.65, price=-150, target="H"),
    ]

    ladder, _ = parlay_selector.select_ticket_ladder(
        candidates,
        min_legs=2,
        max_legs=4,
        min_leg_probability=0.62,
        base_min_ticket_probability=0.40,
        min_combined_decimal_price=parlay_selector.MIN_COMBINED_DECIMAL_PRICE,
        min_expected_return=0.0,
    )

    assert [ticket["leg_count"] for ticket in ladder] == [2]
    assert ladder[0]["reliability_profile"]["status"] == "pass"


def test_anchor_filter_is_over_only_and_requires_playable_market_support() -> None:
    eligible = _candidate(player="Eligible", game_id="g1", probability=0.68, price=-180)
    under = _candidate(player="Under", game_id="g2", probability=0.75, price=-180, direction="UNDER")
    thin = _candidate(player="Thin", game_id="g3", probability=0.70, price=-180)
    thin.market_books = 0
    thin.market_common_books = 0

    kept, rejected = parlay_selector.filter_anchor_candidates(
        [eligible, under, thin],
        min_leg_probability=0.62,
    )

    assert kept == [eligible]
    assert rejected["not_over"] == 1
    assert rejected["insufficient_book_coverage"] == 1


def test_ticket_ladder_keeps_best_two_leg_ticket_and_adds_longer_options() -> None:
    candidates = [
        _candidate(player="Player 1", game_id="g1", probability=0.75, price=-150, target="H"),
        _candidate(player="Player 2", game_id="g2", probability=0.75, price=-150, target="TB"),
        _candidate(player="Player 3", game_id="g3", probability=0.75, price=-150, target="RBI"),
        _candidate(player="Player 4", game_id="g4", probability=0.75, price=-150, target="H"),
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
                "source": "fanduel_public",
                "source_market_id": f"alt-{index}",
                "player_name": candidate.player,
                "market_type": "batter_hits",
                "side": "over",
                "line": 1.5,
                "price_american": 200,
                "sportsbook": "fanduel",
                "sportsbook_deeplink": (
                    f"https://sportsbook.fanduel.com/addToBetslip?marketId=42.20{index}&selectionId=30{index}"
                ),
                "home_team": candidate.team,
                "away_team": "OPP",
                "game_start_utc": "2026-08-06T23:00:00Z",
                "canonical_selected": False,
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


def test_fanduel_betslip_uses_only_provider_issued_selection_ids() -> None:
    legs = [
        {
            "selected_sportsbook_key": "fanduel",
            "sportsbook_deeplink": (
                "https://sportsbook.fanduel.com/addToBetslip?marketId=42.581005148&selectionId=237471"
            ),
        },
        {
            "selected_sportsbook_key": "fanduel",
            "sportsbook_deeplink": (
                "https://sportsbook.fanduel.com/addToBetslip?marketId=42.581005149&selectionId=237472"
            ),
        },
    ]

    url = parlay_selector.build_fanduel_betslip_url(legs)

    assert url is not None
    assert url.startswith("https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?")
    assert "marketId%5B0%5D=42.581005148" in url
    assert "selectionId%5B1%5D=237472" in url
    legs[1]["sportsbook_deeplink"] = "https://evil.example/addToBetslip?marketId=42.1&selectionId=2"
    assert parlay_selector.build_fanduel_betslip_url(legs) is None


def test_main_line_plays_are_repriced_to_exact_linked_fanduel_quotes(tmp_path: Path) -> None:
    candidates = [
        _candidate(player="First Player", game_id="g1", probability=0.72, price=-180),
        _candidate(player="Second Player", game_id="g2", probability=0.70, price=-170),
    ]
    observations = tmp_path / "provider.csv"
    pd.DataFrame(
        [
            {
                "source": "fanduel_public",
                "source_market_id": f"main-{index}",
                "player_name": candidate.player,
                "market_type": "batter_hits",
                "side": "over",
                "line": 0.5,
                "price_american": -160 - index,
                "sportsbook": "fanduel",
                "sportsbook_deeplink": (
                    f"https://sportsbook.fanduel.com/addToBetslip?marketId=42.10{index}&selectionId=20{index}"
                ),
                "home_team": candidate.team,
                "away_team": "OPP",
                "game_start_utc": "2026-08-06T23:00:00Z",
                "canonical_selected": False,
                "validation_status": "VALID",
                "observed_at_utc": "2026-08-06T14:00:00Z",
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
    ).to_csv(observations, index=False)

    plays = parlay_selector.build_fanduel_main_line_plays(candidates, observations)
    ladder, _ = parlay_selector.select_ticket_ladder(
        candidates,
        min_legs=2,
        max_legs=2,
        min_leg_probability=0.62,
        base_min_ticket_probability=0.40,
        min_combined_decimal_price=2.0,
        min_expected_return=0.0,
        plays_override=plays,
    )
    parlay_selector.attach_fanduel_betslip(ladder[0])

    assert len(plays) == 2
    assert all(play["selected_sportsbook_key"] == "fanduel" for play in plays)
    assert all(play["provider_source_market_id"].startswith("main-") for play in plays)
    assert ladder[0]["betslip"]["status"] == "ready"
    assert ladder[0]["betslip"]["leg_count"] == 2


def test_hit_survival_gate_uses_confirmed_role_and_conservative_consensus() -> None:
    candidate = _candidate(player="Reliable Hitter", game_id="g1", probability=0.68, price=-180)

    class Bundle:
        latest_context = {"reliablehitter": {"last_hits": 2.0, "recent_batting_order": 5.0}}

        @staticmethod
        def predict(features: dict[str, float]) -> tuple[float, float]:
            assert features["batting_order"] == 2.0
            return 0.70, 0.69

    kept, rejected = parlay_selector.apply_hit_survival_gate(
        [candidate],
        bundle=Bundle(),
        official_contexts={
            "g1": {
                "batting_orders": {candidate.team: {"reliable hitter": 2}},
            }
        },
    )

    assert kept == [candidate]
    assert not rejected
    assert candidate.raw["Hit_Survival_Batting_Order_Source"] == "confirmed_lineup"
    assert candidate.raw["Parlay_Leg_Probability"] >= 0.62
    assert parlay_selector._candidate_probability(candidate) == candidate.raw["Parlay_Leg_Probability"]


def test_latent_set_profile_is_order_invariant_and_penalizes_concentration() -> None:
    hit = {
        "target": "H",
        "parlay_leg_probability": 0.68,
        "latent_probability_disagreement": 0.04,
        "hit_survival_batting_order_source": "confirmed_lineup",
    }
    other_hit = {**hit, "parlay_leg_probability": 0.66, "hit_survival_batting_order_source": "prior_start_proxy"}
    total_bases = {**other_hit, "target": "TB"}

    concentrated = parlay_selector._latent_set_profile([hit, other_hit], 0.44)
    reversed_profile = parlay_selector._latent_set_profile([other_hit, hit], 0.44)
    diversified = parlay_selector._latent_set_profile([hit, total_bases], 0.44)

    assert concentrated == reversed_profile
    assert concentrated["representation"] == "permutation_invariant_leg_aggregate_v1"
    assert concentrated["set_consistency_score"] < diversified["set_consistency_score"]

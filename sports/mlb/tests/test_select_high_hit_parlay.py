from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MLB_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import select_high_hit_parlay as high_hit  # noqa: E402


def _leg(name, *, game_id, safe_probability, american_price=-110.0, target="TB", direction="OVER",
         market_line=1.5, price_confirmed=True, player_id=None):
    return SimpleNamespace(
        player=name,
        player_id=player_id or name.lower().replace(" ", "_"),
        team=f"TEAM_{name}",
        game_id=game_id,
        target=target,
        direction=direction,
        market_line=market_line,
        safe_probability=safe_probability,
        calibrated_hit_probability=safe_probability,
        selected_side_price=american_price,
        selected_sportsbook="FanDuel",
        price_confirmed=price_confirmed,
        winner_signature_model_status="disabled",
    )


def test_eligible_legs_excludes_low_probability_unconfirmed_and_price_traps() -> None:
    good = _leg("Good", game_id="1", safe_probability=0.80)
    low_probability = _leg("Low", game_id="2", safe_probability=0.60)
    unconfirmed = _leg("Unconfirmed", game_id="3", safe_probability=0.85, price_confirmed=False)
    ultra_short = _leg("UltraShort", game_id="4", safe_probability=0.95, american_price=-600)

    legs = high_hit.eligible_legs([good, low_probability, unconfirmed, ultra_short])

    assert legs == [good]


def test_build_combos_only_builds_cross_game_combinations() -> None:
    a = _leg("A", game_id="1", safe_probability=0.80)
    b = _leg("B", game_id="1", safe_probability=0.80)
    c = _leg("C", game_id="2", safe_probability=0.80)

    combos = high_hit.build_combos([a, b, c], joint_probability_floor=0.0)

    pairs = [{candidate.player for candidate in combo["legs"]} for combo in combos]
    assert {"A", "B"} not in pairs
    assert {"A", "C"} in pairs
    assert {"B", "C"} in pairs


def test_build_combos_computes_real_joint_probability_price_and_ev() -> None:
    a = _leg("A", game_id="1", safe_probability=0.80, american_price=-110.0)
    b = _leg("B", game_id="2", safe_probability=0.75, american_price=120.0)

    combos = high_hit.build_combos([a, b], joint_probability_floor=0.0)

    assert len(combos) == 1
    combo = combos[0]
    assert combo["joint_probability"] == pytest.approx(0.80 * 0.75)
    decimal_a = 1.0 + (100.0 / 110.0)
    decimal_b = 2.2
    assert combo["decimal_price"] == pytest.approx(decimal_a * decimal_b)
    assert combo["expected_value_per_unit"] == pytest.approx(combo["joint_probability"] * combo["decimal_price"] - 1.0)


def test_build_combos_respects_joint_probability_floor() -> None:
    a = _leg("A", game_id="1", safe_probability=0.70)
    b = _leg("B", game_id="2", safe_probability=0.70)
    assert high_hit.build_combos([a, b], joint_probability_floor=0.50) == []


def test_build_combos_requires_meaningful_combined_payout() -> None:
    a = _leg("A", game_id="1", safe_probability=0.90, american_price=-400)
    b = _leg("B", game_id="2", safe_probability=0.90, american_price=-400)
    # 1.25 * 1.25 = 1.5625, despite an 81% model joint probability.
    assert high_hit.build_combos([a, b]) == []


def test_select_high_hit_parlays_ranks_ev_after_probability_and_price_gates() -> None:
    # Higher-probability A+B clears all gates, but C+D has better expected
    # return at still-safe probabilities. V2 should choose C+D first.
    a = _leg("A", game_id="1", safe_probability=0.90, american_price=-180)
    b = _leg("B", game_id="2", safe_probability=0.90, american_price=-180)
    c = _leg("C", game_id="3", safe_probability=0.75, american_price=100)
    d = _leg("D", game_id="4", safe_probability=0.75, american_price=100)

    selected = high_hit.select_high_hit_parlays([a, b, c, d], max_published=5)

    assert len(selected) == 2
    assert {candidate.player for candidate in selected[0]["legs"]} == {"C", "D"}
    assert selected[0]["joint_probability"] < selected[1]["joint_probability"]
    assert selected[0]["expected_value_per_unit"] > selected[1]["expected_value_per_unit"]

    seen: set[str] = set()
    for combo in selected:
        players = {candidate.player for candidate in combo["legs"]}
        assert not (players & seen)
        seen |= players


def test_policy_never_expands_to_three_legs() -> None:
    candidates = [
        _leg("A", game_id="1", safe_probability=0.85, american_price=-110),
        _leg("B", game_id="2", safe_probability=0.85, american_price=-110),
        _leg("C", game_id="3", safe_probability=0.85, american_price=-110),
    ]
    combos = high_hit.build_combos(candidates, max_legs=3)
    assert combos
    assert all(combo["leg_count"] == 2 for combo in combos)


def test_select_high_hit_parlays_returns_nothing_when_no_combo_clears_floor() -> None:
    a = _leg("A", game_id="1", safe_probability=0.71)
    b = _leg("B", game_id="2", safe_probability=0.71)
    assert high_hit.select_high_hit_parlays([a, b], joint_probability_floor=0.99) == []

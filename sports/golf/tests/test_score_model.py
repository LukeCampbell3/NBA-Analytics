from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLF_PREDICTIONS_ROOT = REPO_ROOT / "sports" / "golf" / "predictions"
sys.path.insert(0, str(GOLF_PREDICTIONS_ROOT))

import score_model as model  # noqa: E402


def _event(rounds_by_player: dict[str, list[float]]) -> dict:
    players = []
    for player_id, round_strokes in rounds_by_player.items():
        players.append(
            {
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "rounds": [{"round": i + 1, "strokes": s} for i, s in enumerate(round_strokes)],
            }
        )
    return {"players": players}


def test_build_recent_form_computes_field_relative_differential() -> None:
    # Two players, one event, one round each: field average is 70.0.
    # Player A shot 68 (2 better than field), player B shot 72 (2 worse).
    event = _event({"A": [68.0], "B": [72.0]})
    forms = model.build_recent_form([event])
    assert forms["A"].mean_differential == -2.0
    assert forms["B"].mean_differential == 2.0


def test_build_recent_form_aggregates_across_multiple_events() -> None:
    event1 = _event({"A": [68.0], "B": [72.0]})  # field avg 70
    event2 = _event({"A": [70.0], "B": [70.0]})  # field avg 70, A now at field average
    forms = model.build_recent_form([event1, event2])
    # A: round diffs [-2.0, 0.0] -> mean -1.0
    assert forms["A"].rounds_observed == 2
    assert forms["A"].mean_differential == -1.0


def test_project_field_uses_league_average_for_players_with_no_real_form() -> None:
    event = _event({"A": [65.0], "B": [65.0], "C": [65.0]})  # field avg 65, all players -6 vs a 71 par... just build forms
    forms = model.build_recent_form([event])
    field = [{"player_id": "NEW_PLAYER", "player_name": "No History"}]
    projections = model.project_field(field, forms, scheduled_rounds=4, round_par=71.0)
    assert len(projections) == 1
    proj = projections[0]
    assert proj.form_rounds_observed == 0
    # No real form -> falls back to the real field-wide average differential (0.0 here since every
    # training-event player shot exactly the field average), never a fabricated player-specific number.
    assert proj.projected_round_score == 71.0


def test_project_field_uses_players_own_real_form_when_available() -> None:
    events = [_event({"A": [68.0, 67.0, 69.0]})]  # field of 1 -> differential always 0, not useful in isolation
    # Build a field-relative signal across two players instead.
    events = [_event({"A": [68.0, 67.0, 69.0], "B": [72.0, 73.0, 71.0]})]
    forms = model.build_recent_form(events)
    field = [{"player_id": "A", "player_name": "Player A"}]
    projections = model.project_field(field, forms, scheduled_rounds=4, round_par=71.0)
    proj = projections[0]
    assert proj.form_rounds_observed == 3
    assert proj.projected_round_score < 71.0  # A is real-form better than the field


def test_simulate_tournament_win_probabilities_sum_to_one() -> None:
    projections = [
        model.PlayerProjection("A", "A", "", projected_round_score=69.0, projected_total_score=276.0, round_std=2.0, form_rounds_observed=10),
        model.PlayerProjection("B", "B", "", projected_round_score=71.0, projected_total_score=284.0, round_std=2.0, form_rounds_observed=10),
        model.PlayerProjection("C", "C", "", projected_round_score=72.0, projected_total_score=288.0, round_std=2.0, form_rounds_observed=10),
    ]
    results = model.simulate_tournament(projections, scheduled_rounds=4, has_cut=False, num_simulations=2000, random_seed=1)
    total_win = sum(r.win_probability for r in results)
    assert abs(total_win - 1.0) < 1e-9


def test_simulate_tournament_better_projection_wins_more_often() -> None:
    projections = [
        model.PlayerProjection("A", "A", "", projected_round_score=68.0, projected_total_score=272.0, round_std=2.0, form_rounds_observed=10),
        model.PlayerProjection("B", "B", "", projected_round_score=74.0, projected_total_score=296.0, round_std=2.0, form_rounds_observed=10),
    ]
    results = {r.player_id: r for r in model.simulate_tournament(projections, scheduled_rounds=4, has_cut=False, num_simulations=2000, random_seed=1)}
    assert results["A"].win_probability > results["B"].win_probability


def test_simulate_tournament_applies_a_real_cut_line() -> None:
    """A clearly-worse player in a small cut field must miss the cut in
    (almost) every simulation -- proves the cut logic actually prunes the
    field after round 2, not just decoration."""
    good = model.PlayerProjection("A", "A", "", projected_round_score=68.0, projected_total_score=272.0, round_std=1.0, form_rounds_observed=10)
    bad = model.PlayerProjection("B", "B", "", projected_round_score=80.0, projected_total_score=320.0, round_std=1.0, form_rounds_observed=10)
    results = {
        r.player_id: r
        for r in model.simulate_tournament([good, bad], scheduled_rounds=4, has_cut=True, cut_after_round=2, cut_size=1, num_simulations=1000, random_seed=1)
    }
    assert results["A"].make_cut_probability > 0.99
    assert results["B"].make_cut_probability < 0.01


def test_simulate_tournament_no_cut_event_reports_none_for_make_cut() -> None:
    projections = [
        model.PlayerProjection("A", "A", "", projected_round_score=69.0, projected_total_score=276.0, round_std=2.0, form_rounds_observed=10),
    ]
    results = model.simulate_tournament(projections, scheduled_rounds=4, has_cut=False, num_simulations=100, random_seed=1)
    assert results[0].make_cut_probability is None


def test_simulate_tournament_handles_empty_field() -> None:
    assert model.simulate_tournament([], num_simulations=100) == []

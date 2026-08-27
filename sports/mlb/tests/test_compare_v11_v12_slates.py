from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

MLB_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import compare_v11_v12_slates as comparison  # noqa: E402


def _pick(**overrides) -> SimpleNamespace:
    base = dict(
        player="Real Player",
        game_id="824970",
        target="TB",
        direction="UNDER",
        market_line=1.5,
        run_date=date(2026, 8, 10),
        selected_side_price=-176.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_grade_candidate_grades_a_real_win_and_a_real_loss() -> None:
    win_candidate = _pick()  # UNDER 1.5, actual 1.0 -> win
    loss_candidate = _pick()  # UNDER 1.5, actual 3.0 -> loss
    lookup = {("2026-08-10", "real_player", "TB", "824970"): 1.0}

    assert comparison.grade_candidate(win_candidate, lookup) == "win"

    lookup[("2026-08-10", "real_player", "TB", "824970")] = 3.0
    assert comparison.grade_candidate(loss_candidate, lookup) == "loss"


def test_grade_candidate_excludes_pushes_and_unsettled_rows() -> None:
    push_candidate = _pick(market_line=1.0)  # UNDER 1.0, actual 1.0 -> push
    lookup = {("2026-08-10", "real_player", "TB", "824970"): 1.0}
    assert comparison.grade_candidate(push_candidate, lookup) is None

    unsettled = _pick()
    assert comparison.grade_candidate(unsettled, {}) is None


def test_slate_metrics_computes_real_hit_rate_and_roi() -> None:
    win = _pick(selected_side_price=-176.0)  # profit = 100/176
    loss = _pick(game_id="999", selected_side_price=-176.0)
    lookup = {
        ("2026-08-10", "real_player", "TB", "824970"): 1.0,  # win vs UNDER 1.5
        ("2026-08-10", "real_player", "TB", "999"): 3.0,  # loss vs UNDER 1.5
    }

    metrics = comparison.slate_metrics([win, loss], lookup)

    assert metrics["picks"] == 2
    assert metrics["settled"] == 2
    assert metrics["wins"] == 1
    assert metrics["hit_rate"] == 0.5
    win_profit = 100.0 / 176.0
    assert metrics["roi"] == (win_profit + (-1.0)) / 2.0


def test_slate_metrics_reports_none_metrics_when_nothing_settles() -> None:
    unsettled = _pick()
    metrics = comparison.slate_metrics([unsettled], {})

    assert metrics == {"picks": 1, "settled": 0, "wins": 0, "hit_rate": None, "roi": None}


def test_gate_pass_requires_strictly_better_roi_and_a_bounded_hit_rate_shortfall() -> None:
    # v12 beats v11's ROI and matches its hit rate exactly -> passes.
    assert comparison.gate_pass(roi_v11=0.10, hit_v11=0.75, roi_v12=0.15, hit_v12=0.75, margin=0.05) is True
    # v12 trails hit rate by exactly the margin -> still passes (>=).
    assert comparison.gate_pass(roi_v11=0.10, hit_v11=0.75, roi_v12=0.15, hit_v12=0.70, margin=0.05) is True
    # v12 trails hit rate by more than the margin -> fails even with better ROI.
    assert comparison.gate_pass(roi_v11=0.10, hit_v11=0.75, roi_v12=0.15, hit_v12=0.69, margin=0.05) is False
    # Equal ROI is not "strictly better" -> fails.
    assert comparison.gate_pass(roi_v11=0.10, hit_v11=0.75, roi_v12=0.10, hit_v12=0.80, margin=0.05) is False
    # Missing real metrics on either side -> fails closed, never guesses.
    assert comparison.gate_pass(roi_v11=None, hit_v11=0.75, roi_v12=0.15, hit_v12=0.80, margin=0.05) is False

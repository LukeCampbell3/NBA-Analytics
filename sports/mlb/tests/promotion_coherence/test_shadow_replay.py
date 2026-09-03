"""Tests for the shadow replay: settlement-index parsing, per-slate
grading, and aggregate divergence counts. All fixtures are constructed
inline so the tests never depend on future settlement additions or
production payload rewrites."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence import default_thresholds
from sports.mlb.parlay_v2.promotion_coherence.shadow_replay import (
    _aggregate,
    _grade_selected,
    _load_settlements_index,
    build_row,
    run_shadow_report,
)
from sports.mlb.parlay_v2.promotion_coherence.promotion_confidence import (
    decide_coherent_promotion,
)


def _write_payload(root: Path, sub_path: str, payload: dict) -> Path:
    path = root / sub_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _write_settlements(root: Path, settlements: list[dict]) -> Path:
    path = root / "sports/mlb/data/predictions/unified/historical_settlements.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "settlements": settlements}))
    return path


def _make_payload(
    *,
    date: str,
    live_action: str,
    overlay_action: str,
    joint: float,
    price: float,
    legs: list[float],
    selected_legs: list[dict],
) -> dict:
    return {
        "run_date": date,
        "parlays": {
            "eligible": True,
            "action": live_action,
            "public_quality_overlay": {
                "action": overlay_action,
                "joint_probability": joint,
                "combined_decimal_price": price,
                "leg_probabilities": legs,
                "probability_edge": 0.02,
                "expected_value_per_unit": 0.1,
            },
            "selected_parlay": {
                "candidate_id": f"{date}|test",
                **{f"leg_{i + 1}": leg for i, leg in enumerate(selected_legs)},
            },
        },
    }


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


def test_load_settlements_index_reads_alias_market_and_id(tmp_root: Path) -> None:
    _write_settlements(tmp_root, [
        {"event_date": "2026-08-30", "player_id": "cal_raleigh", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
        {"event_date": "2026-08-30", "player_id": "shohei_ohtani", "market": "K",  # K aliases to SO
         "side": "OVER", "line": 5.5, "settlement": "lost"},
    ])
    index = _load_settlements_index(
        tmp_root / "sports/mlb/data/predictions/unified/historical_settlements.json"
    )
    assert index[("2026-08-30", "cal_raleigh", "H", "OVER", 0.5)] == "won"
    assert index[("2026-08-30", "shohei_ohtani", "SO", "OVER", 5.5)] == "lost"


def test_missing_settlements_file_returns_empty_index(tmp_root: Path) -> None:
    index = _load_settlements_index(tmp_root / "nonexistent.json")
    assert index == {}


def test_grade_selected_wins_all_legs(tmp_root: Path) -> None:
    payload = _make_payload(
        date="2026-08-30", live_action="ACT", overlay_action="ACT",
        joint=0.6, price=3.0, legs=[0.75, 0.8],
        selected_legs=[
            {"player": "Cal Raleigh", "player_id": "cal_raleigh",
             "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 1.6},
            {"player": "Austin Riley", "player_id": "austin_riley",
             "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 1.7},
        ],
    )
    _write_settlements(tmp_root, [
        {"event_date": "2026-08-30", "player_id": "cal_raleigh", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
        {"event_date": "2026-08-30", "player_id": "austin_riley", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
    ])
    idx = _load_settlements_index(
        tmp_root / "sports/mlb/data/predictions/unified/historical_settlements.json"
    )
    grade = _grade_selected(payload, idx)
    assert grade["parlay_result"] == "won"
    # combined price 1.6 * 1.7 = 2.72 -> return 1.72
    assert grade["realized_return_per_unit"] == pytest.approx(1.72)


def test_grade_selected_any_loss_loses_parlay(tmp_root: Path) -> None:
    payload = _make_payload(
        date="2026-08-30", live_action="ACT", overlay_action="ACT",
        joint=0.6, price=3.0, legs=[0.75, 0.8],
        selected_legs=[
            {"player": "Cal Raleigh", "player_id": "cal_raleigh",
             "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 1.6},
            {"player": "Caleb Durbin", "player_id": "caleb_durbin",
             "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 2.0},
        ],
    )
    _write_settlements(tmp_root, [
        {"event_date": "2026-08-30", "player_id": "cal_raleigh", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
        {"event_date": "2026-08-30", "player_id": "caleb_durbin", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "lost"},
    ])
    idx = _load_settlements_index(
        tmp_root / "sports/mlb/data/predictions/unified/historical_settlements.json"
    )
    grade = _grade_selected(payload, idx)
    assert grade["parlay_result"] == "lost"
    assert grade["realized_return_per_unit"] == pytest.approx(-1.0)


def test_grade_selected_unknown_when_any_leg_ungraded(tmp_root: Path) -> None:
    payload = _make_payload(
        date="2026-09-02", live_action="ACT", overlay_action="ABSTAIN",
        joint=0.28, price=5.67, legs=[0.5, 0.55],
        selected_legs=[
            {"player": "Alex Bregman", "player_id": "alex_bregman",
             "target": "TB", "side": "OVER", "line": 1.5, "decimal_price": 2.7},
            {"player": "Shohei Ohtani", "player_id": "shohei_ohtani",
             "target": "TB", "side": "OVER", "line": 1.5, "decimal_price": 2.1},
        ],
    )
    idx = {}
    grade = _grade_selected(payload, idx)
    assert grade["parlay_result"] == "unknown"
    assert grade["realized_return_per_unit"] is None


def test_run_shadow_report_counts_divergences_and_aggregates_returns(tmp_root: Path) -> None:
    # Slate A: live ACT + coherent ACT + settled WIN
    _write_payload(tmp_root, "sports/mlb/web/data/history/runs/2026-08-30/run-a/daily_predictions.json",
                   _make_payload(
                       date="2026-08-30", live_action="ACT", overlay_action="ACT",
                       joint=0.60, price=2.5, legs=[0.75, 0.80],
                       selected_legs=[
                           {"player": "Cal Raleigh", "player_id": "cal_raleigh",
                            "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 1.5},
                           {"player": "Austin Riley", "player_id": "austin_riley",
                            "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 1.7},
                       ],
                   ))
    # Slate B: live ACT + coherent ABSTAIN + settled LOSS -- the case
    # this project exists to catch. Live loses 1.0/unit, coherent avoids
    # publishing at all.
    _write_payload(tmp_root, "sports/mlb/web/data/history/runs/2026-08-30/run-b/daily_predictions.json",
                   _make_payload(
                       date="2026-08-30", live_action="ACT", overlay_action="ABSTAIN",
                       joint=0.28, price=5.67, legs=[0.50, 0.55],
                       selected_legs=[
                           {"player": "Caleb Durbin", "player_id": "caleb_durbin",
                            "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 2.0},
                           {"player": "Spencer Horwitz", "player_id": "spencer_horwitz",
                            "target": "H", "side": "OVER", "line": 0.5, "decimal_price": 2.0},
                       ],
                   ))
    _write_settlements(tmp_root, [
        {"event_date": "2026-08-30", "player_id": "cal_raleigh", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
        {"event_date": "2026-08-30", "player_id": "austin_riley", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "won"},
        {"event_date": "2026-08-30", "player_id": "caleb_durbin", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "lost"},
        {"event_date": "2026-08-30", "player_id": "spencer_horwitz", "market": "H",
         "side": "OVER", "line": 0.5, "settlement": "lost"},
    ])

    report = run_shadow_report(
        repo_root=tmp_root,
        history_glob="sports/mlb/web/data/history/runs/*/*/daily_predictions.json",
        live_payload_rel="__no_live__.json",
    )
    s = report.summary
    assert s["total_payloads"] == 2
    assert s["live_act"] == 2
    assert s["coherent_act"] == 1
    assert s["divergent_live_act_coherent_abstain"] == 1
    assert s["divergent_live_abstain_coherent_act"] == 0
    assert s["concurrent_act"] == 1
    assert s["graded_parlays"] == 2
    # live: +1.55 (1.5*1.7=2.55 -> +1.55) + -1.0 = +0.55
    assert s["live_realized_return_per_unit_sum"] == pytest.approx(1.55 - 1.0)
    # coherent: only the WIN would have been published -- +1.55
    assert s["coherent_realized_return_per_unit_sum"] == pytest.approx(1.55)


def test_build_row_preserves_components_and_grading() -> None:
    payload = {
        "run_date": "2026-08-30",
        "parlays": {
            "eligible": True, "action": "ACT",
            "public_quality_overlay": {
                "action": "ACT",
                "joint_probability": 0.6, "combined_decimal_price": 3.0,
                "leg_probabilities": [0.75, 0.8],
            },
            "selected_parlay": {"candidate_id": "row-test"},
        },
    }
    decision = decide_coherent_promotion(payload, thresholds=default_thresholds())
    row = build_row(Path("/tmp/x.json"), payload, decision, {"parlay_result": "unknown"})
    assert row.candidate_id == "row-test"
    assert row.live_action == "ACT" and row.coherent_action == "ACT"
    assert row.joint_probability == pytest.approx(0.6)


def test_aggregate_handles_empty_input() -> None:
    s = _aggregate([])
    assert s["total_payloads"] == 0
    assert s["live_realized_return_per_unit_sum"] == 0.0
    assert s["coherent_realized_return_per_unit_sum"] == 0.0

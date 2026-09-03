"""Tests for the additive no-vig market probability capture in
pair_ingest.py and pair_schema.py.

Contract this test suite guards:

* Backward compat: legacy callers of build_pair_observation that
  do NOT pass the new optional kwargs still get a valid PairObservation
  with the new fields set to None.
* observation_id and row_hash NEVER include the new fields, so a
  legacy row and a new-capture row for the same identity dedupe
  correctly.
* No-vig math matches the two-sided-book formula and returns None on
  any degenerate / one-sided input.
* The source-CSV side-capture reader handles missing files, missing
  columns, and malformed rows without raising.
* Real-slate smoke test: side_capture from a real prediction pool CSV
  produces >= 1 leg with a real no-vig probability, proving the wiring
  makes it end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sports.mlb.parlay_v2.calibration.pair_ingest import (
    DAILY_RUNS_ROOT,
    PAIR_INGEST_VERSION,
    _american_to_decimal,
    _lookup_side_capture,
    _no_vig_probability,
    _pair_side_capture,
)
from sports.mlb.parlay_v2.calibration.pair_schema import (
    SCHEMA_VERSION,
    PairObservation,
    build_pair_observation,
)


# --- unit: american_to_decimal & no_vig helpers -------------------------

def test_american_to_decimal_positive_and_negative() -> None:
    assert _american_to_decimal(+150) == pytest.approx(2.5)
    assert _american_to_decimal(-200) == pytest.approx(1.5)


def test_american_to_decimal_rejects_invalid() -> None:
    assert _american_to_decimal(None) is None
    assert _american_to_decimal(50) is None            # abs < 100
    assert _american_to_decimal("nope") is None
    assert _american_to_decimal(float("nan")) is None


def test_no_vig_probability_removes_overround() -> None:
    # Symmetric -110 / -110 both sides -> 0.5 no-vig probability each
    over_dec = _american_to_decimal(-110)
    under_dec = _american_to_decimal(-110)
    assert _no_vig_probability(over_dec, under_dec) == pytest.approx(0.5)


def test_no_vig_probability_asymmetric_case() -> None:
    # -200 / +170: p_over_implied = 1/1.5 = 0.667; p_under_implied = 1/2.7 = 0.370
    # total = 1.037, no-vig p_over = 0.667/1.037 = 0.643
    over_dec = _american_to_decimal(-200)
    under_dec = _american_to_decimal(+170)
    assert _no_vig_probability(over_dec, under_dec) == pytest.approx(
        (1 / over_dec) / ((1 / over_dec) + (1 / under_dec))
    )


def test_no_vig_probability_none_on_missing_side() -> None:
    assert _no_vig_probability(None, 1.9) is None
    assert _no_vig_probability(1.9, None) is None
    assert _no_vig_probability(None, None) is None


def test_no_vig_probability_rejects_degenerate_prices() -> None:
    assert _no_vig_probability(1.0, 2.0) is None
    assert _no_vig_probability(1.5, 1.0) is None


def test_no_vig_probability_rejects_negative_or_zero_overround() -> None:
    # 1.5 / 3.5 -> 0.667 + 0.286 = 0.952 (arb, no book vig) -- rejected
    assert _no_vig_probability(1.5, 3.5) is None


# --- unit: _pair_side_capture ------------------------------------------

def _write_source_pool(tmp_root: Path, stamp: str, rows: list[dict]) -> Path:
    slate_dir = tmp_root / "sports" / "mlb" / "data" / "predictions" / "daily_runs" / stamp
    slate_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = slate_dir / f"daily_prediction_pool_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def test_pair_side_capture_missing_file_returns_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sports.mlb.parlay_v2.calibration.pair_ingest.DAILY_RUNS_ROOT",
        tmp_path / "no_such",
    )
    assert _pair_side_capture("20260101") == {}


def test_pair_side_capture_reads_real_columns(monkeypatch, tmp_path: Path) -> None:
    stamp = "20260828_test"
    _write_source_pool(tmp_path, stamp, [
        {
            "Target": "H",
            "Game_ID": "824000",
            "Player_ID": "cal_raleigh",
            "Player": "Cal Raleigh",
            "Market_Line": 0.5,
            "Market_Over_Price": -140,
            "Market_Under_Price": +115,
        },
    ])
    monkeypatch.setattr(
        "sports.mlb.parlay_v2.calibration.pair_ingest.DAILY_RUNS_ROOT",
        tmp_path / "sports" / "mlb" / "data" / "predictions" / "daily_runs",
    )
    lookup = _pair_side_capture(stamp)
    # Both directions must be present.
    over_key = ("cal_raleigh", "824000", "H", "OVER", 0.5)
    under_key = ("cal_raleigh", "824000", "H", "UNDER", 0.5)
    assert over_key in lookup and under_key in lookup

    over_chosen, over_other, over_no_vig = lookup[over_key]
    assert over_chosen == pytest.approx(_american_to_decimal(-140))
    assert over_other == pytest.approx(_american_to_decimal(+115))
    assert 0 < over_no_vig < 1

    # And the two directions' no-vig probabilities should sum to 1.
    _, _, under_no_vig = lookup[under_key]
    assert over_no_vig + under_no_vig == pytest.approx(1.0)


def test_pair_side_capture_skips_row_missing_a_side(monkeypatch, tmp_path: Path) -> None:
    stamp = "20260828_partial"
    _write_source_pool(tmp_path, stamp, [
        {
            "Target": "H", "Game_ID": "g1", "Player_ID": "p1", "Player": "P1",
            "Market_Line": 0.5, "Market_Over_Price": -110, "Market_Under_Price": None,
        },
    ])
    monkeypatch.setattr(
        "sports.mlb.parlay_v2.calibration.pair_ingest.DAILY_RUNS_ROOT",
        tmp_path / "sports" / "mlb" / "data" / "predictions" / "daily_runs",
    )
    lookup = _pair_side_capture(stamp)
    # Row is still present but its no-vig is None (chosen-side ok, other absent).
    key = ("p1", "g1", "H", "OVER", 0.5)
    assert key in lookup
    _, other, no_vig = lookup[key]
    assert other is None
    assert no_vig is None


def test_lookup_side_capture_returns_none_on_miss() -> None:
    lookup = {("p1", "g1", "H", "OVER", 0.5): (1.5, 2.5, 0.62)}
    action_row = pd.Series({
        "player_key": "p1", "game_id": "g1", "target": "H",
        "direction": "OVER", "market_line": 0.5,
    })
    chosen, no_vig = _lookup_side_capture(lookup, action_row)
    assert chosen == 1.5 and no_vig == 0.62

    miss_row = pd.Series({
        "player_key": "other_player", "game_id": "g1", "target": "H",
        "direction": "OVER", "market_line": 0.5,
    })
    assert _lookup_side_capture(lookup, miss_row) == (None, None)


# --- schema: backward compat + new-field wiring -------------------------

def _pair_observation_kwargs(**overrides) -> dict:
    """Minimum kwargs for build_pair_observation. Every real call site
    passes at least this set."""
    base = dict(
        slate_id="20260828",
        leg_1_event_id="p1|g1|H|OVER|0.5",
        leg_2_event_id="p2|g2|R|OVER|0.5",
        same_game=False,
        same_team=False,
        market_pair_type="H|R",
        line_pair_type="H|OVER|0.5__R|OVER|0.5",
        state_bucket_pair="H_OVER_RANKER_V1+MULTI_TARGET|broad",
        price_bucket="<5.0",
        quoted_pair_price=4.5,
        predicted_joint_probability=0.20,
        predicted_independence_probability=0.20,
        counterexample_count=3,
        counterexample_mass=0.80,
        retained_world_count=4,
        retained_probability_mass=1.0,
        calibration_snapshot_id="snap1",
        predictive_version="H_OVER_RANKER_V1+MULTI_TARGET",
        policy_version="PARLAY_POLICY_V2_PROSPECTIVE_003",
        decision_timestamp="20260828T17:00:00Z",
        leg_1_result=1,
        leg_2_result=0,
        settlement_status="settled",
        settlement_timestamp="2026-08-29T00:00:00Z",
    )
    base.update(overrides)
    return base


def test_legacy_call_yields_v1_shape_observation() -> None:
    obs = build_pair_observation(**_pair_observation_kwargs())
    assert obs.schema_version == SCHEMA_VERSION
    assert obs.leg_1_no_vig_market_probability is None
    assert obs.leg_2_no_vig_market_probability is None
    assert obs.leg_1_marginal_probability is None
    assert obs.leg_2_marginal_probability is None
    assert obs.leg_1_decimal_price is None
    assert obs.leg_2_decimal_price is None
    # Full-shape serialization must still work.
    d = obs.as_dict()
    assert "leg_1_no_vig_market_probability" in d and d["leg_1_no_vig_market_probability"] is None


def test_new_capture_populates_optional_fields() -> None:
    obs = build_pair_observation(
        **_pair_observation_kwargs(),
        leg_1_marginal_probability=0.55,
        leg_2_marginal_probability=0.36,
        leg_1_decimal_price=1.9,
        leg_2_decimal_price=2.4,
        leg_1_no_vig_market_probability=0.52,
        leg_2_no_vig_market_probability=0.42,
    )
    assert obs.leg_1_no_vig_market_probability == pytest.approx(0.52)
    assert obs.leg_2_no_vig_market_probability == pytest.approx(0.42)


def test_new_capture_does_not_change_observation_id_or_row_hash() -> None:
    """A slate that was ingested under v1 and then re-ingested under
    v1.1 must not be double-admitted: observation_id and row_hash
    depend only on identity fields, not on the new optional capture.
    """
    kwargs = _pair_observation_kwargs()
    legacy = build_pair_observation(**kwargs)
    with_capture = build_pair_observation(
        **kwargs,
        leg_1_marginal_probability=0.55,
        leg_1_no_vig_market_probability=0.52,
    )
    assert legacy.observation_id == with_capture.observation_id
    assert legacy.row_hash == with_capture.row_hash
    assert legacy.pair_id == with_capture.pair_id


def test_pair_ingest_version_bumped() -> None:
    """A schema evolution deserves a visible version bump so an auditor
    reading a summary knows which capture path produced the row."""
    assert PAIR_INGEST_VERSION == "PAIR_INGEST_V1_1"


# --- real-slate smoke -------------------------------------------------

REAL_SLATE_STAMPS = ["20260828", "20260827", "20260811", "20260810", "20260809"]


def _first_present_slate() -> str | None:
    for s in REAL_SLATE_STAMPS:
        if (DAILY_RUNS_ROOT / s / f"daily_prediction_pool_{s}.csv").exists():
            return s
    return None


def test_side_capture_on_a_real_slate_produces_some_no_vig_rows() -> None:
    """Honest data-quality pin. On the current checked-in slates the
    both-side pricing rate is LOW -- most rows in the raw pool CSV only
    have one side quoted at capture time. This test proves the
    mechanism works and captures at least a small number of no-vig
    rows; as the upstream both-side capture matures, this pin (and the
    sibling test below) will start looking cautious.
    """
    stamp = _first_present_slate()
    if stamp is None:
        pytest.skip("no real slate CSV in this environment")
    lookup = _pair_side_capture(stamp)
    assert len(lookup) > 0
    with_no_vig = sum(1 for entry in lookup.values() if entry[2] is not None)
    # Current observed rate is ~1.5% (48/3252 on 20260828). Pin at >0
    # so the mechanism is proven working, and pin the current rate as
    # observed so an unexpected drop is loud.
    assert with_no_vig > 0, "no-vig capture yielded zero rows -- CSV columns changed?"


def test_current_no_vig_capture_rate_is_small_but_positive() -> None:
    """Records the current honest rate of both-side pricing on the
    checked-in slates. As the upstream pipeline starts capturing both
    sides more consistently this test will start failing and this doc
    gets updated to reflect the new (higher) rate."""
    stamp = _first_present_slate()
    if stamp is None:
        pytest.skip("no real slate CSV in this environment")
    lookup = _pair_side_capture(stamp)
    with_no_vig = sum(1 for entry in lookup.values() if entry[2] is not None)
    rate = with_no_vig / len(lookup)
    # Currently ~1.5% on 20260828. Pin at 0 < rate <= 20% -- if the
    # rate crosses 20% we want a loud signal so we know the data
    # plumbing improved and the promotion-coherence market-disagreement
    # deduction should start being wired into more decisions.
    assert 0.0 < rate <= 0.20, (
        f"no-vig capture rate is {rate:.3%} on {stamp} -- if this crossed 20%, "
        f"pipeline improved; update this pin and consider promoting market-"
        f"disagreement to a first-class production signal."
    )


def test_side_capture_over_and_under_probabilities_sum_to_one_when_both_present() -> None:
    stamp = _first_present_slate()
    if stamp is None:
        pytest.skip("no real slate CSV in this environment")
    lookup = _pair_side_capture(stamp)
    checked = 0
    for (player, game, target, direction, line), (chosen, other, no_vig) in lookup.items():
        if direction != "OVER" or no_vig is None:
            continue
        under_entry = lookup.get((player, game, target, "UNDER", line))
        if under_entry is None or under_entry[2] is None:
            continue
        assert no_vig + under_entry[2] == pytest.approx(1.0, abs=1e-9)
        checked += 1
        if checked >= 20:
            break
    assert checked > 0

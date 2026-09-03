"""Tests for the synthetic cross-game pair ledger.

Two purposes:

    1. Unit-test the synthesis math on tiny inline fixtures.
    2. Pin the observed behavior on the real singles ledger this repo
       carries today, so a drift is loud rather than silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.backtest_pair_ledger import build_report
from sports.mlb.parlay_v2.promotion_coherence.synthesize_pairs import (
    DEFAULT_MAX_SINGLES_PER_GAME,
    DEFAULT_PAIRS_PER_SLATE_CAP,
    DEFAULT_SINGLES_LEDGER,
    SYNTHETIC_LEDGER_VERSION,
    _make_pair_row,
    _pair_id,
    _row_ok_for_synthesis,
    synthesize_pair_ledger,
    write_synthetic_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_SINGLES = REPO_ROOT / DEFAULT_SINGLES_LEDGER


# --- unit ---------------------------------------------------------------

def _single(**overrides):
    base = {
        "observation_id": "single-1",
        "slate_id": "20260803",
        "game_id": "game-A",
        "player_id": "player_1",
        "market_bucket": "R",
        "side": "OVER",
        "line": 0.5,
        "quote_decimal": 2.0,
        "predictive_probability_if_available": 0.6,
        "actual_outcome": 1,
        "settlement_status": "win",
    }
    base.update(overrides)
    return base


def test_row_ok_for_synthesis_requires_full_settlement_info() -> None:
    assert _row_ok_for_synthesis(_single()) is True
    assert _row_ok_for_synthesis(_single(settlement_status="pending")) is False
    assert _row_ok_for_synthesis(_single(quote_decimal=1.0)) is False
    assert _row_ok_for_synthesis(_single(quote_decimal="n/a")) is False
    assert _row_ok_for_synthesis(_single(predictive_probability_if_available=None)) is False
    assert _row_ok_for_synthesis(_single(predictive_probability_if_available=0.0)) is False
    assert _row_ok_for_synthesis(_single(predictive_probability_if_available=1.0)) is False
    assert _row_ok_for_synthesis(_single(game_id=None)) is False


def test_pair_id_is_order_independent() -> None:
    a = _single(observation_id="A")
    b = _single(observation_id="B")
    assert _pair_id(a, b) == _pair_id(b, a)


def test_make_pair_row_independence_math_and_return_when_both_win() -> None:
    a = _single(observation_id="A", game_id="game-A", predictive_probability_if_available=0.6, quote_decimal=1.8, actual_outcome=1)
    b = _single(observation_id="B", game_id="game-B", predictive_probability_if_available=0.5, quote_decimal=2.0, actual_outcome=1)
    row = _make_pair_row(a, b)
    assert row["predicted_joint_probability"] == pytest.approx(0.30)
    assert row["quoted_pair_price"] == pytest.approx(3.6)
    assert row["both_win"] is True
    # combined price 3.6 -> return +2.6 on a unit stake
    assert row["actual_pair_return"] == pytest.approx(2.6)
    assert row["same_game"] is False
    assert row["is_synthetic"] is True
    assert row["synthetic_ledger_version"] == SYNTHETIC_LEDGER_VERSION
    assert row["leg_1_model_probability"] == pytest.approx(0.6)
    assert row["leg_2_model_probability"] == pytest.approx(0.5)


def test_make_pair_row_return_when_one_loses() -> None:
    a = _single(observation_id="A", game_id="game-A", actual_outcome=1)
    b = _single(observation_id="B", game_id="game-B", actual_outcome=0)
    row = _make_pair_row(a, b)
    assert row["both_win"] is False
    assert row["actual_pair_return"] == pytest.approx(-1.0)


def test_synthesize_pair_ledger_caps_per_slate_deterministically(tmp_path: Path) -> None:
    # Two games, five singles each, cap 4 pairs total per slate -> only 4
    # emitted, chosen deterministically by observation_id order.
    singles = []
    for i in range(5):
        singles.append(_single(observation_id=f"a{i:02d}", game_id="game-A", player_id=f"a{i}"))
    for i in range(5):
        singles.append(_single(observation_id=f"b{i:02d}", game_id="game-B", player_id=f"b{i}"))
    p = tmp_path / "singles.jsonl"
    with open(p, "w") as f:
        for row in singles:
            f.write(json.dumps(row) + "\n")

    pairs, meta = synthesize_pair_ledger(
        singles_path=p, max_singles_per_game=5, pairs_per_slate_cap=4,
    )
    assert len(pairs) == 4
    # Determinism check: same call, same output
    pairs2, _ = synthesize_pair_ledger(
        singles_path=p, max_singles_per_game=5, pairs_per_slate_cap=4,
    )
    assert [r["pair_id"] for r in pairs] == [r["pair_id"] for r in pairs2]


def test_synthesize_pair_ledger_only_crosses_different_games(tmp_path: Path) -> None:
    singles = [
        _single(observation_id="a1", game_id="game-A", player_id="ap1"),
        _single(observation_id="a2", game_id="game-A", player_id="ap2"),
        _single(observation_id="b1", game_id="game-B", player_id="bp1"),
    ]
    p = tmp_path / "singles.jsonl"
    with open(p, "w") as f:
        for row in singles:
            f.write(json.dumps(row) + "\n")

    pairs, _ = synthesize_pair_ledger(singles_path=p, pairs_per_slate_cap=100)
    # 2*1 = 2 cross-game pairs; NO within-game pair
    assert len(pairs) == 2
    for r in pairs:
        assert r["same_game"] is False
        # No pair carries two legs from game-A
        assert not (r["leg_1_event_id"].split("|")[1] == r["leg_2_event_id"].split("|")[1])


def test_write_synthetic_ledger_round_trip(tmp_path: Path) -> None:
    rows = [
        _make_pair_row(
            _single(observation_id="A", game_id="game-A"),
            _single(observation_id="B", game_id="game-B"),
        )
    ]
    p = tmp_path / "synth.jsonl"
    write_synthetic_ledger(rows, p)
    read_back = [json.loads(l) for l in p.read_text().splitlines() if l]
    assert len(read_back) == 1
    assert read_back[0]["pair_id"] == rows[0]["pair_id"]


# --- observed behavior on the REAL singles ledger -----------------------

@pytest.fixture(scope="module")
def real_synth_ledger(tmp_path_factory):
    if not REAL_SINGLES.exists():
        pytest.skip(f"real singles ledger missing: {REAL_SINGLES}")
    out = tmp_path_factory.mktemp("synth") / "synthetic.jsonl"
    pairs, meta = synthesize_pair_ledger(
        singles_path=REAL_SINGLES,
        max_singles_per_game=DEFAULT_MAX_SINGLES_PER_GAME,
        pairs_per_slate_cap=DEFAULT_PAIRS_PER_SLATE_CAP,
    )
    write_synthetic_ledger(pairs, out)
    return out, meta


def test_real_singles_produce_meaningfully_more_pairs_than_real_pair_ledger(real_synth_ledger) -> None:
    out, meta = real_synth_ledger
    # The real pair ledger is 3,120 rows. The whole point of the
    # synthetic ledger is to grow evidence past that. This pin says we
    # cleared 5x the real-pair-ledger row count so any accidental
    # regression in scale is loud. If the singles ledger shrinks in the
    # future the pin will fail and this doc gets rewritten.
    assert meta.total_synthetic_pairs > 15000
    assert len(meta.slates_covered) >= 20


def test_synthetic_ledger_backtest_flips_positive_above_zero_floor(real_synth_ledger) -> None:
    """Honest observed finding on the current synthetic ledger:

    Above a small positive promotion-margin floor, the synthetic broader
    cross-game pool has a POSITIVE realized return per unit -- unlike
    the frozen production pair ledger which is negative at every floor.
    This is not a production-ROI claim (the synthetic pool is not the
    real production candidate universe -- see BACKTEST_ANALYSIS.md's
    frank explanation of the discrepancy). It IS a claim that the
    promotion-margin rule has real predictive value on a broader-than-
    production pool, and that the current production candidate selector
    is itself where the negative expectation sits.
    """
    out, _ = real_synth_ledger
    report = build_report(
        ledger_path=out,
        floors=[round(x / 100, 2) for x in range(-10, 11, 1)],
    )
    all_slice = next(s for s in report.slices if s.name == "ALL_SETTLED_PAIRS")

    # At the baseline (accept everything), synthetic is still slightly
    # negative -- reasonable, since the singles pool also loses on
    # average.
    baseline = all_slice.baseline_all_admitted
    assert baseline.total_return_per_unit < 0

    # ... but somewhere above the +0 floor the total return crosses
    # zero.
    positive_floors = [
        r for r in all_slice.floor_sweep
        if r.floor >= 0.0 and r.total_return_per_unit is not None and r.total_return_per_unit > 0
    ]
    assert positive_floors, (
        "no positive-margin floor produces positive total return on the "
        "synthetic ledger any more; BACKTEST_ANALYSIS.md needs updating"
    )

    # And the strict-dominance flag fires: this slice deserves the flag
    # even under the anti-overfit 20-admitted floor.
    assert all_slice.strict_dominance_over_baseline is not None
    dom = all_slice.strict_dominance_over_baseline
    assert dom["dominant_admitted_count"] >= 20
    assert dom["dominant_total_return_per_unit"] > baseline.total_return_per_unit

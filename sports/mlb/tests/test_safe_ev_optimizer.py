from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MLB_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import safe_ev_optimizer as optimizer  # noqa: E402


def _pick(name, *, safe_probability=None, safe_expected_value=None, calibrated_hit_probability=None,
          expected_value_per_unit=None, market_bucket="default", team="TEAM"):
    return SimpleNamespace(
        name=name,
        safe_probability=safe_probability,
        safe_expected_value=safe_expected_value,
        calibrated_hit_probability=calibrated_hit_probability,
        expected_value_per_unit=expected_value_per_unit,
        market_bucket=market_bucket,
        team=team,
    )


def test_effective_probability_and_ev_prefer_safe_fields_but_fall_back_to_v11() -> None:
    veto_active = _pick("active", safe_probability=0.6, safe_expected_value=0.1, calibrated_hit_probability=0.9, expected_value_per_unit=0.9)
    veto_inactive = _pick("inactive", calibrated_hit_probability=0.7, expected_value_per_unit=0.2)

    assert optimizer.effective_probability(veto_active) == 0.6  # not the higher v11 number
    assert optimizer.effective_expected_value(veto_active) == 0.1
    assert optimizer.effective_probability(veto_inactive) == 0.7  # real v11 fallback
    assert optimizer.effective_expected_value(veto_inactive) == 0.2


def test_optimize_slate_picks_the_exact_max_ev_combination_within_the_miss_budget() -> None:
    # miss = 1 - p. Feasible combos under budget=0.3: A+B (miss=0.30, ev=1.1)
    # is the real maximum; A+B+D (miss=0.35) is infeasible.
    a = _pick("a", safe_probability=0.9, safe_expected_value=0.5, team="T1", market_bucket="A")
    b = _pick("b", safe_probability=0.8, safe_expected_value=0.6, team="T1", market_bucket="B")
    c = _pick("c", safe_probability=0.5, safe_expected_value=1.0, team="T2", market_bucket="C")
    d = _pick("d", safe_probability=0.95, safe_expected_value=0.1, team="T2", market_bucket="D")

    outcome = optimizer.optimize_slate(
        [a, b, c, d], miss_budget=0.3, max_picks=10, max_per_market_bucket=2, max_per_team=2
    )

    assert outcome["status"] == "optimal"
    assert {candidate.name for candidate in outcome["selected"]} == {"a", "b"}
    assert outcome["expected_value_total"] == pytest.approx(1.1)
    assert outcome["miss_budget_used"] == pytest.approx(0.3)


def test_optimize_slate_enforces_the_per_team_diversification_cap() -> None:
    # Same candidates as above, but capping to 1 pick per team forces the
    # optimizer off the unconstrained {a, b} answer (both team T1) onto
    # the real next-best feasible combination: {b, d} (ev=0.7).
    a = _pick("a", safe_probability=0.9, safe_expected_value=0.5, team="T1", market_bucket="A")
    b = _pick("b", safe_probability=0.8, safe_expected_value=0.6, team="T1", market_bucket="B")
    d = _pick("d", safe_probability=0.95, safe_expected_value=0.1, team="T2", market_bucket="D")

    outcome = optimizer.optimize_slate(
        [a, b, d], miss_budget=0.3, max_picks=10, max_per_market_bucket=2, max_per_team=1
    )

    assert {candidate.name for candidate in outcome["selected"]} == {"b", "d"}


def test_optimize_slate_enforces_the_per_market_bucket_diversification_cap() -> None:
    e = _pick("e", safe_probability=0.9, safe_expected_value=0.5, team="T1", market_bucket="X")
    f = _pick("f", safe_probability=0.85, safe_expected_value=0.55, team="T2", market_bucket="X")
    g = _pick("g", safe_probability=0.9, safe_expected_value=0.2, team="T3", market_bucket="Y")

    outcome = optimizer.optimize_slate(
        [e, f, g], miss_budget=10.0, max_picks=10, max_per_market_bucket=1, max_per_team=2
    )

    # Only one of e/f (same bucket "X") may be picked -- the optimizer
    # picks the higher-EV one (f), plus g from the unconstrained bucket.
    assert {candidate.name for candidate in outcome["selected"]} == {"f", "g"}


def test_optimize_slate_reports_no_usable_candidates_when_nothing_is_priced() -> None:
    unusable = _pick("unpriced")  # no safe_* and no v11 fallback fields either

    outcome = optimizer.optimize_slate([unusable], miss_budget=2.0)

    assert outcome["status"] == "no_usable_candidates"
    assert outcome["selected"] == []
    assert outcome["candidates_usable"] == 0

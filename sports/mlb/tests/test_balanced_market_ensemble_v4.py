from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import balanced_market_ensemble_v4 as v4  # noqa: E402


def _row(date: str, win: int, balanced: float, market: float, suffix: str = "") -> dict:
    return {
        "candidate_id": f"{date}-{suffix}", "date": date, "win": win,
        "balanced_probability": balanced, "market_probability": market,
    }


def test_blend_endpoints_are_market_and_balanced() -> None:
    assert v4.blend_probability(0.7, 0.55, 0.0) == pytest.approx(0.55)
    assert v4.blend_probability(0.7, 0.55, 1.0) == pytest.approx(0.7)


def test_equal_slate_loss_prevents_large_slate_pseudoreplication() -> None:
    rows = [_row("2026-08-01", 1, 0.9, 0.1, str(i)) for i in range(100)]
    rows += [_row("2026-08-02", 0, 0.9, 0.1)]
    loss = v4.equal_slate_log_loss(rows, 1.0)
    assert loss == pytest.approx((-__import__("math").log(0.9) - __import__("math").log(0.1)) / 2)


def test_fit_uses_strictly_earlier_slates() -> None:
    prior = [_row(f"2026-08-0{day}", 1, 0.8, 0.4) for day in range(1, 5)]
    future = [_row("2026-08-06", 0, 0.8, 0.4)] * 100
    assert v4.fit(prior, before_date="2026-08-05") == v4.fit(prior + future, before_date="2026-08-05")


def test_cross_fitted_residuals_leave_each_slate_out(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[set[str]] = []
    original = v4.select_weight
    def recording(rows):
        rows = list(rows); calls.append({row["date"] for row in rows}); return original(rows)
    monkeypatch.setattr(v4, "select_weight", recording)
    rows = [_row(f"2026-08-0{day}", day % 2, 0.6, 0.55) for day in range(1, 5)]
    fitted = v4.fit(rows, before_date="2026-08-05")
    assert fitted.training_slates == 4
    assert len(fitted.cross_fitted_slate_residuals) == 4
    assert all(len(call) == 3 for call in calls[1:])


def test_positive_residual_cannot_inflate_safe_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v4, "one_sided_mean_lcb", lambda *_args, **_kwargs: 0.08)
    rows = [_row(f"2026-08-0{day}", 1, 0.7, 0.6) for day in range(1, 5)]
    fitted = v4.fit(rows, before_date="2026-08-05")
    scored = v4.score({"candidate_id": "x", "balanced_probability": 0.7, "market_probability": 0.6, "price": -150}, fitted)
    assert fitted.safe_calibration_adjustment == 0.0
    assert scored.safe_probability == pytest.approx(scored.ensemble_probability)


def test_uncertainty_penalty_changes_probability_not_only_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v4, "one_sided_mean_lcb", lambda *_args, **_kwargs: -0.04)
    rows = [_row(f"2026-08-0{day}", day % 2, 0.7, 0.6) for day in range(1, 5)]
    fitted = v4.fit(rows, before_date="2026-08-05")
    scored = v4.score({"candidate_id": "x", "balanced_probability": 0.7, "market_probability": 0.6, "price": -150}, fitted)
    assert scored.safe_probability == pytest.approx(scored.ensemble_probability - 0.04)


def test_price_cannot_rescue_unsupported_probability_edge() -> None:
    fitted = v4.V4Fit(10, 100, 0.5, 0.5, 0.6, (0.0,) * 10, 0.0, 0.0)
    scored = v4.score({"candidate_id": "x", "balanced_probability": 0.602, "market_probability": 0.6, "price": 200}, fitted)
    assert scored.eligible is False
    assert "safe_probability_edge_below_1pct" in scored.reasons


def test_no_pick_quota_all_supported_candidates_survive() -> None:
    fitted = v4.V4Fit(10, 100, 1.0, 0.0, 0.5, (0.0,) * 10, 0.0, 0.0)
    candidates = [
        {"candidate_id": str(i), "balanced_probability": 0.7, "market_probability": 0.6, "price": -150}
        for i in range(50)
    ]
    report = v4.run_shadow(candidates, [], slate_date="2026-08-10")
    # run_shadow refits empty history, so verify the actual score contract
    # directly: every independently supported candidate is eligible.
    assert all(v4.score(candidate, fitted).eligible for candidate in candidates)
    assert report["pick_count_constraint"] == "none"


def test_spec_hash_matches_frozen_constants() -> None:
    assert v4.PREREGISTRATION_SPEC_HASH == v4._spec_hash()

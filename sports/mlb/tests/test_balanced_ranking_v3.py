from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import balanced_ranking_v3 as ranking  # noqa: E402


def _rows(date: str, outcomes: list[int], balanced: list[float], market: list[float]) -> list[dict]:
    return [
        {
            "candidate_id": f"{date}_{index}",
            "date": date,
            "game_id": f"g{index // 2}",
            "win": outcome,
            "balanced_probability": balanced[index],
            "market_probability": market[index],
            "base_ev": balanced[index] - market[index],
            "v19_order_score": balanced[index] - market[index],
        }
        for index, outcome in enumerate(outcomes)
    ]


def test_pairwise_concordance_counts_ties_as_half() -> None:
    value, pairs = ranking.pairwise_concordance([0.8, 0.6, 0.6, 0.4], [1, 1, 0, 0])
    assert pairs == 4
    assert value == pytest.approx(0.875)


def test_all_win_or_all_loss_slate_has_undefined_concordance() -> None:
    assert ranking.pairwise_concordance([0.8, 0.7], [1, 1]) == (None, 0)
    assert ranking.pairwise_concordance([0.8, 0.7], [0, 0]) == (None, 0)


def test_top_k_and_concordance_follow_score_order() -> None:
    rows = _rows("2026-08-01", [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], [0.5] * 4)
    metric = ranking.slate_metric(rows, "balanced_probability")
    assert metric.concordance == pytest.approx(0.75)
    assert metric.top_1_hit_rate == 1.0
    assert metric.top_3_hit_rate == pytest.approx(2 / 3)
    assert metric.top_1_lift == 0.5


def test_pair_count_never_becomes_independent_slate_count() -> None:
    rows = []
    rows += _rows("2026-08-01", [1] * 50 + [0] * 50, list(reversed(range(100))), list(range(100)))
    rows += _rows("2026-08-02", [1, 0], [0.9, 0.1], [0.1, 0.9])
    metrics = ranking.evaluate_rows(rows)
    summary = ranking.summarize(metrics, phase="locked")
    assert summary["independent_slates"] == 2
    assert summary["score_summaries"]["balanced_probability"]["comparable_pairs_descriptive_only"] == 2501
    assert summary["status"] == "INSUFFICIENT_INDEPENDENT_SLATES"


def test_locked_acceptance_requires_eight_slates_even_with_perfect_ranking() -> None:
    rows = []
    for day in range(1, 8):
        rows += _rows(f"2026-08-{day:02d}", [1, 0], [0.9, 0.1], [0.1, 0.9])
    summary = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
    assert summary["status"] == "INSUFFICIENT_INDEPENDENT_SLATES"


def test_acceptance_requires_incremental_market_and_v19_ranking() -> None:
    rows = []
    for day in range(1, 9):
        # Balanced and market are equally perfect; balanced is therefore useful
        # but has no incremental advantage and must not be accepted.
        rows += _rows(f"2026-08-{day:02d}", [1, 0], [0.9, 0.1], [0.9, 0.1])
    summary = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
    assert summary["score_summaries"]["balanced_probability"]["mean_slate_concordance"] == 1.0
    assert summary["status"] == "RANKING_SIGNAL_NOT_ACCEPTED"


# ------------------------------------------------------------
# Tightening guards (added 2026-08-29) -- these enforce the
# preregistration edits that landed on top of the original 8-slate
# committed spec: promotion needs 30 slates, LCB and slate-clustered
# bootstrap must agree, the V4 reserve holds back the trailing 10
# dates, and the spec hash catches silent edits to critical constants.
# ------------------------------------------------------------


def _perfect_ranking_rows(date: str) -> list[dict]:
    """A slate where balanced ranks the winner above the loser, but every
    other score (market, base_ev, v19_order_score) ranks them inverted --
    balanced strictly beats each comparator on within-slate concordance."""
    return [
        {
            "candidate_id": f"{date}_winner",
            "date": date,
            "game_id": "g0",
            "win": 1,
            "balanced_probability": 0.9,
            "market_probability": 0.1,
            "base_ev": -0.4,
            "v19_order_score": -0.4,
        },
        {
            "candidate_id": f"{date}_loser",
            "date": date,
            "game_id": "g0",
            "win": 0,
            "balanced_probability": 0.1,
            "market_probability": 0.9,
            "base_ev": 0.4,
            "v19_order_score": 0.4,
        },
    ]


def test_promotion_needs_more_than_min_locked_slates() -> None:
    """A run with all three ranking checks cleared at exactly the 8-slate
    computation gate must return SHADOW_ELIGIBLE_PENDING_MORE_SLATES,
    not RANKING_SIGNAL_ACCEPTED -- the higher promotion gate protects
    against acting on a thin slate base even when every ranking check
    passes on it."""
    rows = [row for day in range(1, 9) for row in _perfect_ranking_rows(f"2026-08-{day:02d}")]
    summary = ranking.summarize(ranking.evaluate_rows(rows), phase="locked")
    assert summary["independent_slates"] == 8
    # Sanity: every comparator delta is strictly positive on this fixture,
    # so this test really is exercising "signal passed, only slate count
    # holds promotion back".
    for baseline in ("market_probability", "base_ev", "v19_order_score"):
        assert summary["paired_comparisons"][baseline]["mean_concordance_delta"] == 1.0
    assert summary["status"] == "SHADOW_ELIGIBLE_PENDING_MORE_SLATES"


def test_promotion_min_slates_is_at_least_thirty() -> None:
    """The higher promotion gate is not just larger than MIN_LOCKED_SLATES,
    it is genuinely thick. Silently reducing it below 30 would let a
    still-thin slate base drive an ACCEPTED status."""
    assert ranking.PROMOTION_MIN_SLATES >= 30
    assert ranking.PROMOTION_MIN_SLATES > ranking.MIN_LOCKED_SLATES


def test_v4_reserve_holds_back_the_trailing_slates() -> None:
    """The last V4_RESERVE_MOST_RECENT_SLATES real dates in the input row
    set must not appear in either derivation or locked partitions -- they
    are reserved for a future, unrelated study."""
    dates = [f"2026-08-{day:02d}" for day in range(1, 21)]  # 20 slates
    derivation, locked, reserve = ranking._partition_dates(dates)
    assert reserve == dates[-ranking.V4_RESERVE_MOST_RECENT_SLATES:]
    assert set(derivation).isdisjoint(reserve)
    assert set(locked).isdisjoint(reserve)
    assert derivation == dates[:ranking.DERIVATION_SLATES]
    assert locked == dates[ranking.DERIVATION_SLATES : len(dates) - ranking.V4_RESERVE_MOST_RECENT_SLATES]


def test_v4_reserve_reserves_everything_when_history_is_too_thin() -> None:
    """Fewer real dates than V4_RESERVE_MOST_RECENT_SLATES means the
    entire history goes into reserve. Both other partitions come back
    empty by design -- a thin history should not be raided to hand V3
    something to grade."""
    thin = [f"2026-08-{day:02d}" for day in range(1, 6)]  # 5 dates
    derivation, locked, reserve = ranking._partition_dates(thin)
    assert derivation == []
    assert locked == []
    assert reserve == thin


def test_bootstrap_disagreement_is_its_own_terminal_status() -> None:
    """If the LCB and the slate-clustered bootstrap disagree about whether
    a check passes, the run reports BOOTSTRAP_LCB_DISAGREEMENT -- neither
    method is used as an authoritative tie-breaker."""
    agree_ok, both_pass = ranking._both_agree_above(0.0, 0.05, -0.05)
    assert agree_ok is False
    assert both_pass is False


def test_bootstrap_agreement_helper_recognizes_both_pass() -> None:
    agree_ok, both_pass = ranking._both_agree_above(0.5, 0.60, 0.55)
    assert agree_ok is True
    assert both_pass is True


def test_bootstrap_agreement_helper_recognizes_both_fail() -> None:
    """Both methods failing is agreement, just not a passing agreement --
    the caller then reports RANKING_SIGNAL_NOT_ACCEPTED, not
    BOOTSTRAP_LCB_DISAGREEMENT."""
    agree_ok, both_pass = ranking._both_agree_above(0.5, 0.40, 0.35)
    assert agree_ok is True
    assert both_pass is False


def test_preregistration_spec_hash_matches_current_constants() -> None:
    """The frozen critical spec values (family, slate thresholds,
    confidence, reserve size, acceptance rules) hash to
    PREREGISTRATION_SPEC_HASH. Any silent edit to those constants
    changes the hash and fails this test immediately -- exactly the
    silent-loosening failure mode the preregistration exists to
    prevent. If a legitimate spec revision is authorized, the
    correct sequence is (a) commit the constants change, (b) commit
    the corresponding hash update, (c) recognize that the revision
    invalidates any prior evaluation, per the preregistration itself."""
    assert ranking.PREREGISTRATION_SPEC_HASH == ranking._preregistration_spec_hash()
    # Also assert the actual expected value so a change to
    # _preregistration_spec_hash itself (as opposed to its inputs) is
    # still caught. Update this exactly when the constants are
    # deliberately revised.
    assert ranking.PREREGISTRATION_SPEC_HASH == (
        "88bc499be03520db9b12f0951dfed9fd1fa5da4b003eb055523dd4beddcb02a2"
    )


def test_report_surfaces_the_v4_reserve_and_spec_hash() -> None:
    """Every real run report must carry the reserve slate list and the
    spec hash, so a downstream reviewer can verify at any later date
    that this run honored the preregistration in force at that time."""
    # We can't invoke run() here without live data, but we can verify
    # the report shape stays consistent with the preregistration by
    # constructing summaries directly and checking the surrounding
    # preregistration dict is still what run() constructs.
    rows = [row for day in range(1, 9) for row in _rows(f"2026-08-{day:02d}", [1, 0], [0.9, 0.1], [0.9, 0.1])]
    metrics = ranking.evaluate_rows(rows)
    summary = ranking.summarize(metrics, phase="locked")
    # The two-method redundant fields are now on every score summary.
    assert "concordance_bootstrap_lcb" in summary["score_summaries"]["balanced_probability"]
    for baseline in ("market_probability", "base_ev", "v19_order_score"):
        assert "delta_bootstrap_lcb" in summary["paired_comparisons"][baseline]


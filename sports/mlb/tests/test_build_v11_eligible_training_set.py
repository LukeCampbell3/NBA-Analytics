from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
SITE_PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))
sys.path.insert(0, str(SITE_PIPELINE_ROOT))

import build_v11_eligible_training_set as builder  # noqa: E402
import run_daily_predictions as shared_daily_predictions  # noqa: E402


def test_v11_selector_args_stay_in_sync_with_the_real_production_policy():
    """This must never silently drift from what the live selector actually
    runs -- see MLB_PRIMARY_POLICY_ARGS in run_daily_predictions.py."""
    assert builder.V11_SELECTOR_ARGS == shared_daily_predictions.MLB_PRIMARY_POLICY_ARGS


def test_find_raw_pool_csvs_skips_the_already_selected_sibling(tmp_path):
    date_dir = tmp_path / "20260810"
    date_dir.mkdir()
    raw = date_dir / "daily_prediction_pool_20260810.csv"
    raw.write_text("raw", encoding="utf-8")
    (date_dir / "daily_prediction_pool_20260810_high_precision_predictions.csv").write_text("selected", encoding="utf-8")
    (date_dir / "daily_prediction_pool_20260810_max_winrate.csv").write_text("other", encoding="utf-8")

    found = builder.find_raw_pool_csvs(tmp_path)

    assert found == [raw]


def test_find_raw_pool_csvs_skips_a_date_dir_missing_its_own_raw_csv(tmp_path):
    date_dir = tmp_path / "20260811"
    date_dir.mkdir()
    (date_dir / "daily_prediction_pool_20260811_high_precision_predictions.csv").write_text("selected", encoding="utf-8")

    assert builder.find_raw_pool_csvs(tmp_path) == []


def _candidate(**overrides) -> SimpleNamespace:
    base = dict(
        player="Real Player",
        game_id="824970",
        target="TB",
        direction="UNDER",
        market_line=1.5,
        run_date=date(2026, 8, 10),
        raw={"Player_Type": "hitter"},
        calibrated_hit_probability=0.75,
        calibrated_graded_hit_rate=0.75,
        survival_probability=0.7,
        edge=-0.66,
        abs_edge=0.66,
        market_implied_probability=0.6,
        market_line_std=0.0,
        market_books=6,
        market_common_books=3,
        history_rows=40,
        historical_bucket_win_rate=0.86,
        historical_bucket_support=9710,
        historical_bet_profile_win_rate=0.6,
        historical_bet_profile_roi=0.1,
        historical_bet_profile_support=20,
        historical_market_availability_rate=0.8,
        historical_market_availability_support=5,
        live_confidence_calibration_adjustment=-0.04,
        selected_side_price=-176.0,
        price_confirmed=True,
        expected_value_per_unit=0.25,
        market_bucket="TB|UNDER|1.5",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_candidate_to_row_grades_a_real_win():
    candidate = _candidate()  # UNDER 1.5, actual 1.0 -> win
    lookup = {("2026-08-10", "real_player", "TB", "824970"): 1.0}

    row = builder.candidate_to_row(candidate, lookup)

    assert row is not None
    assert row["win"] == 1
    assert row["target"] == "TB"
    assert row["model_hit_probability"] == 0.75


def test_candidate_to_row_grades_a_real_loss():
    candidate = _candidate()  # UNDER 1.5, actual 3.0 -> loss
    lookup = {("2026-08-10", "real_player", "TB", "824970"): 3.0}

    row = builder.candidate_to_row(candidate, lookup)

    assert row is not None
    assert row["win"] == 0


def test_candidate_to_row_excludes_a_push():
    candidate = _candidate(market_line=1.0)  # UNDER 1.0, actual 1.0 -> push
    lookup = {("2026-08-10", "real_player", "TB", "824970"): 1.0}

    assert builder.candidate_to_row(candidate, lookup) is None


def test_candidate_to_row_returns_none_when_no_real_settled_actual_exists():
    candidate = _candidate()
    assert builder.candidate_to_row(candidate, {}) is None

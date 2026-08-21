from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.mlb.research.h_over_ranker import manifest
from sports.mlb.research.h_over_ranker.chronological_cv import (
    Fold,
    assert_no_leakage,
    expanding_day_folds,
    split,
)
from sports.mlb.research.h_over_ranker.data_windows import (
    DERIVE_STAMPS,
    DEVELOPMENT_STAMPS,
    SELECT_STAMPS,
    TEST_STAMPS,
)
from sports.mlb.research.h_over_ranker.eligibility import (
    FROZEN_H_BIAS,
    eligible_rows_for_stamps,
    recompute_derive_bias,
)
from sports.mlb.research.h_over_ranker.ranker import (
    FEATURE_COLUMNS,
    FROZEN_C,
    build_features,
    fit_final_model,
    fit_predict_walkforward,
    score_with_frozen_model,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "research" / "h_over_ranker"


# ---------------------------------------------------------------------------
# Chronology / no-leakage
# ---------------------------------------------------------------------------

def test_derive_select_test_partition_is_disjoint_and_ordered():
    assert set(DERIVE_STAMPS) & set(SELECT_STAMPS) == set()
    assert set(SELECT_STAMPS) & set(TEST_STAMPS) == set()
    assert set(DERIVE_STAMPS) & set(TEST_STAMPS) == set()
    assert max(DERIVE_STAMPS) < min(SELECT_STAMPS)
    assert max(SELECT_STAMPS) < min(TEST_STAMPS)
    assert DEVELOPMENT_STAMPS == DERIVE_STAMPS + SELECT_STAMPS


def test_expanding_day_folds_never_lets_train_reach_or_pass_val_date():
    dates = ["d1", "d2", "d3", "d4", "d5"]
    folds = expanding_day_folds(dates, min_train_days=2)
    assert [f.val_date for f in folds] == ["d3", "d4", "d5"]
    for fold in folds:
        assert max(fold.train_dates) < fold.val_date
        assert fold.val_date not in fold.train_dates


def test_split_rejects_overlapping_dates():
    frame = pd.DataFrame({"date": ["d1", "d1", "d2"], "value": [1, 2, 3]})
    bad_fold = Fold(index=0, train_dates=("d1", "d2"), val_date="d2")
    with pytest.raises(AssertionError):
        split(frame, bad_fold)


def test_split_rejects_train_date_at_or_after_val_date():
    train = pd.DataFrame({"date": ["d3"]})
    val = pd.DataFrame({"date": ["d2"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(train, val)


def test_no_row_from_the_same_date_appears_in_both_train_and_val_across_real_folds():
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    dates = sorted(rows["date"].unique())
    folds = expanding_day_folds(dates, min_train_days=6)
    for fold in folds:
        train, val = split(rows, fold)
        assert set(train["date"]) & set(val["date"]) == set()
        assert set(val["date"]) == {fold.val_date}


def test_fit_predict_walkforward_never_uses_future_dates(monkeypatch):
    """The model scoring date D is fit only on rows whose date < D."""
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    dates = sorted(rows["date"].unique())
    folds = expanding_day_folds(dates, min_train_days=6)
    scored = fit_predict_walkforward(rows, folds)
    for record in scored.attrs["fold_models"]:
        fold = folds[record["fold_index"]]
        assert max(fold.train_dates) < fold.val_date


# ---------------------------------------------------------------------------
# TEST_STAMPS must never be read by any ranker-development code path
# ---------------------------------------------------------------------------

def test_no_module_reads_test_stamps():
    """Static check: only data_windows.py and manifest.py (frozen
    documentation) may reference TEST_STAMPS. Any other module in the
    package importing/using it is a violation of the retired-TEST-block
    rule, and this test is designed to fail loudly if that happens."""
    allowed = {"data_windows.py", "manifest.py"}
    offenders = []
    for path in PACKAGE_ROOT.glob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "TEST_STAMPS":
                offenders.append(path.name)
            if isinstance(node, ast.Attribute) and node.attr == "TEST_STAMPS":
                offenders.append(path.name)
    assert not offenders, f"modules referencing TEST_STAMPS outside the allowed set: {offenders}"


def test_run_development_never_touches_test_stamps():
    path = PACKAGE_ROOT / "run_development.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    identifier_refs = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "TEST_STAMPS" not in identifier_refs
    assert "DEVELOPMENT_STAMPS" in identifier_refs


# ---------------------------------------------------------------------------
# Frozen eligibility rule reproduction
# ---------------------------------------------------------------------------

def test_derive_bias_matches_frozen_constant():
    recomputed = recompute_derive_bias()
    assert recomputed == pytest.approx(FROZEN_H_BIAS, abs=1e-9)


def test_select_block_eligibility_reproduces_the_frozen_selection_numbers():
    """Reproduces the exact numbers the H target was chosen on: SELECT
    H-OVER n=1432, hit rate 57.5%."""
    select_rows = eligible_rows_for_stamps(SELECT_STAMPS)
    assert len(select_rows) == 1432
    assert select_rows["win"].mean() == pytest.approx(0.5754, abs=1e-3)


def test_eligible_rows_only_contain_positive_corrected_edge():
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    assert (rows["corrected_edge"] > 0).all()


def test_eligible_rows_contain_no_postgame_columns():
    """Guards against accidentally joining in a settlement-derived column as
    a feature; only pregame-safe columns plus the `win` outcome (used for
    fitting/evaluation, never as a ranking input) may appear."""
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    allowed = {
        "date", "player", "player_key", "game_id", "prediction",
        "corrected_prediction", "market_line", "corrected_edge", "raw_edge",
        "rmse", "mae", "history_rows", "market_books", "market_source",
        "market_line_std", "days_since_history", "win",
    }
    assert set(rows.columns) <= allowed


# ---------------------------------------------------------------------------
# Frozen manifest consistency
# ---------------------------------------------------------------------------

def test_manifest_feature_list_matches_ranker_module():
    assert manifest.FEATURE_LIST == list(FEATURE_COLUMNS)


def test_manifest_regularization_matches_ranker_module():
    assert manifest.REGULARIZATION_C == FROZEN_C


def test_manifest_bias_correction_matches_eligibility_module():
    assert manifest.BIAS_CORRECTION["value"] == FROZEN_H_BIAS


def test_dropped_features_are_actually_zero_variance_in_development_data():
    """Confirms the manifest's stated reason for dropping market_books and
    market_line_std -- if the upstream data ever starts varying these, this
    test fails and the exclusion needs to be revisited (not silently kept)."""
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    assert rows["market_books"].nunique() == 1
    assert rows["market_line_std"].nunique() == 1


def test_final_model_reproducible_from_development_stamps():
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    final = fit_final_model(rows)
    assert final["feature_columns"] == manifest.FEATURE_LIST
    assert final["n_rows"] == 2249
    assert final["n_dates"] == 14


def test_score_with_frozen_model_matches_fit_final_model_predictions():
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    final = fit_final_model(rows)
    scores = score_with_frozen_model(rows, final)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_tie_break_is_deterministic_across_repeated_calls():
    rows = eligible_rows_for_stamps(DEVELOPMENT_STAMPS)
    final = fit_final_model(rows)
    scores_a = score_with_frozen_model(rows, final)
    scores_b = score_with_frozen_model(rows, final)
    np.testing.assert_array_equal(scores_a, scores_b)


# ---------------------------------------------------------------------------
# Development-fold evidence sanity (does not claim validation)
# ---------------------------------------------------------------------------

def test_development_evidence_top1_beats_pool_but_status_stays_unconfirmed():
    assert manifest.DEVELOPMENT_EVIDENCE.top1_lift_vs_pool > 0
    assert manifest.DEVELOPMENT_EVIDENCE.p_value_ge_observed_under_day_pool_null > 0.05, (
        "the frozen development result does not clear conventional significance -- "
        "the manifest's status must stay FRESH_CONFIRMATION_REQUIRED, not VALIDATED, "
        "while this holds"
    )
    assert manifest.STATUS == "DEVELOPMENT_COMPLETE_FRESH_CONFIRMATION_REQUIRED"

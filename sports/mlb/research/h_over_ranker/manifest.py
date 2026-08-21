from __future__ import annotations

"""FROZEN: H_OVER_RANKER_V1.

Confirmation policy (in effect from the moment this file is committed):
  - no parameter updates
  - no threshold changes
  - no feature changes
  - no retrospective reinterpretation
Any change to the ranker requires a new version (H_OVER_RANKER_V2, ...) in a
new file, not an edit to the constants below. `test_h_over_ranker.py`
enforces this by hashing this file's frozen content.

=== Status ===
H-OVER eligibility:      STRONG_CANDIDATE_FRESH_CONFIRMATION_REQUIRED
current probability ranking (baseline): INCREMENTAL_VALUE_NOT_ESTABLISHED
H_OVER_RANKER_V1:         DEVELOPMENT_COMPLETE_FRESH_CONFIRMATION_REQUIRED
combo model:               DEFERRED
dependence model:          DEFERRED

=== Why "fresh confirmation required", not "validated" ===
Development-fold result (DERIVE+SELECT, 8 walk-forward day-folds, TEST never
touched): top-1 lift vs. eligible pool = +30.3 points (87.5% vs 57.2%),
top-2 individual-leg hit rate 87.5%, day-clustered one-sided 95% lower bound
62.5%. But under the honest null (each fold's top-1 pick wins at that
day's own pool rate, no ranking skill), P(>=7 of 8 folds succeed) ~= 7.9%
-- a real, promising signal, but it does not clear conventional
significance with only 8 day-folds. Model/feature search (single-feature
ablations, a C-regularization sweep, an edge-shape comparison) happened on
this same DEVELOPMENT data; every trial is reported in reports/, but the
final choice (5 features, C=0.1) was selected by minimizing out-of-fold
Brier score -- a secondary diagnostic -- specifically to avoid picking
whichever configuration maximized the top-1/top-2 primary endpoint after
seeing it (that would have been C=1.0, which showed a suspiciously perfect
8/8 and was rejected for exactly that reason). See reports/ for every
tried variant, including the ones that did not make the cut.
"""

from dataclasses import dataclass, field

from .data_windows import DERIVE_STAMPS, DEVELOPMENT_STAMPS, SELECT_STAMPS, TEST_STAMPS
from .eligibility import FROZEN_H_BIAS
from .ranker import FROZEN_C, FEATURE_COLUMNS

VERSION = "H_OVER_RANKER_V1"
STATUS = "DEVELOPMENT_COMPLETE_FRESH_CONFIRMATION_REQUIRED"

ELIGIBILITY_VERSION = "H_OVER_ELIGIBILITY_V1"
ELIGIBILITY_RULE = (
    "target == 'H' and (Prediction - FROZEN_H_BIAS) - Market_Line > 0, "
    "graded via validate_historical_final_pools.grade_result(actual, Market_Line, 'OVER')"
)

# Frozen preprocessing / bias correction (see eligibility.py for the
# reproduction recipe and its consistency test).
BIAS_CORRECTION = {
    "method": "mean(Prediction - Actual) for target=='H' rows, DERIVE_STAMPS only",
    "value": FROZEN_H_BIAS,
    "derive_stamps": DERIVE_STAMPS,
}

# Frozen local-error estimate: Model_Val_RMSE straight from the raw daily
# pool CSV (a per-row, per-player walk-forward-validated RMSE the upstream
# pipeline already computes -- this package does not re-derive it).
LOCAL_ERROR_SOURCE = "Model_Val_RMSE column, daily_prediction_pool_*.csv (already walk-forward validated upstream)"

FEATURE_LIST = list(FEATURE_COLUMNS)
FEATURES_TRIED_AND_REJECTED = {
    "corrected_edge_sq": (
        "edge-shape test only marginally preferred quadratic (OOF Brier 0.24315 "
        "vs 0.24329 linear -- not a robust margin); ablation with it in the full "
        "model changed OOF Brier/top1/top2 by 0.0000 -- dead weight, dropped"
    ),
    "market_books": "constant at 0.0 for every H-target development row; zero-variance, dropped",
    "market_line_std": "constant at 0.0 for every H-target development row; zero-variance, dropped",
}

PREPROCESSING = "z-score standardize (train-fold mean/std) each feature before logistic regression"
MODEL_FORMULA = "logistic regression, R(x) = sigmoid(coef . standardize(x) + intercept)"
REGULARIZATION_C = FROZEN_C
REGULARIZATION_SELECTION_METHOD = (
    "minimum mean out-of-fold Brier score across a C sweep (1.0, 0.3, 0.1, 0.03, "
    "0.01, 0.003, 0.001); Brier was flat across C=1.0..0.01 (0.2430-0.2431), so C "
    "was chosen from the middle of that plateau (0.1) rather than the "
    "least-regularized point, which happened to show the best (and least "
    "trustworthy) top-1 number"
)

TIE_BREAKING = "sort by (score desc, rmse asc, player asc) -- deterministic, no row-order dependence"
DAILY_RANKING_PROCEDURE = (
    "restrict to H-OVER-eligible rows for the slate date; score each with the "
    "frozen model; sort per TIE_BREAKING; rank 1 = highest score"
)
TOP1_ACTION_DEFINITION = "the single rank-1 row for the date is the model's one recommended H-OVER pick"
TOP2_OBSERVATION_DEFINITION = (
    "ranks 1 and 2 for the date, tracked and reported for future analysis; "
    "NOT to be combined into a parlay/combo -- see status above, combo model is deferred"
)

SOURCE_DATA_CUTOFF = DEVELOPMENT_STAMPS[-1]  # last DEVELOPMENT date used to fit the final model

# Frozen historical TEST result -- documentation only. TEST_STAMPS itself
# must never be read again by any code path in this package; see
# test_h_over_ranker.py::test_no_module_reads_test_stamps.
FROZEN_TEST_RESULT = {
    "test_stamps": TEST_STAMPS,
    "target": "H-OVER (post FROZEN_H_BIAS correction), old probability-ranking era, pre-H_OVER_RANKER_V1",
    "n": 464,
    "hit_rate": 0.543,
    "naive_per_leg_clopper_pearson_95_lb": 0.504,
    "day_clustered_bootstrap_95_lb": 0.521,
    "day_clusters": 3,
    "old_top2_ranking_result": "3/6 = 0.500",
    "rest_of_eligible_pool_result": "n=458, 0.544",
}


@dataclass(frozen=True)
class FoldEvidence:
    n_folds: int
    top1_hit_rate: float
    top2_hit_rate: float
    top1_lift_vs_pool: float
    day_clustered_one_sided_95_lb: float
    p_value_ge_observed_under_day_pool_null: float


DEVELOPMENT_EVIDENCE = FoldEvidence(
    n_folds=8,
    top1_hit_rate=0.875,
    top2_hit_rate=0.875,
    top1_lift_vs_pool=0.303294,
    day_clustered_one_sided_95_lb=0.625,
    p_value_ge_observed_under_day_pool_null=0.0785,
)


PROSPECTIVE_CONFIRMATION_PROTOCOL = """
Primary endpoint (frozen, not to be changed after this point):
    On each fresh, never-before-used slate date, does H_OVER_RANKER_V1's
    daily top-1 pick win at a rate that beats the contemporaneous
    H-OVER-eligible pool's own hit rate that day?

Procedure:
    1. Each new slate date, compute H-OVER eligibility (eligibility.py,
       FROZEN_H_BIAS) and score every eligible row with
       ranker.score_with_frozen_model() using this manifest's frozen coef /
       intercept / standardize_mean / standardize_std -- never refit.
    2. Record the top-1 pick and its outcome once settled. Record the top-2
       pair and its outcome too, but only as a secondary observation (see
       TOP2_OBSERVATION_DEFINITION) -- not as a combo/parlay recommendation.
    3. Accumulate day-level (top1_hit, pool_hit_rate) pairs prospectively.
       Do not touch sports/mlb/data/predictions/daily_runs/ days already
       inside DEVELOPMENT_STAMPS or TEST_STAMPS for this -- only slates that
       postdate SOURCE_DATA_CUTOFF count as prospective confirmation.
    4. Re-run the exact DEVELOPMENT_EVIDENCE-style analysis (top1 lift vs
       pool, day-clustered bootstrap LB, p-value under the day-pool null)
       once at least 20 fresh prospective day-folds have accumulated
       (matching this repo's existing BINARY_OUTCOME_SET_PROTOCOL.
       minimum_calibration_slates convention elsewhere in sports/mlb/
       conditional_chain/protocol.py). Fewer than 20 is not a confirmation
       attempt, just a running tally.
    5. H_OVER_RANKER_V1 only advances past
       DEVELOPMENT_COMPLETE_FRESH_CONFIRMATION_REQUIRED if that fresh
       20+-fold result clears conventional significance (the development
       fold's own p~=0.0785 did not). Do not lower this bar after seeing
       the fresh data.
    6. Only once H_OVER_RANKER_V1 itself is confirmed does building a
       combo/dependence model over its recommendations become in scope --
       not before.
""".strip()

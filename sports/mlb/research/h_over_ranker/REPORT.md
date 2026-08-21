# H_OVER_RANKER_V1 — development report

**Status: `DEVELOPMENT_COMPLETE_FRESH_CONFIRMATION_REQUIRED`** — not validated. See "Why not validated" below.

## Data discipline

`TEST_STAMPS` (the 9-day block: `20260803`–`20260811`) was **never read** by any
code in this package except `data_windows.py` (defines it) and `manifest.py`
(documents the already-frozen historical result). This is enforced by
`sports/mlb/tests/test_h_over_ranker.py::test_no_module_reads_test_stamps`
(an AST-based static check, not a convention) and
`test_run_development_never_touches_test_stamps`.

All development below uses `DEVELOPMENT_STAMPS` = `DERIVE_STAMPS` (8 days) +
`SELECT_STAMPS` (8 days) = 14 distinct calendar days, of which **14 had at
least one H-OVER-eligible row** (2 of the 16 raw day-stamps had zero).

## 1. Eligibility, reconstructed exactly

`eligibility.FROZEN_H_BIAS = 0.0749739701851401` — `mean(Prediction - Actual)`
for `target == "H"` rows across `DERIVE_STAMPS` only.
`eligibility.eligible_rows_for_stamps(SELECT_STAMPS)` reproduces the exact
numbers H was originally selected on: **n=1432, hit rate=57.54%**
(`test_select_block_eligibility_reproduces_the_frozen_selection_numbers`).

## 2. Chronological, day-grouped development validation

`chronological_cv.expanding_day_folds(dates, min_train_days=6)` → 8
walk-forward folds (fold *k* trains on every date strictly before val date
*k*, validates on that one date alone; see `reports/fold_models_ranker_v1_candidate.json`
for the exact coefficients used in each fold). No row from a validation
date is ever in that fold's training set (tested).

## 3. Edge-shape test (disciplined, out-of-fold — not hand-picked)

Three logistic-regression shape families, fit per-fold on TRAIN only,
scored out-of-fold on Brier (`reports/edge_shape_comparison.csv`):

| shape | mean OOF Brier | quadratic coef sign consistency |
|---|---|---|
| quadratic | 0.24315 | 8/8 folds negative |
| log | 0.24323 | — |
| linear | 0.24329 | — |

The quadratic (inverted-U) coefficient is consistently negative across
every fold — a real, if very small, hint of "moderate edge beats extreme
edge" — but the margin over linear/log (0.00014, 0.00007) is not a robust
one. **Verdict: not a strong enough signal to build the frozen feature set
around**, confirmed directly in step 5.

## 4. Baseline audit (no fitting — existing/naive scores)

| baseline | top-1 hit rate | top-2 hit rate | lift vs. pool |
|---|---|---|---|
| current probability ranking (Poisson+RMSE formula) | 37.5% | 37.5% | **−19.7pp** |
| raw bias-corrected edge | 37.5% | 43.75% | **−19.7pp** |
| Q = edge / RMSE | 62.5% | 50.0% | +5.3pp |
| *(pool average, for reference)* | 57.2% | 57.2% | — |

The incumbent probability ranking and raw edge are **worse than picking
randomly from the eligible pool** — confirms the earlier session finding
(top-2-by-probability underperformed the rest of the H-OVER pool) with a
full walk-forward protocol instead of one ad hoc day. `Q` is a mild
improvement. This is exactly why `current probability ranking` is labeled
`INCREMENTAL_VALUE_NOT_ESTABLISHED` in the manifest.

## 5. Candidate ranker — every variant tried (`reports/all_tried_variants.csv`)

Single-feature ablations (C=1.0):

| feature alone | top-1 | top-2 |
|---|---|---|
| corrected_edge | 37.5% | 43.75% |
| **rmse** | **75.0%** | 62.5% |
| q_edge_over_rmse | 62.5% | 50.0% |
| log1p_history_rows | 50.0% | 62.5% |
| is_real_market | 50.0% | 56.25% |

`rmse` alone is the strongest single signal — stronger than raw edge or Q.

Full 6-feature model (incl. `corrected_edge_sq`) across a C sweep:

| C | top-1 | top-2 | mean OOF Brier |
|---|---|---|---|
| 1.0 | **100%** (8/8) | 81.25% | 0.242957 |
| 0.3 | 100% | 81.25% | 0.242965 |
| 0.1 | 87.5% | 87.5% | 0.242979 |
| 0.03 | 87.5% | 87.5% | 0.243003 |
| 0.01 | 87.5% | 81.25% | 0.243083 |
| 0.003 | 75.0% | 75.0% | 0.243371 |
| 0.001 | 75.0% | 81.25% | 0.243940 |

**The C=1.0 result (8/8 top-1) was rejected as the frozen configuration**,
even though it's numerically the best. Reasoning:

- Brier is essentially flat across C=1.0→0.01 (0.24296–0.24308) — it does
  not cleanly prefer C=1.0.
- Picking C=1.0 *because* it showed 8/8 would be tuning the regularization
  strength directly against the primary endpoint after seeing it — exactly
  what the development protocol prohibits.
- Under the honest null (each fold's top-1 pick wins at that day's own pool
  rate, no real skill), `P(8/8) ≈ 1.1%` but `P(≥7/8) ≈ 7.9%` — the more
  honest, less-cherry-picked number does not clear conventional
  significance.

**C=0.1 was selected instead** — the middle of the flat-Brier plateau, a
materially more conservative choice that still captures nearly all of the
lift (87.5% / 87.5% vs. 100% / 81.25%).

`corrected_edge_sq` (the quadratic term step 3 only weakly favored) was
then tested at C=0.1 and changed **nothing** (top-1/top-2/Brier identical
to 4+ decimals with or without it) — dropped per "prefer simple unless a
more complex model shows robust improvement."

`market_books` and `market_line_std` were tried and are **constant at 0.0**
for every H-target development row (verified against raw values, not a
code bug) — zero-variance, coefficient 0.0 in every fit, dropped.

## 6. Frozen result: `H_OVER_RANKER_V1`

5 features (`corrected_edge`, `rmse`, `q_edge_over_rmse`,
`log1p_history_rows`, `is_real_market`), `C=0.1`, z-score standardized,
logistic regression. Full coefficients in `manifest.py`.

| | H_OVER_RANKER_V1 | best baseline (Q) | pool |
|---|---|---|---|
| top-1 hit rate | **87.5%** | 62.5% | 57.2% |
| top-2 hit rate | **87.5%** | 50.0% | 57.2% |
| top-1 lift vs. pool | **+30.3pp** | +5.3pp | — |
| day-clustered one-sided 95% LB | **62.5%** | 37.5% | — |
| P(≥ observed \| day-pool-rate null) | **7.9%** | — | — |

## Why not validated

- **n=8 walk-forward folds.** `P(≥7/8) ≈ 7.9%` does not clear conventional
  significance (0.05). A real, promising signal — not a proven one.
- **Model/feature search happened on this same development data.** Every
  trial is recorded above and in `reports/`, including the ones rejected
  specifically for looking too good (C=1.0). The selection criterion
  (minimum OOF Brier, a secondary metric) was chosen to reduce, not
  eliminate, the risk of the frozen config being an artifact of this
  particular 14-day window.
- **`rmse` carries the most weight**, and is itself a per-player,
  per-target quantity the upstream pipeline computes fresh each day — its
  relationship to H-OVER reliability deserves independent scrutiny before
  being trusted at face value.

## Deferred (per the task's confirmation policy)

Combo/parlay construction and any same-game/dependence modeling over
H_OVER_RANKER_V1's picks. `certify_perfect_parlay`/`search_parlay_proof_frontier`-style
selection is explicitly **not** in scope until the ranker itself clears
prospective confirmation — see `manifest.PROSPECTIVE_CONFIRMATION_PROTOCOL`.

## Next step

Score every future slate's H-OVER-eligible pool with
`ranker.score_with_frozen_model()` using the frozen manifest coefficients
(never refit), record top-1/top-2 outcomes prospectively, and re-run this
exact analysis once 20+ fresh day-folds have accumulated — see
`manifest.PROSPECTIVE_CONFIRMATION_PROTOCOL` for the full procedure. No
parameter, threshold, or feature changes until then.

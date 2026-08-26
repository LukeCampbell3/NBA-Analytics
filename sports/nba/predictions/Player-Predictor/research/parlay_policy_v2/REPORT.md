# NBA Two-Leg Parlay Policy — Mechanism Report

## Scope

This adds a leakage-safe, gate-then-rank policy (`policy.py`) for two-leg
NBA parlay candidates, plus a unit test suite
(`sports/nba/tests/test_parlay_policy_v2.py`, 34 tests) that verifies the
policy's mechanics against synthetic fixtures.

**This report does not contain an NBA historical hit-rate backtest of the new
policy**, and no NBA-specific number below should be read as a measured NBA
hit rate. This repository does not currently have a settled NBA two-leg
parlay dataset carrying the full field set the policy requires per candidate:
`joint_sigma`, `joint_lcb`, an actual sportsbook SGP quote
(`actual_quote_decimal`), injury/role/support state, and a
`shared_failure_risk` score. `sports/nba/tests/`, `sports/nba/data/`, and
`sports/nba/validation/` were checked and no such file exists yet. MLB and
NFL both have a much larger `parlay_certification_v2` research program
(`sports/mlb/parlay_v2/`, `sports/nfl/research/parlay_certification_v2/`)
built up over many committed data-collection cycles; NBA does not yet have
the equivalent candidate-level logging, so nothing here can honestly cite an
NBA hit-rate figure from the new policy. The `sports/parlay_analysis.py`
correlation-factor heuristic already in production for NBA is a different,
simpler mechanism (static same-game/same-player/same-direction multipliers)
and is untouched by this change.

What *is* real, in the "Real-data backtests" section below: (a) the current
NBA strategy's own already-computed, real production validation numbers
(small sample), (b) a real, unsettled shadow-annotation of an actual NBA
board using this gate, and (c) the new policy's gate mechanism run against
real, settled MLB leg-level data — the closest genuine substitute this repo
has for "how does this mechanism do against real numbers," since MLB is the
only sport here with a committed leg-level settled dataset. That MLB result
has since been ported natively into MLB's own tree
(`sports/mlb/research/parlay_policy_v2/`) as an additive, shadow-only
package — see "MLB comparison" below for exactly what that does and does
not authorize.

## What the policy does

`evaluate_candidate` gates a two-leg candidate in this order, returning every
reason it failed (not just the first):

1. Schema check — every required field must be present, and none of the
   forbidden legacy cross-game path fields (`turn`, `accel_ratio`,
   `path_efficiency`) may appear on the candidate at all.
2. Leg count must equal the policy's configured `leg_count` (default 2).
3. Each leg's uncertainty-adjusted probability (`p - lambda*sigma`) must
   clear `min_leg_probability`.
4. The joint probability, after the same uncertainty penalty *and* an
   explicit `dependency_penalty`, must clear `min_joint_probability`.
5. `joint_lcb` must clear `min_joint_lcb`; `joint_sigma` must not exceed
   `max_joint_uncertainty`.
6. `shared_failure_risk` must not exceed `max_shared_failure_risk`, and
   `compatible_state_score` must clear `min_compatible_state_score` — this is
   the hook for a "these two legs tend to lose together" rule (e.g. two
   scoring-OVER legs sharing a slow-pace/blowout failure mode), not just a
   static per-market-pair blocklist.
7. `shift_risk` must not exceed `max_shift_risk` — a regime-health gate,
   independent of the point probability.
8. Execution-state gates: lineup confirmed, role stable, no material injury
   uncertainty, all legs in support (in-distribution for the underlying
   model), joint model reliable.
9. EV is computed from `actual_quote_decimal` — the real sportsbook same-game
   parlay quote — never from multiplying each leg's individually-quoted
   price. `actual_quote_ev <= min_actual_quote_ev` is a rejection, so EV
   alone can never rescue a candidate that failed the probability/state
   gates above it, and a synthetically-attractive multiplied price can never
   substitute for the real, more heavily vigged SGP quote.

Only candidates that pass every gate are ranked, and ranking (`rank_eligible`)
is EV first, then joint LCB, then joint probability — eligibility is decided
before EV is allowed to influence order at all.

Two supporting utilities are for tuning without leaking the future into the
past:

- `optimize_policy_grid` picks a grid cell by the Wilson lower bound of hit
  rate (subject to minimum sample size and coverage), not raw hit rate — a
  tiny 5/5 cell cannot win over a larger, still-strong cell just because
  100% > 75%.
- `date_blocked_walk_forward` retunes the grid using only strictly earlier
  dates before scoring each date, and `rolling_regime_gate` computes each
  slate's health flag from strictly prior outcomes only — both are covered
  by tests that recompute the gate with future outcomes blinded out and
  assert the decision doesn't change.

## Test coverage (34 tests, all passing)

- Probability/uncertainty math: `usable_probability` penalty and clipping,
  `naive_joint_probability`, and that `conservative_joint_probability` sits
  strictly below the naive independent product once a dependency penalty is
  applied.
- Gate rejections, one per failure reason: joint probability floor (even
  with an attractive quote), actual-quote EV (even with high probability),
  leg probability floor, leg count, shared-failure risk, state
  compatibility, shift risk, joint uncertainty, and each of the five
  execution-state flags individually.
- `actual_quote_ev` is verified to use only the real SGP quote, with an
  explicit case where the synthetic multiplied-leg price and the real quote
  disagree on sign.
- Schema validation: missing required fields, and the forbidden
  legacy-path-field check.
- Leakage safety: the rolling regime gate is recomputed with each slate's
  own (and every later) outcome zeroed out and must produce the same
  decision; `date_blocked_walk_forward` asserts every row's `train_end` is
  strictly before its own date.
- Grid selection prefers the Wilson lower bound over a small lucky sample.
- Pricing helpers (`american_to_decimal`, `decimal_to_break_even`) and
  `wilson_interval` edge cases.

Run with:

```
pip install --user numpy pandas pytest   # if not already present
python3 -m pytest sports/nba/tests/test_parlay_policy_v2.py -q
```

## Real-data backtests

### Current NBA strategy, real production numbers (small sample)

`real_data_summary_nba.py` reads the `parlay_validation` field already
embedded in the real, committed daily production exports at
`sports/nba/web/data/history/*.json` — computed by the live
`sports/parlay_analysis.py::evaluate_historical_parlays` (the same CONTROL
module NBA runs today) against real settled results, on a machine that also
had the underlying leg-level CSV (not committed here). This script only
aggregates that output; it computes nothing itself. Full output:
`reports/real_data_summary_nba.json`.

| snapshot | history rows | graded days | selected hit rate | baseline (all pairs) hit rate |
|---|---:|---:|---:|---:|
| 2026-04-26 | 326 | 3 | 1/3 = 33.3% | 2892/8822 = 32.8% |
| 2026-04-27/28/29 | 27 | 2 | 0/2 = 0% | 83/161 = 51.6% |
| 2026-04-30 | 326 | 3 | 1/3 = 33.3% | 4399/17265 = 25.5% |
| 2026-05-01/02 | 326 | 3 | 1/3 = 33.3% | 3391/11683 = 29.0% |
| 2026-05-26 | 27 | 2 | 0/2 = 0% | 211/486 = 43.4% |

**This is too small to conclude anything about the current NBA strategy's
real hit rate** — the largest graded sample across every committed snapshot
is 3 selected parlays. Rows are cumulative, restated views from different
runs, not independent samples to sum. This small-sample problem is itself
the strongest argument for the new policy's Wilson-lower-bound selection
criterion in `optimize_policy_grid` (Real-data backtests below shows what
that criterion actually buys on a real, much larger sample).

### Shadow-annotating a real NBA board (unsettled, prospective)

`shadow_annotate_board.py` is the concrete answer to "what would be needed"
item 1 below, done now instead of only documented: it reads a real,
committed NBA board export (`sports/nba/web/data/history/*.json`'s `plays`
array), builds every real cross-game 2-leg pair CONTROL's own gates
(`sports/parlay_analysis.py`, sport="nba") would consider, and runs the new
policy's eligibility gate on them. Every leg's probability comes from the
real `expected_win_rate`; the per-leg uncertainty penalty uses
`max(0, expected_win_rate - lcb_probability)` — the real, already-computed
gap between the point estimate and production's own lower-confidence-bound
estimate, used as a probability-scale proxy since the export's raw
`uncertainty_sigma` is in stat units (points/rebounds/assists), not
probability units, and is never fed into a probability-scale penalty. Price
uses the real product-of-decimal-odds convention (see the MLB section
below). **No `won` field exists anywhere in this output** — these plays
haven't been settled yet, and the script never fabricates one.

Running it against the most recent fully-populated snapshot
(`sports/nba/web/data/history/2026-05-26.json`, 12 real plays):

| | value |
|---|---:|
| Real candidate pairs (CONTROL's own gates) | 66 |
| New-policy-eligible pairs | 0 |
| Most common rejection reasons | `LEG_PROBABILITY`, `JOINT_LCB` |

Zero eligible is a real result, not a bug: this board's `lcb_probability`
values sit well below `expected_win_rate` for most plays, so the real
uncertainty-gap proxy is large enough that nothing clears the default
policy's floors. Two earlier snapshots (`2026-04-26/30`, `2026-05-01/02`)
produce **zero candidate pairs at all**, not zero-eligible — those exports
have `lcb_probability` and `market_side_price` as `None` for every play (an
earlier pipeline vintage didn't populate them), and the script skips a leg
rather than substituting a fabricated value. That gap in field coverage
across snapshot vintages is itself useful information for what to keep
logging consistently going forward.

Full output: `reports/shadow_annotate_2026-05-26.json`. This doesn't move
the "no NBA hit-rate claim" conclusion above — it's still unsettled — but it
means the day a settled-results file shows up, these records (or fresh ones
from this same script run daily) are already in the exact shape
`real_data_summary_nba.py`'s "what would be needed" list calls for.

### New policy gate vs. current strategy, real settled MLB legs

`real_data_backtest_mlb.py` runs the new policy's gate against
`sports/mlb/data/predictions/backtests/mlb_walk_forward_backtest_rows.csv`
(policy source `published_real_market`: 337 real, `market_source="real"`
settled legs across 11 dates — real model probability, real settled
win/loss/push, real American side price). This is MLB data, used here only
because MLB is the one sport in this repo with a committed leg-level settled
dataset; it is **not an NBA result**. Full output:
`reports/real_data_backtest_mlb.json`.

Only the probability-floor and real-quote-EV mechanism is exercised here —
this dataset doesn't log per-candidate `joint_sigma`, `shared_failure_risk`,
`compatible_state_score`, `shift_risk`, or lineup/role/injury/support state,
so those gates run at pass-through defaults and are **not** being tested for
real by this run (see the script's own `gates_not_exercised_...` output key).

| | n | hit rate | Wilson 95% |
|---|---:|---:|---:|
| Current MLB strategy (`sports/parlay_analysis.py`), top pick/day | 9 | 8/9 = 88.9% | wide (n=9) |
| Full real eligible 2-leg pool (CONTROL's own gates, ungated) | 5,012 | 2652/5012 = 52.9% | [51.5%, 54.3%] |
| **New policy gate**, every eligible pick (41.1% coverage) | 2,061 | 1230/2061 = **59.7%** | **[57.5%, 61.8%]** |
| New policy gate, top-EV pick/day (apples-to-apples with CONTROL) | 9 | 5/9 = 55.6% | wide (n=9) |

Reading this honestly:

- Applied broadly (not just one pick a day), the new gate selects 41% of the
  real eligible pool and clears the ungated baseline by ~7 points on a
  *much* tighter interval (n=2,061 vs n=9) — this is the real analog of the
  "robust filtering" lift the earlier synthetic-fixture design discussion
  assumed; here it's measured, not assumed.
- CONTROL's own top-1-pick-per-day already does very well on this slice
  (8/9), and the new policy's top-1-pick-per-day (5/9) does not beat it —
  but both are n=9, Wilson intervals nearly span the full [0,1] range, and
  this difference is not distinguishable from noise. Do not read 88.9% vs
  55.6% as "CONTROL beats the new policy" — read it as "neither single-pick
  comparison has enough data to say anything," which is exactly why the
  broader, tighter 2,061-pair comparison above is the more trustworthy row.
- Calibration check (real, not assumed): for the new-policy-selected subset,
  mean predicted joint probability was 0.604 against an actual hit rate of
  0.597 (Brier 0.237) — close, on this sample. This is the opposite finding
  from a naive-independence-is-~5%-too-optimistic claim; it is not evidence
  that dependency penalties are unnecessary in general (MLB's own
  `joint_position_builder_v2/reports/calibration_summary.json` shows the
  *broad* eligibility universe overconfident by 14.3 points versus its
  *narrow*, CONTROL-matching universe's calibration gap of 0.4 points), only
  that this particular real, narrow sample happened to calibrate well.
- `mean_actual_quote_ev` (0.90) uses the model's own probability estimate
  against the real decimal-price product — it is a modeled edge, not a
  guaranteed realized return, and should not be read as " +90% ROI."

### MLB comparison: what MLB's own equivalent research already found

Following this real-data result, the gate mechanism itself has since been
**ported and incorporated natively into MLB**, as its own additive,
shadow-only package:
`sports/mlb/research/parlay_policy_v2/` (mirrors this directory exactly —
`policy.py`, `real_data_backtest.py`, `REPORT.md`,
`sports/mlb/tests/test_parlay_policy_v2*.py`). It does not touch CONTROL or
either of MLB's existing V2 programs, and is not production-authorized —
see that package's own REPORT.md for the details and for why its
`INSUFFICIENT_EVIDENCE` conclusion still governs.

MLB has separately already built and run the direct, far more rigorous
analog of this NBA design (`sports/mlb/research/joint_position_builder_v2/`)
against real MLB data at much larger scale than anything above:

- `reports/calibration_summary.json`: on the **narrow** universe (matching
  CONTROL's own eligibility exactly), predicted-vs-actual calibration gap was
  0.37 points (1,470 pairs / 14 days). On the **broad** universe (adding
  UNDER-direction and negative-edge legs), the gap widened to 14.3 points
  (also 1,470 pairs / 14 days) — expanding the eligible universe made the
  model measurably overconfident.
- `reports/multi_target_broad_summary.json`: at full scale (619,191 priced
  pairs / 11 days), realized mean return was slightly negative
  (-0.98%, 90% bootstrap CI [-5.1%, +4.5%]) — modeled EV confidence was not
  realized.
- The program's own conclusion (`REPORT.md`): **`INSUFFICIENT_EVIDENCE`**,
  **`production_authorized = False`**, specifically because no real
  same-game SGP price coverage existed in the development window — the exact
  same real-price gap this NBA policy's schema (`actual_quote_decimal`) is
  built to enforce before promotion.
- Separately, MLB's live `PARLAY_CERTIFICATION_V2` production program
  (`sports/mlb/research/parlay_certification_v2/reports/program_alpha_ledger.json`)
  has recorded **zero real evidence rows** as of this repo's last update — a
  frozen world-gate bug made every real day abstain, and its alpha-spend was
  retired and re-frozen under a new prospective policy version with no real
  outcomes yet. Even MLB, with a much larger real-data foundation than NBA,
  has not yet cleared its own promotion bar for the joint-state upgrade.

**Bottom line for NBA:** the mechanism (gate-then-rank, real-quote EV,
Wilson-lower-bound tuning) measurably beats an ungated baseline on the one
real, large-enough sample available anywhere in this repo (MLB, 2,061
pairs), and is unit-tested end to end. But neither NBA's own real production
numbers (too small) nor MLB's own more mature research program (still
`INSUFFICIENT_EVIDENCE`, still no real same-game price data) support
promoting this to gate a live NBA board yet. The path is the same one MLB is
still walking: real candidate-level logging, a frozen development/holdout
split, then an independent prospective shadow block.

## What would be needed before this can gate a real NBA board

1. Candidate-level logging that captures the full `REQUIRED_SELECTION_FIELDS`
   set for every settled two-leg NBA parlay, including the actual sportsbook
   SGP quote (not a synthetic product of leg prices).
   `shadow_annotate_board.py` (above) now produces most of this shape daily
   from the real board, prospectively — what's still missing is (a) a
   settled `won` field once results land, (b) real
   `joint_sigma`/`shared_failure_risk`/`compatible_state_score`/`shift_risk`/
   lineup-role-injury-support state, none of which exist in the current
   export, and (c) a real same-game SGP quote in place of the
   product-of-decimal-odds proxy.
2. A development/holdout date split, so `optimize_policy_grid` and
   `date_blocked_walk_forward` can be run for real instead of on synthetic
   fixtures, and a shared-failure-family rule (e.g. same-direction,
   same-market pairs) can be learned from strictly prior history and frozen
   before being scored on a later block.
3. An independent prospective shadow block scored after the policy is frozen,
   per the promotion bar MLB and NFL already use in
   `sports/mlb/research/joint_position_builder_v2/REPORT.md` and
   `sports/nfl/research/parlay_certification_v2/` — hit-rate calibration and
   positive EV at actual quotes, not backtested hit rate alone.

None of the above exists yet for NBA. This report only claims the mechanism
is implemented and unit-tested; it does not claim any hit-rate result.

# NBA Two-Leg Parlay Policy — Mechanism Report

## Scope

This adds a leakage-safe, gate-then-rank policy (`policy.py`) for two-leg
NBA parlay candidates, plus a unit test suite
(`sports/nba/tests/test_parlay_policy_v2.py`, 34 tests) that verifies the
policy's mechanics against synthetic fixtures.

**This report does not contain an NBA historical backtest**, and no numbers
below should be read as measured NBA hit rates. This repository does not
currently have a settled NBA two-leg parlay dataset carrying the full field
set the policy requires per candidate: `joint_sigma`, `joint_lcb`, an actual
sportsbook SGP quote (`actual_quote_decimal`), injury/role/support state, and
a `shared_failure_risk` score. `sports/nba/tests/` and `sports/nba/data/`
were checked and no such file exists yet. MLB and NFL both have a much larger
`parlay_certification_v2` research program (`sports/mlb/parlay_v2/`,
`sports/nfl/research/parlay_certification_v2/`) built up over many committed
data-collection cycles; NBA does not yet have the equivalent candidate-level
logging, so nothing here can honestly cite an NBA win-rate figure. The
`sports/parlay_analysis.py` correlation-factor heuristic already in
production for NBA is a different, simpler mechanism (static same-game/
same-player/same-direction multipliers) and is untouched by this change.

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

## What would be needed before this can gate a real NBA board

1. Candidate-level logging that captures the full `REQUIRED_SELECTION_FIELDS`
   set for every settled two-leg NBA parlay, including the actual sportsbook
   SGP quote (not a synthetic product of leg prices).
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

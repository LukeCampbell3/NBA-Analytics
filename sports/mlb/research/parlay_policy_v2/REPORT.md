# MLB Two-Leg Parlay Policy (parlay_policy_v2) — Incorporation Report

## What this is, and what it is not

This package is a **port**, not a new invention: the same generic,
sport-agnostic gate-then-rank mechanism originally written and unit-tested
for NBA
(`sports/nba/predictions/Player-Predictor/research/parlay_policy_v2/`),
incorporated here because it was backtested for real against **this sport's
own settled data** and beat an ungated baseline (see "Real-data backtest"
below).

**This is not a replacement for, and does not compete with,
`sports/mlb/research/parlay_certification_v2/` or
`sports/mlb/research/joint_position_builder_v2/`.** Those are MLB's own,
far more heavily instrumented research programs for the same underlying
idea — world certificates, an anytime-valid reference monitor, a
multiple-testing alpha budget, an evidence store, a decision-record state
machine. This module has none of that machinery. It is a much simpler
probability-floor + real-quote-EV gate. It exists as an additive,
shadow-only, cross-sport-portable layer, not a rival production candidate.

**This module is not production-authorized.** Nothing here is imported by,
or imports, `sports/parlay_analysis.py` (CONTROL), `select_daily_parlay.py`,
or either of MLB's existing V2 programs. Wiring this into any path that
places or represents real wagers requires a separate, explicit decision —
this report documents evidence, not an authorization.

## Real-data backtest

`real_data_backtest.py` runs the gate against
`sports/mlb/data/predictions/backtests/mlb_walk_forward_backtest_rows.csv`,
policy source `published_real_market`: 337 real (`market_source="real"`)
settled legs across 11 dates, each with a real model probability, a real
settled win/loss/push result, and a real American side price. Full output:
`reports/real_data_backtest.json`.

Only the probability-floor and real-quote-EV mechanism is exercised — this
dataset does not log per-candidate `joint_sigma`, `shared_failure_risk`,
`compatible_state_score`, `shift_risk`, or lineup/role/injury/support state,
so those gates run at pass-through defaults (see the script's own
`gates_not_exercised_not_logged_in_this_data` output key).

| | n | hit rate | Wilson 95% |
|---|---:|---:|---:|
| Current MLB strategy (`sports/parlay_analysis.py`), top pick/day | 9 | 8/9 = 88.9% | wide (n=9) |
| Full real eligible 2-leg pool (CONTROL's own gates, ungated) | 5,012 | 2652/5012 = 52.9% | [51.5%, 54.3%] |
| **New policy gate**, every eligible pick (41.1% coverage) | 2,061 | 1230/2061 = **59.7%** | **[57.5%, 61.8%]** |
| New policy gate, top-EV pick/day (apples-to-apples with CONTROL) | 9 | 5/9 = 55.6% | wide (n=9) |

Applied broadly (not one pick a day), the gate selects 41% of CONTROL's own
real eligible pool and clears the ungated baseline's hit rate by ~7 points
on a much tighter interval (n=2,061 vs n=9). Calibration on the selected
subset: mean predicted joint probability 0.604 vs. actual hit rate 0.597
(Brier 0.237) — close, on this sample. The single-pick-per-day row (5/9)
does not beat CONTROL's own top pick (8/9), but both are n=9 and not
distinguishable from noise; the 2,061-pair row is the trustworthy one.

`mean_actual_quote_ev` (0.90) uses the model's own probability estimate
against the real decimal-price product — a modeled edge, not a guaranteed
realized return.

## How this relates to MLB's existing, more mature V2 research

This program has already run the direct, much more rigorous analog of this
idea against real MLB data at far larger scale, and its own conclusions
still govern:

- `joint_position_builder_v2/reports/calibration_summary.json`: narrow
  (CONTROL-matching) universe calibration gap 0.4pp; broad universe
  (adding UNDER-direction and negative-edge legs) 14.3pp — expanding
  eligibility measurably degrades calibration.
- `joint_position_builder_v2/reports/multi_target_broad_summary.json`: at
  619,191 priced pairs / 11 days, realized mean return was slightly
  negative (-0.98%, 90% CI [-5.1%, +4.5%]).
- `joint_position_builder_v2/REPORT.md`'s own conclusion:
  **`INSUFFICIENT_EVIDENCE`**, **`production_authorized = False`**,
  specifically for lack of real same-game SGP price coverage.
- `parlay_certification_v2/reports/program_alpha_ledger.json`: the live
  certification program has recorded **zero real evidence rows** — a frozen
  world-gate bug made every real day abstain; alpha-spend was retired and
  re-frozen under a new prospective policy version with no real outcomes
  yet.

**This module's real-data result (above) does not override any of that.**
It is a narrower, real, positive finding on a different (smaller, simpler)
slice of MLB data, using a different (simpler, weaker) mechanism. Treat it
as one more data point supporting the same direction MLB's own research is
already pointing — probability/EV gating measurably helps — not as a
reason to bypass the certified program's stricter bar. The certified
program's `INSUFFICIENT_EVIDENCE` verdict stands.

## What would still be needed before any promotion decision

Same requirements as the certified program already documents, unchanged by
this port:

1. Real same-game SGP price coverage (this module's `actual_quote_decimal`
   schema field exists specifically to refuse a synthetic substitute).
2. Per-candidate `joint_sigma`, `shared_failure_risk`,
   `compatible_state_score`, `shift_risk`, and lineup/role/injury/support
   state logged for real — none of the above real-data result exercised
   these gates.
3. A frozen development/holdout split (`optimize_policy_grid`,
   `date_blocked_walk_forward` are ready to use) followed by an independent
   prospective shadow block, per the same promotion bar
   `joint_position_builder_v2/REPORT.md` and the certified program's own
   freeze-readiness process already require.

## Test coverage

`sports/mlb/tests/test_parlay_policy_v2.py` (34 tests, ported unchanged from
the NBA mechanism suite) and
`sports/mlb/tests/test_parlay_policy_v2_real_data_backtest.py` (1 test,
asserts the real-data gate beats the real ungated baseline). Run with:

```
python3 -m pytest sports/mlb/tests/test_parlay_policy_v2.py sports/mlb/tests/test_parlay_policy_v2_real_data_backtest.py -q
```

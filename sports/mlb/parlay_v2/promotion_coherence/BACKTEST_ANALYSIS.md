# Pair-observation ledger backtest — honest findings

Run: `python3 -m sports.mlb.parlay_v2.promotion_coherence.backtest_pair_ledger`

## What the dataset is

`sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl`
is the largest **real settled** parlay-pair dataset this repo carries as
of this branch:

- **3,120 rows** across 4 slates (20260824–20260901)
- 100% settled
- Each row carries `predicted_joint_probability` (already-calibrated),
  `quoted_pair_price` (combined decimal), `actual_pair_return` (real +profit
  or −1), and per-leg outcomes.
- Composition: 2,841 cross-game + 279 same-game. Markets: R|TB (1,546),
  R|R (1,060), TB|TB (514).

This ledger is written by the frozen `PARLAY_POLICY_V2_PROSPECTIVE_003`
research policy. It is the joint-model's own ground truth on real graded
priced pairs — the most valuable dataset in the repo for testing the
`promotion_margin` rule.

## What the backtest does

For every settled pair-observation row it computes

    promotion_margin = predicted_joint_probability − (1 / quoted_pair_price)

(the deductions in the general promotion-margin identity default to 0.0
here — the pair ledger does not carry the additional signals the market-
disagreement / shared-failure / fragility deductions need). It then
sweeps a `min_promotion_margin` floor from −0.10 to +0.10 in 1 pp steps
and reports admitted-count, hit rate, and realized return per unit for
every floor, sliced by all pairs / cross-game / same-game / market pair
type.

## The honest read

The margin rule **reduces exposure to a broadly-losing pool** but does
**not** turn the pool profitable. Every floor in the sweep still yields
a negative total return per unit on this ledger. This is the most
important finding on this branch, and the one the promotion-coherence
proposal should not overclaim past.

### ALL_SETTLED_PAIRS (3,120 rows)

| floor | admitted | share | hit rate | total return / unit | mean return / unit |
|:-----:|---------:|------:|---------:|--------------------:|-------------------:|
| −0.10 (baseline) | 2,631 | 84.3% | 7.6% | −1,647.44 | −0.626 |
|  0.00 |   846 | 27.1% | 6.7% |   −509.26 | −0.602 |
| +0.05 |   117 |  3.7% | 5.1% |    −73.21 | −0.626 |
| +0.10 |     1 |  0.0% | 0.0% |     −1.00 | −1.000 |

The margin rule cuts total unit-loss from about −1,647 to about −73 at
its most exposure-reducing floor (+0.05). Fewer bets, less bleed. But
the mean return per admitted pair barely moves — the ledger is
systematically negative-EV across the entire margin spectrum, and the
gate is filtering out priced junk rather than isolating a positive
subgroup.

### CROSS_GAME vs SAME_GAME

- **Cross-game** looks the same shape as ALL, since cross-game is 91% of
  the ledger.
- **Same-game** has zero rows above margin 0.0 — every single same-game
  pair has predicted joint below break-even. The margin rule alone
  would abstain on 100% of same-game pairs on this ledger. That is
  data-driven support for the proposal's Item 4 (parlay-specific shared-
  failure penalties): same-game parlays deserve their own, tougher
  gate; treating them as "two independent legs" is exactly what the
  independence assumption on this ledger already does, and it does not
  survive settlement.

### Per-market

- **R|TB**: best floor +0.04, admits 83 pairs, total return −47.9 (vs
  baseline −816). Big loss reduction, still net negative.
- **R|R**: best floor is actually _negative_ (−0.04): the +0.0 and above
  region is dominated by 0% hit-rate cells at low admission — this
  market carries very little signal at the margin the joint model
  measures.
- **TB|TB**: best floor +0.05, admits 60 pairs, total return −24.2 (vs
  baseline −337). The cleanest loss-reduction story, but still not
  profitable.

## What this evidence supports, exactly

**Supported by this backtest:**

- The margin rule reduces total publication of negative-EV pairs by a
  large multiple on the ALL slice (loss cut from ~−1,650 to ~−73 at
  floor +0.05).
- Same-game pairs, on this ledger, are 100% below break-even under the
  frozen joint model. This is direct evidence that same-game parlays
  need their own shared-failure penalty, not the general-purpose margin
  rule alone.
- The Sept 2 overlay-authority regression (`test_sept2_recovered_richest_
  reproduces_the_coherence_gap`) is orthogonal and passes: the coherent
  gate correctly rejects the recorded losing publication for exactly
  the three reasons the payload's own overlay already gave.

**NOT supported by this backtest:**

- The claim "the coherent promotion rule makes the parlay pool
  profitable." No floor in the sweep produces positive returns. The
  `test_real_ledger_margin_gate_reduces_exposure_but_stays_negative`
  regression test pins this behavior loudly — if a future ledger update
  changes it, that test fails and this document gets rewritten.
- The claim "the margin rule alone is sufficient." The market-
  disagreement, fragility, and per-leg penalties from the proposal
  remain necessary; this ledger has no way to test them because the
  required per-leg model/market probabilities and starter/game-script
  signals are not persisted in it.

## Synthetic ledger — grown from settled singles

`synthesize_pairs.py` cross-joins settled singles from the 9,051-row
singles calibration ledger (25 slates, back to 2026-04-29) into
**18,020 synthetic cross-game pair observations**, capped at 800 per
slate and 6 singles per game so no single dominates. Every row is
flagged `is_synthetic: true` and stored in a separate file. This is
exploratory evidence, never production data — the flags exist so a
reader cannot accidentally treat it otherwise.

### SYNTHETIC ALL_SETTLED_PAIRS (18,020 rows)

| floor | admitted | share | hit rate | total return / unit | mean return / unit |
|:-----:|---------:|------:|---------:|--------------------:|-------------------:|
| −0.10 (baseline) | 16,596 | 92.1% | 26.1% | −1,522.78 | −0.092 |
| −0.02 | 11,442 | 63.5% | 25.9% |   −89.52 | −0.008 |
| −0.01 | 10,473 | 58.1% | 26.1% |  **+144.79** | **+0.014** |
|  0.00 |  9,477 | 52.6% | 26.3% |  +362.02 | +0.038 |
| +0.05 |  5,038 | 28.0% | 27.4% | +1,124.39 | +0.223 |
| +0.06 |  4,479 | 24.9% | 27.6% | **+1,196.45** | +0.267 |
| +0.10 |  2,842 | 15.8% | 26.3% | +1,142.04 | +0.402 |

`strict_dominance_over_baseline` flag fires cleanly on this pool at
floor +0.06.

### Why the two ledgers disagree — read carefully

The real pair ledger's average hit rate is 7–8%. The synthetic
ledger's is 26%. That gap is real, not a bug, and it is important:

- The real pair-observation ledger is the pool the frozen
  `PARLAY_POLICY_V2_PROSPECTIVE_003` policy actually scored — a narrow,
  deliberately-selected candidate universe with predicted joint
  probabilities [0.14, 0.29].
- The synthetic ledger is the pool of "any two cross-game singles that
  the production system had a real quote and probability for," which
  covers a much wider quality range.

Two interpretations of the gap are consistent with the data and both
are honest:

1. **The production candidate selector is systematically choosing
   worse pairs than the broader admitted-singles pool would offer.**
   The margin rule works fine on a broader pool but the upstream
   selector is discarding the good pairs.
2. **The independence-assumed synthetic joint is optimistic** relative
   to a proper joint model that accounts for shared-game context. The
   hit rate on synthetic pairs reflects singles hit rates well because
   the singles are from different games and are actually close to
   independent, but the joint model in the real pair ledger may be
   correctly-conservative about correlated failure.

Interpretation (1) argues for revisiting the upstream candidate
selector; interpretation (2) argues that the real ledger's pessimism
is the calibrated truth. This backtest cannot distinguish them from
the current data, and the honest report says so.

### What the synthetic ledger actually supports

- **The promotion-margin rule has real predictive value on a broader-
  than-production pool.** At floor +0.06 on the synthetic ledger the
  strict-dominance flag fires with a +1,196-unit-return improvement
  over baseline across 4,479 admitted pairs.
- **The rule alone does not automatically translate to production
  ROI**, because the production candidate pool is systematically
  narrower and worse-performing than the synthetic pool. Adopting the
  margin rule on the current production pool would reduce exposure
  (real ledger evidence, above) but not turn the pool positive.
- **The full promotion-coherence proposal is right to layer several
  signals**: market-disagreement, fragility, per-leg floors, shared-
  failure. No single margin rule can rescue a systematically-
  negative-EV upstream pool.

Both findings are pinned by regression tests:

- `test_real_ledger_margin_gate_reduces_exposure_but_stays_negative`
- `test_synthetic_ledger_backtest_flips_positive_above_zero_floor`

If either flips direction on a future run, the corresponding test
fails loudly and this document gets rewritten.

## Resolution — beta calibration on the real ledger

`pair_ledger_calibration.py` fits a 2-parameter beta calibrator
(logit-space slope + intercept) on the real pair ledger's
`predicted_joint_probability` → `both_win` pairs, then
`PromotionConfidenceComponents` optionally passes the raw joint
through it before computing the margin.

Result — headline table now reads:

| pool | rows | hit rate | mean per-decile calibration gap (raw → calibrated) |
|:---|---:|---:|---:|
| Real pair ledger | 3,120 | 8.2% | +0.1200 → **+0.0114** |
| Synthetic pool  | 18,020 | 28.1% | +0.0180 → −0.1479 (over-corrected; expected) |
| Δ (real − synth) | | | +0.102 → +0.159 |

Under **leave-one-slate-out** cross-validation (fit on 3 slates,
scored on the held-out 1), the picture is the same:

| fold | held-out slate | raw gap | calibrated gap |
|:---|---|---:|---:|
| 1 | 20260824 | +0.132 | +0.015 |
| 2 | 20260825 | +0.142 | +0.033 |
| 3 | 20260826 | +0.117 | −0.004 |
| 4 | 20260901 | +0.085 | −0.045 |
| mean | | **+0.119** | **−0.0001** |

**The real-pool miscalibration is resolved.** The calibrator over-
corrects on the synthetic pool — expected, since the two pools have
different predicted-joint distributions and the calibrator was fitted
on the real one.

A second finding falls out of the fit: the calibrator's global slope
is **−1.58** (with intercept −4.70). A negative slope means the raw
joint model's confidence has *negative* correlation with actual
outcome within the range it operates ([0.14, 0.29]). That is a
consequential structural finding about the frozen production joint
model, pinned by
`test_global_calibrator_slope_indicates_narrow_range_correction` and
worth an independent investigation of its own.

### How to consume the calibrator

`decide_coherent_promotion(payload, joint_calibrator=cal)` threads the
calibrator through; `PromotionConfidenceComponents.calibrated_joint_
probability` then carries the calibrated value, and `promotion_margin`
uses it. Deletion of the calibrator argument returns the earlier
behavior exactly, so this is opt-in with zero effect on existing call
paths.

To rebuild the calibrator:

```
python3 -m sports.mlb.parlay_v2.promotion_coherence.pair_ledger_calibration
```

To rerun the pool-gap investigation before-and-after:

```
python3 -m sports.mlb.parlay_v2.promotion_coherence.investigate_pool_gap --with-calibrator
```

## What to do next

1. **Ship the shadow layer.** It's a clean win on the Sept 2
   regression, reduces exposure on 3,120 real settled pairs,
   demonstrates real predictive value on 18,020 synthetic cross-game
   pairs, and now carries a calibrator that resolves the +12 pp
   miscalibration under leave-one-slate-out.
2. **Investigate the negative-slope finding.** The fitted calibrator
   has slope −1.58: the raw joint model's confidence is inverted
   relative to actual outcome inside its operating range. That is a
   deeper problem than miscalibration, and it deserves its own
   investigation (not addressable by any calibrator).
3. **Recompute the calibrator on new slates as they land.** As the
   real ledger grows past 10 slates
   (`ledger_maturity.py` reports readiness), the calibrator's OOS
   confidence widens and the fit becomes actionable for a production
   flip.

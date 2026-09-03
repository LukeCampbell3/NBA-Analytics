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

## What to do next

1. **Ship the shadow layer as it stands.** It's a clean win on the Sept 2
   regression, and it demonstrably reduces exposure on 3,120 real
   settled pairs.
2. **Do not force this data to promote the rule.** The current backtest
   supports a "less-bad publication" claim, nothing more. The next real
   improvement will come from adding the missing signals — a per-leg
   probability floor (available on the normal-parlay overlay path), a
   market-disagreement penalty (needs no-vig market probability captured
   at decision time), and a same-game-specific shared-failure penalty
   (needs game-script, total-line, and bullpen signals).
3. **Keep the pair ledger growing.** Four slates is enough to see
   direction but not enough for a serious calibration decision. As the
   ledger extends past 10 slates the `strict_dominance_over_baseline`
   diagnostic in the report becomes worth acting on, not just noting.

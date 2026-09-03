# MLB parlay promotion-coherence shadow

This subpackage is an additive, non-invasive layer that lets us **prove**
the tighter promotion rule improves the system before we flip any live
publication path.

## Why it exists

On 2026-09-02 the normal parlay's own conservative `public_quality_
overlay` returned `ABSTAIN` with three blocking reasons (both leg
probabilities below 70%, joint below 50%). The payload's live
`parlays.action` was `ACT` and the parlay was published anyway. Same
pattern on the following slate. The overlay already exists inside the
payload; nothing enforces it.

## What the shadow does

For every recorded `daily_predictions.json`
(`sports/mlb/web/data/history/runs/<date>/<run>/…` plus the live one),
`shadow_replay.py` computes a parallel `CoherentPromotionDecision`:

- `ACT` only when **all** of these agree
  1. slate is eligible,
  2. the payload's own `public_quality_overlay.action == "ACT"`,
  3. every leg clears `min_leg_probability` (default 0.70),
  4. the calibrated joint clears `min_joint_probability` (default 0.50),
  5. the `promotion_margin` clears its floor.
- The full explainable margin:

  ```
  promotion_margin = calibrated_joint_probability
                     - uncertainty_deduction
                     - market_disagreement_deduction
                     - shared_failure_deduction
                     - fragility_deduction
                     - break_even_probability
  ```

- Each deduction defaults to `0.0`, so today the margin identity is
  `joint - break_even`. Tuners supply concrete penalties later in shadow
  (the tuning items in the promotion-coherence proposal).

Where the payload's selected legs are in the settlements index
(`sports/mlb/data/predictions/unified/historical_settlements.json`), the
report joins them and reports realized return per unit for both the
`live` and `coherent` paths -- so any claim that the shadow "improves the
system" is a claim you can check against real graded outcomes, not a
narrative.

## Guardrails

- Nothing here imports from a live selector or writes to any published
  payload. Delete this whole directory and the live pipeline is
  byte-identical.
- All decisions are deterministic given the payload and thresholds.
- `promotion_confidence` is bookkeeping, never a competing model. The
  only new numbers are subtractions the caller explicitly supplies.

## Running the shadow report

From the repo root:

```
python -m sports.mlb.parlay_v2.promotion_coherence.shadow_replay
```

Writes `sports/mlb/parlay_v2/promotion_coherence/reports/latest_shadow_report.json`.
Console output shows counts of live-ACT/coherent-ABSTAIN divergences and
realized return per unit for both paths on graded parlays.

Threshold overrides on the CLI (`--min-leg-probability`,
`--min-joint-probability`, `--min-promotion-margin`, …) let you scan
tuning candidates without touching this code.

## Path to production

1. **Shadow** (this state): recorded slates only. Report the divergences
   and grow the graded-slate count until it supports a decision.
2. **Enforce as final publication authority** for normal parlays: the
   only live-code change is one line where `parlays.action` is written,
   guarded by the coherent decision.
3. Wire the concrete deductions (market-disagreement, shared-failure,
   fragility) into `PromotionPenalties` as more graded data accumulates.

## Next-steps status (from the promotion-coherence proposal)

1. **Grow the pair ledger past 10 slates** — `synthesize_pairs.py`
   cross-joins settled singles into 18,020 synthetic cross-game pair
   observations across 25 slates (see `BACKTEST_ANALYSIS.md` for the
   full read). The synthetic ledger is not the real production
   candidate universe, and every row is flagged `is_synthetic: true`
   so no consumer can accidentally mix it with real evidence, but it
   raises `strict_dominance_over_baseline` to decision-quality
   confidence on the broader-than-production pool.
2. **Per-leg model probability + no-vig market probability capture** —
   `pair_schema_v2.py` defines the additive v2 pair-observation shape
   and `market_disagreement_deduction()`. The live pair-ingest still
   writes v1 rows (test regression proves it), so the deduction is
   currently 0.0 on every real row. The synthetic ledger already
   populates the per-leg model probabilities; once no-vig market
   probability capture lands upstream, market-disagreement becomes
   a real signal without any further code change here.
3. **Same-game shared-failure penalty as a first-class deduction** —
   `same_game_penalty.py` defines `SameGamePenaltyProfile` and
   `same_game_shared_failure_deduction()`. Wired into
   `PromotionPenalties.from_pair_row`, and reported in the backtest
   as a side-by-side slice so the effect is visible. Defaults are
   conservative-but-real, grounded in the ledger evidence
   (100% same-game below break-even in the real pair ledger).

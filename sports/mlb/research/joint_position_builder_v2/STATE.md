# STATE — mission checkpoint (multi-target generalization)

Continuation of the JOINT_POSITION_BUILDER_V2 research question under the
"resolve to PROVEN or FALSIFIED/INFORMATION-LIMITED" mission. This
checkpoint covers Phase 1 (audit) and Phase 2-3 (bottleneck + first
hypothesis test) of the mission's research loop, scoped to one concrete
question: **does restricting to the H target explain the prior
"INSUFFICIENT_EVIDENCE_NO_REAL_PRICE_COVERAGE" conclusion, and if that
restriction is lifted, does real evidence support a parlay edge?**

## Phase 1 — Audit finding: the prior "zero price coverage" conclusion was H-specific, not universal

`manifest.CONCLUSION_REASONING` (frozen at the end of the prior development
pass) states real H-target market prices are 0% covered in
`DEVELOPMENT_STAMPS`. That is correct **for H specifically**, but the
mission explicitly warns against assuming "H-OVER only" — auditing the
other targets in the same daily pools shows real price coverage is *not*
uniformly zero:

| target | real price coverage in DEVELOPMENT_STAMPS |
|---|---|
| H   | 0% |
| RBI | 0% |
| ERA | 0% |
| K   | ~10% (excluded here — too thin to trust) |
| ER  | ~10% (excluded here — too thin to trust) |
| **R**   | **57% (2137/3726)** |
| **TB**  | **58% (2161/3726)** |
| **HR**  | **53% (2170/4081)** |

**BOTTLENECK (category: ACTION COVERAGE, revised):** the H-only scoping of
the prior pass, not a genuine absence of priced pregame markets. Built
`multi_target_universe.py` to generalize `observation_universe.py` to
R/TB/HR (both directions, "broad" mode — the mission's stated default to
test), reusing the frozen marginal model
(`h_over_ranker.baselines.probability_score`) and computing a separate
frozen DERIVE-only bias per target (`frozen_bias("R")=+0.0793`,
`frozen_bias("TB")=+0.0177`, `frozen_bias("HR")=+0.0646` — same
methodology as the existing frozen `H_BIAS`, just generalized).

Result: **3623 real, action-eligible (priced, in-support) rows across 12
of the 16 DEVELOPMENT_STAMPS days** — a real, non-degenerate action
universe, vs. 0 for H alone. This closes the literal "no price data" gap
that blocked the prior conclusion.

## Bug found and fixed while generalizing: leg-ID collision in `pairs.py`

`enumerate_candidate_pairs` built leg IDs as `player|direction|market_line`.
Once rows can come from more than one target, a player's R row and TB row
(or similar) can coincidentally share direction+market_line and collide
into one leg ID, which `outcome_worlds.build_world_distribution` correctly
rejects (`ValueError: candidate IDs must be unique`). Fixed by adding
`target` into the leg-ID string. **Verified**: `test_joint_position_builder_v2.py`
(14 tests, including the two tests that hand-construct leg dicts) still
passes unchanged after the fix — those tests already carried a `"target"`
key in their row fixtures. `test_h_over_ranker.py` (20 tests, untouched
module) also still passes. No other file needed changes; this generalizes
cleanly because `pairs.py`/`risk_gate.py`/`calibration_check.py` were
already target-agnostic by construction (they operate on generic
probability/price/support columns).

## Second bug found and fixed: doubleheader leg-ID collision in `pairs.py`

Running the real multi-target backtest (below) surfaced a second, distinct
`ValueError: candidate IDs must be unique` even after the target-aware fix
above. Root cause: on a doubleheader day, the same player can appear with
an *identical* `target|direction|market_line` in two different games (both
games' lines land on the same number) -- e.g. Max Muncy, `R|UNDER|0.5`,
once in `game_id=824990` and once in `game_id=823936` on 2026-06-19. These
are genuinely two different legs (different games, different outcomes),
not the same leg twice. Fixed by adding `game_id` into the leg-ID string
alongside `target`. Re-verified: `test_joint_position_builder_v2.py` (14
tests) still passes after this second fix.

## Phase 2-3 — Bottleneck: MARGINAL MODEL overconfidence, not the joint/pair mechanism

Ran the real 2-leg pair backtest (`multi_target_backtest.py`, broad mode,
all pair classes, DEVELOPMENT_STAMPS only, same walk-forward
calibration-warmup discipline as `ablation.run_variant`). Full committed
run (`reports/multi_target_broad_summary.json`, `reports/
multi_target_broad_day_class_summary.csv`): **619,191 evaluated priced
pairs across 11 of 12 action-eligible days** (the 12th day, 20260429, had
only one game's worth of eligible legs, so every pair that day was
same-game and got no synthesized price — expected behavior, not a bug;
see `pairs.py`'s D_S convention).

An earlier exploratory pass (kept only in this file's git history, not
committed as code) had restricted each day to its top ~25 legs by model
probability before pairing, giving a much smaller n=3580. That restriction
is now understood to have been itself a source of bias — see the
"effective sample size" note below — so the **full, uncapped universe run
below supersedes it** as the checkpoint's real evidence; the two are not
in conflict, they are measuring different (and now-explained) things.

**Critical distinction this pass makes explicit that the prior H-only
pass's raw `mean_joint_ev` number did not:** `joint_ev` (`p_joint * D_S -
1`) is a **model-confidence** figure — it says what the frozen marginal
model *believes* its edge is, using its own probability estimate. It is
NOT the realized backtest return. Reporting it alone invites exactly the
kind of "too good" false positive this session's discipline has flagged
before (the H_OVER_RANKER_V1 8/8 scare). So this pass computes both, side
by side, on the full universe:

| pair class | n | hit rate (both win) | mean p_joint (model) | mean market-implied p (1/D_S) | overconfidence vs. actual | **mean_joint_ev (model belief)** | **mean REALIZED return** | day-clustered 90% CI |
|---|---|---|---|---|---|---|---|---|
| ++ | 233,721 | 0.2842 | 0.3905 | 0.2767 | +0.1063 | +0.6745 | **+0.0610** | [-0.016, +0.147] |
| +- | 288,530 | 0.2979 | 0.3219 | 0.3067 | +0.0240 | +0.1158 | -0.0259 | [-0.061, +0.019] |
| -- | 96,940  | 0.3061 | 0.2700 | 0.3427 | **-0.0360** | -0.2056 | **-0.1324** | **[-0.199, -0.038]** |
| **overall** | 619,191 | 0.2940 | 0.3397 | 0.3010 | +0.0457 | +0.2764 | -0.0098 | [-0.051, +0.045] |

**HYPOTHESIS tested:** "the frozen marginal `probability_score` model,
calibrated/frozen on H-OVER data, generalizes cleanly to R/TB/HR broad
(both-direction) data." **RESULT: partially rejected, more nuanced than
the smaller-sample pass suggested.** The model IS measurably overconfident
on broad multi-target data (+4.6pp overall, worse in the ++ class at
+10.6pp) — the direction of the earlier finding holds — but the magnitude
on the full universe (+4.6 to +10.6pp) is much smaller than the +20.5pp
seen in the top-25-by-probability exploratory subset. That gap is itself
informative: **restricting to the highest-model-probability legs
concentrates the model's worst overconfidence** (a familiar ML pattern —
predictions nearest 0/1 are typically least well calibrated), so a "select
the model's most confident legs" policy would be walking directly into the
bottleneck rather than around it. This matters directly for Phase 9
(selective action policy) — "rank by raw model confidence" is one of the
mission's explicitly-flagged assumptions **not** to make, and this is
concrete evidence for why not.

**Useful negative control:** the `--` class (both legs individually -EV by
the market) shows a REALIZED return of -13.2%, with a day-clustered 90% CI
that does **not** cross zero ([-0.199, -0.038]) — the one statistically
clear result in this pass. This is reassuring for the pipeline's basic
validity (the class the market itself prices worst does in fact perform
worst, in the expected direction, with real signal) even though it is not
by itself evidence for the policy question (nobody is proposing to bet the
`--` class).

### Statistical caution: effective sample size and CIs

Each action-eligible day has on the order of 300-450 real legs (not ~25 —
that was an artifact of the smaller exploratory pass's own top-K
restriction, corrected here), producing tens of thousands of pairs per day
that are **not independent draws** — they share legs, so C(L,2) pairs from
L legs carries at most ~L independent bits of information, not C(L,2). The
only statistically defensible resampling unit is the **day** (11 evaluated
days), not the pair. All CIs in the table above are day-clustered bootstrap
(resample dates, not rows; 3000 resamples) for exactly this reason.

**Every CI in the table crosses zero except the `--` class.** With only 11
archived DEVELOPMENT days, there is not enough data to distinguish a real
edge from noise in either direction for the ++ , +-, or overall slices —
including the ++ class's positive-leaning point estimate (+6.1%), which is
plausible but not established.

**Multiplicity caution on the "value subset" filter** (`p_joint >
market_implied_p`, i.e. pairs where the model disagrees with the market in
the profitable direction): n=399,772, realized return +4.9%, day-clustered
90% CI [-0.018, +0.130] — still crosses zero, and this filter was
constructed *during* this exact exploratory pass, after seeing the raw
pair-class results, not predeclared before looking at DEVELOPMENT
outcomes. Per this repo's established discipline (reject best-looking-
after-search results as the frozen choice; see `h_over_ranker`'s C=1.0
rejection), this is reported as a **hypothesis for a future frozen/
SELECT-confirmed rule**, not as confirmed evidence of edge.

## Decision at this checkpoint

- Multi-target real price coverage: **confirmed real** (contradicts and
  supersedes the "no real price coverage" half of the prior manifest
  conclusion — `manifest.py` updated accordingly).
- Real, held-out-in-development-window evidence of a positive parlay edge
  under the CURRENT frozen marginal model + independence joint mechanism:
  **not established at 90% confidence for any actionable slice** — the
  `++` class and the market-disagreement "value subset" both lean positive
  (+6.1%, +4.9%) but their CIs cross zero at 11 days of data; only the
  non-actionable `--` class reaches significance, and in the expected
  (negative) direction, which is a validity check rather than a finding.
- This is `INSUFFICIENT_EVIDENCE` in the *same* sense as before, but now
  for a different, better-grounded reason: not "no priced markets exist"
  (false, corrected here) but "the marginal probability model is
  measurably overconfident outside the narrow H-OVER slice it was frozen
  against (worse for the model's own most-confident legs specifically),
  and even the most promising-looking slices don't clear a properly
  day-clustered significance bar at only 11 days of data."
- **`PRODUCTION_AUTHORIZED` stays `False`.** Not touched.
- Neither terminal mission state
  (`DAILY_PARLAY_POLICY_PROSPECTIVELY_SUPPORTED` /
  `DAILY_PARLAY_POLICY_NOT_SUPPORTED_BY_AVAILABLE_INFORMATION`) is reached
  by this checkpoint. Reaching either honestly requires either (a) a
  properly calibrated marginal model for the broad multi-target state
  (a MARGINAL MODEL fix, analogous to the `h_over_ranker` bias-correction
  work but for R/TB/HR broad, itself requiring a DERIVE/SELECT/TEST split
  *within* this newly-available multi-target data, which at only 16 real
  days total is too little to safely subdivide three ways again without
  running out of usable days), or (b) new real days accumulating beyond
  the existing archive (still frozen at 25 total day-stamps; TEST_STAMPS
  remains retired and unread by any code in this package).

## Next (per mission's bottleneck-driven loop)

1. Do not chase the "value subset" further inside this same DEVELOPMENT
   data — that would be the exact multiplicity violation flagged above.
2. The correct next MARGINAL MODEL experiment is a proper walk-forward
   recalibration (isotonic or Platt-style, or a per-target bias term like
   `h_over_ranker`'s frozen `H_BIAS`) fit *only* on a DERIVE-equivalent
   slice of the multi-target data, evaluated on a held-out slice, before
   ever touching pair-level joint EV again — mirrors exactly what
   `h_over_ranker` already did for H-OVER, generalized to R/TB/HR broad.
   Not attempted in this checkpoint; flagged as the next hypothesis.
3. Explicitly do NOT build a "rank by raw model confidence" selective
   action policy — this checkpoint's own evidence (overconfidence
   concentrates in the model's highest-probability legs) argues directly
   against it. Any future selective-risk gate should rank by something
   that degrades gracefully with model confidence (e.g. joint_ev_lcb,
   which already shrinks by uncertainty) rather than raw probability.
4. Given only 16 usable multi-target days exist, a 3-way split (as strict
   as `h_over_ranker`'s 8/8/9) is likely too thin to be trustworthy here;
   this itself is worth reporting as a DATA-category bottleneck if it
   can't be resolved before the next checkpoint.

## Generated artifacts (this checkpoint)

- `reports/multi_target_broad_summary.json` — full-universe summary (the
  table above), committed.
- `reports/multi_target_broad_day_class_summary.csv` — per-day x
  pair-class breakdown (33 rows), committed.
- `reports/multi_target_broad_pairs_sample.csv` — a reproducible
  stratified sample (200 pairs/class, seed=0) of the full pair-level
  output, committed for spot-checking.
- `reports/multi_target_broad_pairs.csv` — the FULL pair-level output
  (619,191 rows, ~215MB). **Not committed** (repo-impractical size) — it
  is combinatorially large by construction (C(L,2) pairs from ~300-450
  legs/day) and fully reproducible by re-running
  `python3 -m sports.mlb.research.joint_position_builder_v2.multi_target_backtest`.
  Excluded via `.gitignore`, not silently dropped.

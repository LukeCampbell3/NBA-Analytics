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
calibration-warmup discipline as `ablation.run_variant`) — 3580
evaluated priced pairs across 12 days.

**Critical distinction this pass makes explicit that the prior pass's raw
`mean_joint_ev` number did not:** `joint_ev` (`p_joint * D_S - 1`) is a
**model-confidence** figure — it says what the frozen marginal model
*believes* its edge is, using its own probability estimate. It is NOT the
realized backtest return. Reporting it alone, as the raw ablation output
does, invites exactly the kind of "too good" false positive this session's
discipline has flagged before (the H_OVER_RANKER_V1 8/8 scare). So this
pass computes both, side by side:

| pair class | n | hit rate (both win) | mean p_joint (model) | mean market-implied p (1/D_S) | **mean_joint_ev (model belief)** | **mean REALIZED return** |
|---|---|---|---|---|---|---|
| ++ | 2294 | 0.4648 | 0.6698 | 0.4677 | **+0.5134** | **-0.0595** |
| +- | 508  | 0.4452 | 0.6252 | (n/a, mixed) | +0.1502 | -0.3023 |
| -- | 124  | 0.5357 | 0.5371 | (n/a, mixed) | -0.2446 | -0.5986 |
| **overall** | 3580 | — | — | — | — | **-0.1151** |

**HYPOTHESIS tested:** "the frozen marginal `probability_score` model,
calibrated/frozen on H-OVER data, generalizes cleanly to R/TB/HR broad
(both-direction) data." **RESULT: rejected.** The model's own `p_joint`
(0.6698 for the ++ class) is ~20.5 percentage points above both the
**actual** both-win rate (0.4648) *and* the **market-implied** joint
probability (0.4677, i.e. `1/D_S`, which for cross-game pairs is exactly
the product of two independently-priced legs' implied probabilities). The
market's own implied probability is essentially dead-on calibrated against
reality here (gap -0.0029); the model's is not (gap +0.2051). This
isolates the bottleneck cleanly: **the marginal model overconfidence
problem this session diagnosed and partially addressed for H-OVER in
`h_over_ranker` (Phase-4/5 of the earlier turns) reappears, worse, when
the same frozen model is applied to targets/directions it was never
tuned on** — R/TB/HR broad-mode probabilities are not well-calibrated
under the current frozen model. This is a MARGINAL MODEL bottleneck, not a
JOINT MODEL or PAIR SEARCH bottleneck: the independence joint-probability
mechanism itself is unmodified and was previously shown well-calibrated in
narrow H state (see `manifest.CONCLUSION_REASONING`) — it is the *inputs*
feeding it here that are miscalibrated, and compounding two overconfident
marginals into a product makes the joint overconfidence larger, not
smaller.

### Statistical caution: effective sample size and CIs

Each day has a hard MAX_PER_DAY-style cap of ~25 action-eligible legs (an
existing, unmodified convention inherited from `observation_universe.py`'s
sibling), producing up to C(25,2)=300 pairs/day that are **not
independent draws** — they share legs. The only statistically defensible
resampling unit is the **day** (12 days total here), not the pair. Day-
clustered bootstrap (resample dates, not rows; 3000 resamples):

| slice | n pairs | n days | realized-return point estimate | 90% CI (day-clustered) |
|---|---|---|---|---|
| all evaluated pairs | 3580 | 12 | -0.115 | [-0.301, +0.050] |
| ++ class only | 2294 | 12 | -0.066 (bootstrap mean) | [-0.240, +0.084] |
| **value subset** (`p_joint > market_implied_p`, i.e. pairs where the model disagrees with the market in the profitable direction) | 2294 | 10 | +0.088 (bootstrap mean) | [-0.041, +0.207] |

**Every one of these CIs crosses zero.** With only 12 archived
DEVELOPMENT days (and effectively ~25 legs/day of independent
information), there is not enough data to distinguish a real edge from
noise in either direction, for any of these slices.

**Multiplicity caution on the "value subset" row:** that filter
(`p_joint > market_implied_p`) was constructed *during this exact
exploratory pass*, after seeing the raw pair-class results — it was not
predeclared before looking at DEVELOPMENT outcomes. Per this repo's
established discipline (reject best-looking-after-search results as the
frozen choice; see `h_over_ranker`'s C=1.0 rejection), this row is reported
as a **hypothesis for a future frozen/SELECT-confirmed rule**, not as
confirmed evidence of edge. Treating its positive-leaning point estimate as
a finding would repeat the exact mistake this session's own discipline
exists to avoid.

## Decision at this checkpoint

- Multi-target real price coverage: **confirmed real** (contradicts and
  supersedes the "no real price coverage" half of the prior manifest
  conclusion — `manifest.py` updated accordingly, see below).
- Real, held-out-in-development-window evidence of a positive parlay edge
  under the CURRENT frozen marginal model + independence joint mechanism:
  **not established** — point estimates lean negative-to-flat, CIs cross
  zero at 12 days of data, and the one filter that looks promising
  (market-disagreement subset) is a same-pass hypothesis, not a confirmed
  result.
- This is `INSUFFICIENT_EVIDENCE` in the *same* sense as before, but now
  for a different, better-grounded reason: not "no priced markets exist"
  (false, corrected here) but "the marginal probability model this system
  is built on is not calibrated outside the narrow H-OVER slice it was
  frozen against, and even where it disagrees with the market in the
  hoped-for direction, 12 days is not enough evidence either way."
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
3. Given only 16 usable multi-target days exist, a 3-way split (as strict
   as `h_over_ranker`'s 8/8/9) is likely too thin to be trustworthy here;
   this itself is worth reporting as a DATA-category bottleneck if it
   can't be resolved before the next checkpoint.

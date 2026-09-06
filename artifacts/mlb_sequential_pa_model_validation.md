# MLB Game-Conditioned Sequential PA H/TB/HR Validation

- Run date: `2026-09-06`
- Probability model: `game_conditioned_hitter_moe_v2`
- Structural simulator: `sequential_pa_contact_model_v1`
- Residual fit: `FITTED_EXPANDING_WINDOW_RESIDUAL_MOE_NON_REGRESSION_GATED`
- Evidence class: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`
- Evaluated H/TB/HR rows: **810**
- Modeled rows: **0**
- Blocked rows: **810**
- Data freshness: `FRESH`

## Target authority

| Target | Diagnostic gate | Positive authority | Validation status |
|---|---|---|---|
| H | False | False | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |
| TB | False | False | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |
| HR | False | False | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |

## Data

Sources: `baseball_savant_statcast_via_pybaseball_2_2_7, fangraphs_via_pybaseball_2_2_7`

Baseball Savant / Statcast status: `SUCCESS`

FanGraphs status: `UNAVAILABLE`

Effective as-of date: `2026-09-05`

Profile coverage: 0 batter profiles, 9 pitcher profiles, 0 direct BvP process profiles.

Raw Statcast data are cached by pybaseball and processed same-as-of partitions are cached rather than committed as large raw datasets. Every profile partition is dated and carries source, fetch, and effective timestamps.

## Architecture

`legacy/no-vig prior -> game state -> expert activations -> residual logit -> sequential PA distribution -> target-specific uncertainty/authority gate`

Six experts are evaluated per game: strikeout/contact, contact quality, power/TB/HR, defensive conversion, PA opportunity, and starter-removal/bullpen transition. Global residual coefficients are multiplied by game-specific activations, so a high-K matchup emphasizes contact survival while a low-K matchup can emphasize batted-ball quality. Power relevance increases from H to TB to HR.

The event tree is `PA -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER`, followed by a non-HR contact outcome distribution. PA and AB are tracked separately and later PA transition away from the starter. H, TB and HR probabilities are calculated from their own simulated outcome arrays rather than from point projections.

## Probability authority

The structural simulator directly estimates `P(H>=1)`, `P(TB>=2)` and `P(HR>=1)`. The game-conditioned layer learns a bounded residual around the legacy/no-vig prior in logit space.

A target that fails expanding-window Brier/log-loss validation is shadow-only and cannot change production probability. A target that clears the diagnostic gate may apply conservative negative-only authority only after train/serve feature parity is independently proven. Positive residual authority additionally requires exact point-in-time advanced-feature evidence.

Every live run now writes a content-addressed, outcome-free pregame feature snapshot under the publication history tree. These snapshots preserve the exact live expert state needed to eliminate reconstruction and train/serve skew in future validation.

## Current modeled rows

| Player | Target | Prior P | Conditioned P | Production P | P(0H) | P(HR>=1) | E[PA] | E[H] | E[TB] | Support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Blocked/degraded data

810 rows were fail-closed because required MLBAM identity, Statcast profile, pitcher profile, or freshness evidence was unavailable.

## Validation status

Residual validation uses expanding-window folds. Every held-out block is scored with coefficients fit only on strictly earlier dates. Aggregate Brier and log loss must both improve, at least three folds must exist, and at least 60% of folds must improve both metrics before negative-only authority is considered. Independent authority validation also requires minimum metric gains and train/serve feature parity.

Historical proxy evidence may initialize shadow residuals but cannot unlock positive authority because the processed corpus does not preserve every exact Savant/FanGraphs pregame state. Exact point-in-time evidence is required for promotion.

## Known limitations

- Specific fielder OAA/location assignment remains zero-centered/uncertain when unavailable; no fielder data are fabricated.
- Direct BvP remains heavily shrunk because sample sizes are usually small.
- FanGraphs xFIP/SIERA availability is source-dependent and missingness lowers evidence strength.
- Bullpen identity is still a transition toward neutral relief state until named-reliever distributions are supported.
- Weather currently consumes temperature when present; wind/humidity are not invented when absent.
- Handedness is preserved in game state, but no fixed platoon coefficient is fabricated without split evidence.
- Live pitch-compatibility is preserved in snapshots but cannot authorize a residual until the same feature is represented in training evidence.

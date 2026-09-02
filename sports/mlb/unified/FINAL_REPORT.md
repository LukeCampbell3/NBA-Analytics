# Unified MLB Engine — gated implementation report

All numeric results in this report are **DEVELOPMENT** unless another evidence
state is explicitly stated. Nothing is `CERTIFIED`.

## Architecture implemented

Approved legacy state and real quotes are normalized into `BetCandidate`, then
passed through one decision contract: calibrated probability, uncertainty,
usable probability, break-even comparison, conservative EV, support/role/
identity checks. Only survivors reach bounded and independent 2/3/4 ticket
construction. Same-game joint probability requires common-world masks. One
`unified_predictions.json` drives grouped shadow rendering.

Specialist legacy models remain compatibility inputs. The new compact
`TrajectoryBatch` is the coherent shared-world API and supports team-run/F5/
game queries, player-share allocation, conditional diagnostics and future
event-level extension.

## Milestones

| Milestone | Result |
|---|---|
| M0 audit | Complete |
| M1 schemas/contract | Complete |
| M2 supported-market compatibility | Complete, shadow |
| M3 2/3/4 and SGP framework | Complete, shadow/fail-closed |
| M4 artifact/frontend | Complete, shadow |
| M5 player/team foundation | Complete, development |
| M6 market conditioning | Complete, development diagnostic |
| M7 exotic foundation | Complete, fail-closed |
| M8 validation interfaces | Complete; empirical evidence insufficient |
| M9 production migration | Blocked; legacy remains authoritative |

## Functional current-slate result

| Surface | Candidates/tickets | Result |
|---|---:|---|
| Normalized candidates | 19 | processed |
| Unified singles | 1 | Pete Crow-Armstrong TB over 1.5, -130; usable p 61.38%, edge +4.86 pp, conservative EV +8.60% |
| Rejected candidates | 18 | explicit diagnostics; chiefly missing unified uncertainty/support |
| Qualified 2-leg | 0 | independent abstention |
| Qualified 3-leg | 0 | independent abstention |
| Qualified 4-leg | 0 | independent abstention |
| SGP | 0 | common-world player/game masks or executable combined quote unavailable |

This demonstrates execution semantics, not predictive accuracy.

## Capabilities and blockers

| Market | State | Blocker when not supported |
|---|---|---|
| H, TB, R, RBI, HR, pitcher K | SUPPORTED compatibility input | unified calibration/uncertainty evidence still required for authority |
| Pitcher outs | SHADOW_ONLY | sparse quote/calibration support |
| Moneyline, game total, F5 total | SHADOW_ONLY | aggregate shared worlds exist; unified calibration remains uncertified |
| Team total | DISCOVERY | incomplete exact two-sided/alternate quote coverage |
| Team hits | MODEL_REQUIRED | coherent player-hit allocation/training not established |
| Runs/team runs/Ks/pitches by inning | EVENT_MODEL_REQUIRED | point-in-time event-state model and quotes unavailable |
| Exact PA pitch count | EVENT_IDENTITY_UNAVAILABLE | sportsbook PA ordinal cannot be unambiguously resolved |

Deterministic settlement hooks exist for aggregate markets. Event contracts
block without exact identity.

## Models and dependency results

Team runs are first-class compact trajectory arrays. F5 totals are queried from
the same trajectories as full-game totals. Player share estimates use smoothing
and event allocation enforces team events equal summed player events. Conditional
player/team metrics are implemented, but their evidence state is DEVELOPMENT.

Market conditioning requires a declared identification level, clipped weights
and adequate effective sample size. A single incomplete binary line is retained
as a disagreement diagnostic and cannot authorize a conditioned PMF.

## Validation results

Hit rate, ROI, confidence intervals, drawdown, Brier/log loss, calibration and
bankroll results for the unified strategy are **UNAVAILABLE**: the repository
does not preserve the complete point-in-time probability, calibration, quote
and identity state across a representative unified replay. Current-price
substitution was not used. The validator distinguishes EXACT_POINT_IN_TIME,
RECONSTRUCTED_WITH_VALID_PRIOR_DATA and UNAVAILABLE and excludes unavailable
rows from results.

The bankroll engine supports $100 starting balance with flat $1/$5/$10 and 1%/
2% bankroll staking. Fractional Kelly remains intentionally absent from decision
authority because calibration is uncertified.

## Performance

Seeded aggregate simulation benchmark on the implementation runner:

| Trials/game | Runtime/game | Peak memory | P(game over 8.5) |
|---:|---:|---:|---:|
| 5,000 | 0.0311 s | 1.47 MB | 48.50% |
| 10,000 | 0.0063 s | 1.03 MB | 48.04% |
| 50,000 | 0.0324 s | 5.15 MB | 48.87% |

The 10k-to-50k probability difference was 0.826 pp. A 15-game, 10,000-trial
aggregate slate took 0.0535 s with 1.03 MB measured peak memory. These are
DEVELOPMENT measurements of the aggregate backbone, not a pitch-event model.
There is no comparable repository-native event-simulator baseline, so the 2x
regression rule cannot yet be evaluated for that future layer.

## CI, compatibility and deployment

Unified focused tests cover fail-closed selection, the -4500 trap, bounded and
independent ticket classes, common-world SGP masks, accounting, deterministic
simulation, event-market blockers, settlement, point-in-time validation and
frontend separation. The branch workflow rebuilds and validates the artifact
and benchmarks simulation. The normal daily workflow is wired to generate the
shadow artifact after legacy generation and copy it to all frontend targets.

Current production fallback: all legacy daily/same-game/pitcher/exotic entrypoints
and artifacts. Retained: useful specialist models and their evidence. Deprecated
after a future migration: downstream competing selection authority and separate
primary surfaces. No legacy system is removed in this branch.

Deployment status: **BLOCKED / SHADOW_ONLY**. Required next evidence is a
representative point-in-time comparison, locked validation, prospective shadow
settlement, calibration by market/leg count, and populated browser verification.
Rollback is documented in `MIGRATION_AND_ROLLBACK.md`.

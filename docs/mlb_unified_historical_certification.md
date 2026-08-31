# MLB unified locked historical certification

Result: **HISTORICAL_VALIDATION_FAIL**  
Evidence state: **LOCKED_HISTORICAL_VALIDATION**  
Frozen policy commit: `7c56729a9914eb9f903edffe9ca58b1a0a749ad4`  
Frozen policy hash: `5f8b247e7781717ffb39a01f581dd36c7466f9da061aa7519c5ab6777b73b67b`

## Result

| Measure | Locked result | Predeclared requirement |
|---|---:|---:|
| Eligible independent slates | 0 | 20 |
| Eligible selected singles | 0 | 50 per capability |
| Eligible 2/3/4-leg tickets | 0 / 0 / 0 | 30 per class |

No hit-rate, ROI, calibration, drawdown, discrimination, concentration,
bankroll, baseline, or ablation statistic is reported because the denominator
is zero. `null` is the correct value; manufacturing a reconstructed probability
would test a different policy.

## Evidence inventory

The settled historical universe contains 242,425 candidates across 156 slates.
All rows retain result and structural prediction. Only 6,105 over-side and 2,962
under-side rows retain a price plus quote timestamp, and the corresponding rows
do not retain a verifiable game start timestamp in that corpus. More
importantly, no row retains the frozen unified decision's final/usable
probability, uncertainty, lineup/role validation and exact calibration state.

Git history of `daily_predictions.json` retains one deduplicated exact pregame
frozen-policy candidate from August 30. Its settlement is not preserved in the
repository artifact. It is therefore valid prospective input evidence but not
a gradable locked observation.

## Recovery audit

The initial zero-row result was re-audited against additional committed
sources. The repository contains 134 selected-only high-precision rows across
25 slates, but only 41 have real prices and only 8 preserve the frozen final
probability. It also contains 170,127 immutable full-universe rows across 8
slates; those rows remain unsettled and preserve no confirmed lineup states.
The sources therefore cannot be combined into `RECONSTRUCTED_HIGH_FIDELITY`
records without inventing missing prediction-time state.

The certification engine's unconditional-failure path was removed and tested
with a synthetic corpus that clears every predeclared capability gate. The
current failure is now exclusively an evidence-sufficiency result.

## Capability decision

Aggregate player and game markets remain `VALIDATION_ONLY`. Same-game parlays
remain `SHADOW`. Team hits and inning/PA markets remain `BLOCKED` by their
previously recorded model/data/identity gaps. No capability advances to
`PRODUCTION_CANDIDATE`.

## Gate consequence

The state transition is:

`LOCKED_HISTORICAL_VALIDATION → LOCKED_HISTORICAL_VALIDATION_FAILED`

The state machine therefore prohibits dark production merge, live canary
authority, and activation. The legacy engine remains authoritative.

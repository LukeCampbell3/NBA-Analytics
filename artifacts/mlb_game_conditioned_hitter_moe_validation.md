# MLB Game-Conditioned Hitter MoE Validation

Model: `game_conditioned_hitter_moe_v2`

Evidence class: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`

The model fits a residual in logit space around the legacy probability. Global coefficients are learned on earlier games only; per-game expert activations change each expert's effective weight for the current matchup.

Positive publication authority remains disabled because this historical corpus does not contain exact pregame snapshots of every Savant/FanGraphs feature used live.

| Target | Train | Validation | Prior Brier | Candidate Brier | Prior LogLoss | Candidate LogLoss | Diagnostic gate |
|---|---:|---:|---:|---:|---:|---:|---|
| H | 99 | 301 | 0.2463 | 0.2492 | 0.6909 | 0.6984 | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |
| TB | 99 | 301 | 0.1609 | 0.1582 | 0.4794 | 0.4731 | IMPROVED_DIAGNOSTIC_ONLY |

## Experts

- strikeout/contact compatibility
- contact quality / expected contact
- power and total-base tail
- specific defensive conversion residual
- plate-appearance opportunity
- starter-removal / bullpen transition

Live production additionally uses exact-day Savant pitch-type matchup information, xFIP/SIERA when available, team scoring state, park/weather state, and support/uncertainty shrinkage. This historical fit is a conservative initialization, not certification.

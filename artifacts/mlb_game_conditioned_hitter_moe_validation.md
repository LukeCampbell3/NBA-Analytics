# MLB Game-Conditioned Hitter MoE Validation

Model: `game_conditioned_hitter_moe_v2`

Evidence class: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`

The model fits target-specific residuals in logit space around the legacy/structural prior. Every validation prediction comes from an expanding-window fit using strictly earlier dates.

Positive publication authority remains disabled because this corpus does not preserve exact pregame snapshots of every advanced live feature. A target receives negative-only authority only if aggregate Brier and log-loss improve and at least 60% of expanding-window folds improve both.

| Target | Fit rows | OOF rows | Folds pass | Prior Brier | Candidate Brier | Prior LogLoss | Candidate LogLoss | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H | 300 | 260 | 1/5 | 0.2447 | 0.2479 | 0.6797 | 0.6877 | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |
| TB | 300 | 260 | 5/5 | 0.1656 | 0.1606 | 0.4897 | 0.4785 | IMPROVED_DIAGNOSTIC_ONLY |
| HR | 300 | 260 | 2/5 | 0.0496 | 0.0495 | 0.1792 | 0.1744 | DID_NOT_CLEAR_DIAGNOSTIC_IMPROVEMENT_GATE |

## Experts

- strikeout/contact compatibility
- contact quality / expected contact
- power / total-base / home-run tail
- specific defensive conversion residual
- plate-appearance opportunity
- starter-removal / bullpen transition

Live production additionally uses exact-day Savant pitch-type matchup information, xFIP/SIERA when available, team scoring state, park/weather state, and support/uncertainty shrinkage. This historical fit is diagnostic initialization, not certification.

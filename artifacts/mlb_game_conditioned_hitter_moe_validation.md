# MLB Game-Conditioned Hitter MoE Non-Regression Validation

Model: `game_conditioned_hitter_moe_v2`

Evidence: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`

This report requires the new residual model to beat the prior in rolling-origin probability scoring and to preserve or improve supported pick hit-rate slices after replaying the live negative-only authority rule.

| Target | OOF | Folds pass | Prior Brier | Candidate | Production | Prior LL | Candidate | Production | Pick guard | Authority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H | 1126 | 2/6 | 0.2479 | 0.2492 | 0.2479 | 0.7147 | 0.7171 | 0.7147 | PASS | False |
| TB | 1126 | 2/6 | 0.1502 | 0.1487 | 0.1502 | 0.4514 | 0.4491 | 0.4514 | PASS | False |
| HR | 1126 | 3/6 | 0.0619 | 0.0630 | 0.0619 | 0.2115 | 0.2069 | 0.2115 | PASS | False |

A target that fails any gate has zero production authority, so its production probability is the previous prior unchanged. Positive/bidirectional authority remains disabled until exact point-in-time locked or prospective advanced-feature evidence exists.

No ROI claim is made because this processed-history replay does not preserve exact decision-time prices for every observation.

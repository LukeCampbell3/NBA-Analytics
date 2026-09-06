# MLB Game-Conditioned Hitter MoE Non-Regression Validation

Model: `game_conditioned_hitter_moe_v2`

Evidence: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`

This report requires the new residual model to beat the prior in rolling-origin probability scoring and to preserve or improve supported pick hit-rate slices after replaying the negative-only guard. Passing that diagnostic still does not grant production authority without exact train/serve feature parity.

| Target | OOF | Folds pass | Prior Brier | Candidate | Guarded | Prior LL | Candidate | Guarded | Diagnostic NR | Authority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H | 1126 | 2/6 | 0.2479 | 0.2493 | 0.2479 | 0.7147 | 0.7171 | 0.7147 | False | False |
| TB | 1126 | 2/6 | 0.1502 | 0.1487 | 0.1502 | 0.4514 | 0.4491 | 0.4514 | False | False |
| HR | 1126 | 3/6 | 0.0619 | 0.0630 | 0.0619 | 0.2115 | 0.2069 | 0.2115 | False | False |

Historical diagnostic passes remain shadow-only because the processed corpus does not preserve the exact live pitch-compatibility, direct-matchup, handedness-split, chase/EV, weather, and defense feature state. New content-addressed pregame snapshots are the evidence source for closing that gap.

Positive/bidirectional authority additionally requires locked exact point-in-time certification with the frozen policy hash and sufficient independent slates/selections.

No ROI claim is made because this processed-history replay does not preserve exact decision-time prices for every observation.

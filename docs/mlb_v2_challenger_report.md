# MLB V2.1 challenger

Slate: `MLB_20260830`

Baseline: `52deb038a076b39a1bc840b77ae26648d9e4ffa20194135e7d48b9761edbc611`

Challenger: `80f335d2501d54502909d7f8587ebfef56d725a67d44ae125f9df4337d489b1c`

| Population | Count |
|---|---:|
| Normalized | 19 |
| Fully valid | 0 |
| Admissible | 0 |
| Selected | 0 |

## Rejections

- `CAPABILITY_NOT_SUPPORTED`: 16
- `LINEUP_INVALID`: 18
- `PLAYER_STATUS_INVALID`: 19
- `PROBABILITY_UNAVAILABLE`: 18
- `QUOTE_FRESHNESS_UNPROVABLE`: 19
- `SUPPORT_INVALID`: 18
- `UNCERTAINTY_COMPONENTS_UNAVAILABLE`: 19
- `UNCERTAINTY_INVALID`: 19

## Scientific status

No historical outcome was used to tune V2.1. Current inputs do not preserve quote time, independent player status, measured uncertainty components, or OOD state, so the challenger abstains. Coverage-risk, rank, Top-K, boundary, and uncertainty-discrimination claims remain `INSUFFICIENT_PROSPECTIVE_EVIDENCE` until settled all-candidate slates accumulate.

Parlays remain shadow-only.

## Final decision

NO_RELIABLE_EDGE_FOUND

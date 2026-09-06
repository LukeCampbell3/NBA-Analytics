# Sequential PA historical validation

Evidence class: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`

This is a strict rolling-origin predictive diagnostic using only each player's rows before the evaluated game. It does **not** claim full live-model certification because historical pitch-level xFIP/SIERA/OAA snapshots were not preserved for every replay date.

| Target | Rows | Legacy Brier | Seq raw Brier | Seq usable Brier | Legacy logloss | Seq raw logloss | Legacy MAE | Seq MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | 120 | 0.2611 | 0.2881 | 0.2700 | 0.7203 | 0.7744 | 0.7373 | 0.7240 |
| TB | 120 | 0.1482 | 0.1982 | 0.1776 | 0.4449 | 0.5873 | 0.9476 | 1.3544 |

## Zero-hit calibration

Predicted zero-hit rate: `0.34307`; observed zero-hit rate: `0.5416666666666666` across `120` H observations.

## Economic evidence

No ROI claim is made unless an exact preserved decision-time price and timestamp are present. This processed-history diagnostic does not fabricate historical sportsbook prices.

## Limitations

- Historical processed rows preserve useful rolling hitter/process and opponent-starter fields but not the complete live Statcast/FanGraphs/OAA state now used by production.
- The pitcher portion of this replay is therefore a leakage-safe process proxy, explicitly not a reconstruction of missing historical xFIP/SIERA snapshots.
- Results are predictive diagnostics only and cannot promote the new model out of negative-authority/shadow status.
